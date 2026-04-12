import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from .dataset import load_datasets, load_data_process
from .modules import AmplitudeRecoveryNetwork, FastTransformerModel, ArrayAdapter, CNNGeneratorModel, MLPAutoEncoderModel
from .weight_generator import PredefinedGenerator, ParametricGenerator, transform_weights, transform_weights_full_array
from .utils import scale_in_last_dim, count_parameters, calculate_power_db, gpu_tensor_to_np, nmse_loss_db, get_db, visualize_pred_vs_gt, interpolate_angle_spectrum, find_closest_in_angle_spectrum
import shutil
from .related_work import load_related_work, cache_max_rss
import warnings
from .peak_scale_utils import PeakScaleGenerator


class Evaluator:
    def __init__(self, config, peak_scale_mean=None, peak_scale_std=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = config.request.results_folder
        os.makedirs(self.results_dir, exist_ok=True)

        self.ml_generator_list = ["co-train"]

        if peak_scale_mean is not None and peak_scale_std is not None:
            self.peak_scale_generator = PeakScaleGenerator(
                mean=peak_scale_mean,
                std=peak_scale_std,
                distribution="normal",
                device="cpu",
            )
            print(f"[Info] Initialized PeakScaleGenerator: mean={peak_scale_mean:.4f}, std={peak_scale_std:.4f}")
        else:
            self.peak_scale_generator = None

        self.related_work_dict = load_related_work(self.config.include_related_work)
        self.all_performance_dict = self.load_models_performance()
        self.all_performance_dict.update(self.related_work_dict)

    def initialize_generator(self, setting):
        sample_num = setting.assumption.sample_num
        ds = setting.dataset
        if setting.scheme in self.ml_generator_list:
            if setting.generator.type == "PARAM":
                generator = ParametricGenerator(sample_num, ds.M, ds.N)
            else:
                raise ValueError("Setting Scheme is not supported")
            if hasattr(setting.generator, "generator_pretrained_model") and setting.generator.generator_pretrained_model:
                generator.load_state_dict(torch.load(setting.generator.generator_pretrained_model, map_location=self.device, weights_only=True))
                print(f"Loaded generator from {setting.generator.generator_pretrained_model}")
            generator.eval()
            generator.to(self.device)
            count_parameters(generator, "Generator")
        elif setting.scheme == "pre-defined":
            generator = PredefinedGenerator(
                M_act=setting.generator.M_act,
                N_act=setting.generator.N_act,
                sample_mode=setting.generator.type,
                sample_num=sample_num,
                M_base=ds.M,
                N_base=ds.N,
                start_freq=ds.start_freq,
                end_freq=ds.end_freq,
                angle_steps_theta=setting.assumption.angle_steps_theta,
                angle_steps_phi=setting.assumption.angle_steps_phi,
                freq_num=ds.freq_num,
                d_col=ds.d_col,
                d_row=ds.d_row,
                max_theta=ds.max_theta,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported training scheme: {setting.scheme}")
        return generator

    def _prepare_setting_dir(self, setting_name):
        setting_dir = os.path.join(self.results_dir, setting_name)
        if os.path.exists(setting_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{setting_dir}_backup_{timestamp}"
            shutil.move(setting_dir, backup_dir)
            print(f"[Info] Existing result folder '{setting_name}' moved to backup '{backup_dir}'")
        os.makedirs(setting_dir, exist_ok=True)

    def initialize_estimator(self, setting):
        cfg = setting.estimator
        estimator_type = cfg.type.lower()
        if estimator_type == "cnn_generator":
            cfg.angle_steps_phi   = setting.assumption.angle_steps_phi
            cfg.angle_steps_theta = setting.assumption.angle_steps_theta
            cfg.sample_num        = setting.assumption.sample_num
            cfg.max_theta         = setting.dataset.max_theta
            estimator = CNNGeneratorModel(estimator_config=cfg)
        elif estimator_type == "mlp_ae":
            cfg.angle_steps_phi   = setting.assumption.angle_steps_phi
            cfg.angle_steps_theta = setting.assumption.angle_steps_theta
            estimator = MLPAutoEncoderModel(estimator_config=cfg)
        else:
            estimator = FastTransformerModel(cfg)
        estimator.load_state_dict(torch.load(cfg.estimator_pretrained_model, map_location=self.device, weights_only=True))
        print(f"Loaded estimator from {cfg.estimator_pretrained_model}")
        estimator.to(self.device)
        count_parameters(estimator, "Estimator")
        return estimator

    def init_arn(self, setting_config):
        cfg = setting_config.arn_model
        sample_num = setting_config.assumption.sample_num
        arn_model = AmplitudeRecoveryNetwork(sample_num=sample_num, d_model=cfg.hidden_dims)
        arn_model.has_pretrained = False
        if cfg.ARN_model_pretrained_model is not None:
            arn_model.load_state_dict(torch.load(cfg.ARN_model_pretrained_model, map_location=self.device, weights_only=True))
            arn_model.has_pretrained = True
            print(f"Loaded ARN model from {cfg.ARN_model_pretrained_model}")
        else:
            warnings.warn(
                "You are not using any ARN model, amplitude will be set to 1.",
                RuntimeWarning
            )
        count_parameters(arn_model, "ARN")
        arn_model.to("cpu")
        return arn_model

    def compute(self, estimator, weights, csi, dp, array_adapter=None, grid_indices=None):
        """
        Compute the RSS predictions and related metrics.
        returns:
            sample_rss: [batch_size, sample_num]
            scale: [batch_size, 1]
            query_rss: [batch_size, angle_spectrum_length]
            query_rss_pred: [batch_size, angle_spectrum_length]
        """
        sample_pos_enc = dp.generate_sample_position_encoding(weights).to(dtype=torch.float32)
        query_pos_enc = dp.generate_query_position_encoding(batch_size=csi.shape[0]).to(dtype=torch.float32)
        sample_rss = dp.generate_sample_rss(csi, weights).to(dtype=torch.float32)
        query_rss = dp.generate_query_rss(csi).to(dtype=torch.float32)
        scale = torch.max(sample_rss, dim=1, keepdim=True).values
        sample_rss /= scale
        query_rss /= scale

        if array_adapter is not None:
            sample_pos_enc = array_adapter(sample_pos_enc)
            query_pos_enc = array_adapter(query_pos_enc)

        sample_pos_encoding, query_pos_encoding = estimator.prepare_positional_encoding(sample_pos_enc, query_pos_enc)
        query_rss_pred, _ = estimator(sample_rss, sample_pos_encoding, query_pos_encoding, grid_indices)

        return sample_rss, scale, query_rss, query_rss_pred

    def analyse_my_model(self, arn_model, sample_rss, scale, query_rss, query_rss_pred, criterion, csi, angle_steps_phi=80, angle_steps_theta=20, gt_phi=None, gt_theta=None):
        """
        Analyse the model's performance and calculate various metrics.

        Returns:
            Core metrics, plus random_peak_lobe and pred_rss_at_gt_random_peak_db
            when peak_scale_generator is configured (None otherwise).
        """
        csi = csi.unsqueeze(0)

        query_rss_scaled, actual_scale = scale_in_last_dim(query_rss)
        compute_scale_loss = get_db(criterion(query_rss_pred, query_rss_scaled).item())

        if hasattr(arn_model, 'has_pretrained') and arn_model.has_pretrained:
            pred_amp_inv = arn_model(sample_rss.unsqueeze(0))
            pred_amp_inv = torch.mean(pred_amp_inv).item()
            pred_lobe = 1 / pred_amp_inv
        else:
            pred_lobe = 1.0

        measure = sample_rss * scale
        original_angle_spectrum = query_rss.reshape(angle_steps_phi, angle_steps_theta) * scale
        pred_angle_spectrum = query_rss_pred.reshape(angle_steps_phi, angle_steps_theta) * pred_lobe * scale

        pred_angle_spectrum_high_res = interpolate_angle_spectrum(gpu_tensor_to_np(query_rss_pred.reshape(angle_steps_phi, angle_steps_theta))) # 360, 91
        phi_idx, theta_idx = np.unravel_index(np.argmax(pred_angle_spectrum_high_res), pred_angle_spectrum_high_res.shape)
        theta_max, phi_max = theta_idx * 90 / 90, phi_idx * 359 / 359

        nmse_loss = nmse_loss_db(original_angle_spectrum, pred_angle_spectrum)
        compute_loss = get_db(criterion(original_angle_spectrum, pred_angle_spectrum).item())
        measure_max, pred_max, pred_max_in_gt = calculate_power_db(
            gpu_tensor_to_np(measure),
            gpu_tensor_to_np(original_angle_spectrum),
            gpu_tensor_to_np(pred_angle_spectrum)
        )

        if gt_phi is not None and gt_theta is not None:
            pred_rss_at_gt_normalized = find_closest_in_angle_spectrum(
                gt_phi, gt_theta,
                query_rss_pred.reshape(angle_steps_phi, angle_steps_theta),
                max_theta=90
            )
            pred_rss_at_gt = pred_rss_at_gt_normalized * pred_lobe * scale
            pred_rss_at_gt_db = get_db(pred_rss_at_gt)

            if self.peak_scale_generator is not None:
                random_peak_scale_rec = self.peak_scale_generator.generate(batch_size=1, sample_num=sample_rss.shape[0])
                random_peak_lobe = random_peak_scale_rec[0, 0].item()
                pred_rss_at_gt_random_peak_db = get_db(pred_rss_at_gt_normalized * random_peak_lobe * scale)
            else:
                random_peak_lobe = None
                pred_rss_at_gt_random_peak_db = None
        else:
            raise ValueError("GT direction not found")

        actual_scale_db = get_db(actual_scale)
        pred_lobe_db = get_db(pred_lobe)
        scale_pred_error_db = actual_scale_db - pred_lobe_db

        return (nmse_loss, compute_loss, measure_max, pred_max, pred_max_in_gt,
                compute_scale_loss, pred_lobe, actual_scale, scale_pred_error_db,
                theta_max, phi_max, pred_rss_at_gt_db,
                random_peak_lobe, pred_rss_at_gt_random_peak_db)

    def evaluate_one_step(self, dp, generator, estimator, arn_model, setting, csi_tensor, criterion, csi_path_batch, max_rss_performance, visualize=False, save_dir=None, array_adapter=None):
        ds = setting.dataset
        csi = csi_tensor.to(self.device)
        batch_size = csi.shape[0]

        result = []

        with torch.no_grad():
            if setting.scheme == "pre-defined":
                weights_out, active_antenna = generator.generate(batch_size)
                grid_indices = getattr(generator, '_last_grid_indices', None)
            else:
                z = torch.randn(batch_size, setting.assumption.sample_num * ds.M * ds.N).to(self.device)
                raw_weights = generator(z)
                if getattr(setting, 'hardware', False):
                    weights_out, _ = transform_weights_full_array(raw_weights)
                else:
                    weights_out, _ = transform_weights(raw_weights)
                grid_indices = None

            sample_rss, scale, query_rss, query_rss_pred = self.compute(estimator, weights_out, csi, dp, array_adapter=array_adapter, grid_indices=grid_indices)
            sample_rss, query_rss, query_rss_pred = sample_rss.cpu(), query_rss.cpu(), query_rss_pred.cpu()

            for batch_idx in range(batch_size):
                if isinstance(csi_path_batch, str):
                    csi_path_batch = [csi_path_batch]
                csi_path = csi_path_batch[batch_idx]

                gt_row = max_rss_performance[max_rss_performance['csi_path'] == csi_path]
                if not gt_row.empty:
                    gt_phi = gt_row['gt_phi'].values[0]
                    gt_theta = gt_row['gt_theta'].values[0]
                else:
                    raise ValueError("CSI Path not found")

                phi_steps   = setting.assumption.angle_steps_phi
                theta_steps = setting.assumption.angle_steps_theta
                (nmse_loss, compute_loss, measure_max, pred_max, pred_max_in_gt,
                 compute_scale_loss, pred_lobe, actual_scale, scale_pred_error_db,
                 theta_max, phi_max, pred_rss_at_gt_db,
                 random_peak_lobe, pred_rss_at_gt_random_peak_db) = self.analyse_my_model(
                    arn_model,
                    sample_rss[batch_idx, :], scale[batch_idx].item(),
                    query_rss[batch_idx, :], query_rss_pred[batch_idx, :],
                    criterion, csi[batch_idx, :, :, :],
                    angle_steps_phi=phi_steps, angle_steps_theta=theta_steps,
                    gt_phi=gt_phi, gt_theta=gt_theta
                )

                if visualize and save_dir:
                    title = f"Setting: {setting.name}, Index: {os.path.basename(csi_path)}, AS Loss: {compute_scale_loss:.4f}"
                    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                    csi_stem = os.path.splitext(os.path.basename(csi_path))[0]
                    save_path = os.path.join(save_dir, setting.name, f"{time_now}_{csi_stem}.png")
                    if getattr(self.config.request, 'noise_gt', False):
                        gt_vis = torch.zeros_like(query_rss[batch_idx, :])
                    else:
                        gt_vis = query_rss[batch_idx, :]
                    visualize_pred_vs_gt(
                        query_rss_pred[batch_idx, :].reshape(phi_steps, theta_steps),
                        gt_vis.reshape(phi_steps, theta_steps),
                        save_path=save_path, title_adder=title,
                        show_angle_labels= False,
                    )

                entry = {
                    "csi_path": csi_path,
                    "my_rss": float(pred_max_in_gt),
                    "abs_loss": float(compute_loss),
                    "relative_loss": float(compute_scale_loss),
                    "nmse_loss": float(nmse_loss),
                    "measure_max": float(measure_max),
                    "pred_max": float(pred_max),
                    "pred_lobe": float(pred_lobe),
                    "actual_scale": float(actual_scale),
                    "scale_pred_error_db": float(scale_pred_error_db),
                    "pred_theta": theta_max,
                    "pred_phi": phi_max,
                    "rss_at_gt": pred_rss_at_gt_db,
                }
                if random_peak_lobe is not None:
                    entry["random_peak_lobe"] = float(random_peak_lobe)
                    entry["rss_at_gt_random_peak"] = pred_rss_at_gt_random_peak_db
                result.append(entry)

        return result

    def evaluate_batch(self, setting):
        print(f"\n[Batch Evaluation] {setting.name}")
        batch_size = self.config.request.batch_size
        dp = load_data_process(setting, device=self.device)

        generator, estimator, arn_model, test_dataset, array_adapter = self._initial_setting(setting)
        criterion = torch.nn.MSELoss()

        test_len = self.config.request.test_batch_length or len(test_dataset)
        test_len = min(test_len, len(test_dataset))
        step = max(1, len(test_dataset) // test_len)
        indices = [i * step for i in range(test_len)]

        results = []
        save_folder = os.path.join(self.results_dir, setting.name)
        os.makedirs(save_folder, exist_ok=True)
        max_rss_cache = cache_max_rss(setting.dataset)
        max_rss_performance = max_rss_cache.performance

        for i in range(0, len(indices), batch_size):
            print(f"Batch {i//batch_size + 1}/{(len(indices) + batch_size - 1)//batch_size} - samples {i+1}-{min(i+batch_size, len(indices))}")
            batch_indices = indices[i:i+batch_size]
            batch_csi_tensor = torch.stack([test_dataset[idx][0] for idx in batch_indices])
            batch_csi_path = [test_dataset[idx][1] for idx in batch_indices]

            result = self.evaluate_one_step(
                dp, generator, estimator, arn_model, setting, batch_csi_tensor,
                criterion, batch_csi_path, max_rss_performance,
                visualize=False, save_dir=save_folder, array_adapter=array_adapter
            )
            results.extend(result)

        df = pd.DataFrame(results)
        csv_save_path = os.path.join(save_folder, "metrics.csv")
        df.to_csv(csv_save_path, index=False, float_format="%.16f")

        return df

    def initialize_array_adapter(self, setting):
        cfg = setting.array_adapter
        adapter = ArrayAdapter(cfg.input_array_sampling, cfg.output_array_sampling)
        if getattr(cfg, 'pretrained_model', None):
            adapter.load_state_dict(torch.load(cfg.pretrained_model, map_location=self.device, weights_only=True))
            print(f"Loaded array adapter from {cfg.pretrained_model}")
        adapter.eval()
        adapter.to(self.device)
        count_parameters(adapter, "ArrayAdapter")
        return adapter

    def _initial_setting(self, setting):
        generator = self.initialize_generator(setting)
        estimator = self.initialize_estimator(setting)
        arn_model = self.init_arn(setting)
        _, test_dataset = load_datasets(setting)

        if setting.scheme in self.ml_generator_list:
            generator.eval()
        estimator.eval()
        arn_model.eval()

        array_adapter = None
        if hasattr(setting, 'array_adapter') and setting.array_adapter is not None:
            array_adapter = self.initialize_array_adapter(setting)

        return generator, estimator, arn_model, test_dataset, array_adapter

    def visualize_figures(self, setting):
        print(f"\n[Visualization] {setting.name}")
        generator, estimator, arn_model, test_dataset, array_adapter = self._initial_setting(setting)
        criterion = torch.nn.MSELoss()
        dp = load_data_process(setting, device=self.device)

        figure_count = self.config.request.figures
        step = max(1, len(test_dataset) // figure_count)
        indices = [i * step for i in range(figure_count)]
        max_rss_cache = cache_max_rss(setting.dataset)
        max_rss_performance = max_rss_cache.performance

        for idx in indices:
            csi_tensor = test_dataset[idx][0].unsqueeze(0)
            csi_path = test_dataset[idx][1]
            self.evaluate_one_step(
                dp, generator, estimator, arn_model, setting, csi_tensor,
                criterion, csi_path, max_rss_performance,
                visualize=True, save_dir=self.results_dir, array_adapter=array_adapter
            )

    def infer_examples(self):
        for model_setting in self.config.models:
            self.visualize_figures(model_setting)

    def visualize_one_file(self, target_filename):
        """Visualize a single CSI sample identified by its filename (basename only)."""
        for setting in self.config.models:
            generator, estimator, arn_model, test_dataset, array_adapter = self._initial_setting(setting)
            criterion = torch.nn.MSELoss()
            dp = load_data_process(setting, device=self.device)
            max_rss_cache = cache_max_rss(setting.dataset)
            max_rss_performance = max_rss_cache.performance

            idx = next(
                (i for i in range(len(test_dataset))
                 if os.path.basename(test_dataset[i][1]) == target_filename),
                None,
            )
            if idx is None:
                raise FileNotFoundError(
                    f"{target_filename} not found in test dataset for model '{setting.name}'"
                )

            csi_tensor = test_dataset[idx][0].unsqueeze(0)
            csi_path   = test_dataset[idx][1]
            self.evaluate_one_step(
                dp, generator, estimator, arn_model, setting, csi_tensor,
                criterion, csi_path, max_rss_performance,
                visualize=True, save_dir=self.results_dir, array_adapter=array_adapter,
            )
            print(f"[Visualization] {setting.name} — {target_filename} ✓")

    def load_models_performance(self):
        models_performance_dict = {}
        for model_setting in self.config.models:
            current_model_performance = self.evaluate_batch(model_setting)
            max_rss_cache = cache_max_rss(model_setting.dataset)
            models_performance_dict[model_setting.name] = pd.merge(
                max_rss_cache.performance, current_model_performance, on="csi_path"
            )
            models_performance_dict[model_setting.name]["loss"] = (
                models_performance_dict[model_setting.name]["max_rss"] -
                models_performance_dict[model_setting.name]["my_rss"]
            )
        return models_performance_dict

    def run(self):
        """Main execution method for the evaluator."""
        if self.peak_scale_generator is not None:
            print(f"Peak Scale Generator: {self.peak_scale_generator}")
        print("Starting evaluation...")
        self.infer_examples()
        print("Evaluation completed.")


if __name__ == "__main__":
    from .utils import load_config
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation Script with Config Support")
    parser.add_argument("--config", type=str, default="ideal_co-train_0",
                        help="name of config file under configs")
    parser.add_argument("--peak_mean", type=float, default=None,
                        help="Peak scale mean for random peak evaluation (indoor default: 26.0940, outdoor: 31.8946)")
    parser.add_argument("--peak_std", type=float, default=None,
                        help="Peak scale std for random peak evaluation (indoor default: 11.4801, outdoor: 12.2739)")

    args = parser.parse_args()

    config_name = args.config
    print(f"Using config: {config_name}")

    config = load_config(config_name, predix="evaluate")
    evaluator = Evaluator(config, peak_scale_mean=args.peak_mean, peak_scale_std=args.peak_std)
    evaluator.run()
