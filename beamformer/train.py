import os
import random
import glob
import pprint
import torch
import numpy as np
from accelerate import Accelerator

from beamformer.utils import count_parameters, scale_in_last_dim, load_config, get_log_path, namespace_to_dict
from beamformer.modules import TransformerModel, get_scheduler_by_type
from beamformer.dataset import load_data_process, load_datasets
from beamformer.weight_generator import ParametricGenerator, transform_weights


class Trainer:
    def __init__(self, config):
        self.config = config
        self.log_path = get_log_path()
        self.M = config.dataset.M
        self.N = config.dataset.N
        self.accelerator = Accelerator(gradient_accumulation_steps=1)
        self.device = self.accelerator.device
        self._set_seeds()
        self.train_dataset, self.test_dataset = load_datasets(config)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            drop_last=True,
        )

    def log(self, message):
        if self.accelerator.is_main_process:
            print(message)
            with open(self.log_path, 'a') as f:
                f.write(message + '\n')

    def initialize_generator(self):
        sample_num = self.config.assumption.sample_num
        generator = ParametricGenerator(sample_num, self.M, self.N)
        if self.config.generator.generator_pretrained_model:
            generator.load_state_dict(torch.load(self.config.generator.generator_pretrained_model, weights_only=True))
            self.log(f"Generator loaded from {self.config.generator.generator_pretrained_model}")
        if self.accelerator.is_main_process:
            count_parameters(generator, "Generator")
        return generator

    def initialize_estimator(self):
        cfg = self.config.estimator
        estimator = TransformerModel(estimator_config=cfg)
        if cfg.estimator_pretrained_model:
            estimator.load_state_dict(torch.load(cfg.estimator_pretrained_model, weights_only=True))
            self.log(f"Estimator loaded from {cfg.estimator_pretrained_model}")
        if self.accelerator.is_main_process:
            count_parameters(estimator, "Estimator")
        return estimator.to(self.device)

    def save_model(self, model, name):
        path = os.path.join(self.config.training.model_save_path, f"{name}.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_files = sorted(
            glob.glob(os.path.join(self.config.training.model_save_path, "*.pth")),
            key=os.path.getmtime
        )
        if len(model_files) >= 2:
            oldest_file = model_files[0]
            os.remove(oldest_file)
            self.log(f"Deleted old model: {oldest_file}")
        self.accelerator.save(model.state_dict(), path)
        self.log(f"Model saved at {path}")

    def _check_model_exist(self, path, size_kb=10):
        if not os.path.exists(path) or not os.path.isdir(path):
            return False
        total_size = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    continue
        return total_size > size_kb * 1024

    def _forward_pass_and_loss(self, csi, generator, estimator, dp, z):
        raw_weights = generator(z)
        weights_out, _ = transform_weights(raw_weights)

        sample_pos_enc = dp.generate_sample_position_encoding(weights_out)
        query_pos_enc = dp.generate_query_position_encoding(batch_size=self.config.training.batch_size)
        sample_rss = dp.generate_sample_rss(csi, weights_out)
        query_rss = dp.generate_query_rss(csi)

        scale = torch.max(sample_rss, dim=1, keepdim=True).values
        sample_rss /= scale
        query_rss /= scale

        sample_rss = sample_rss.to(dtype=torch.float32)
        query_rss = query_rss.to(dtype=torch.float32)
        sample_pos_enc = sample_pos_enc.to(dtype=torch.float32)
        query_pos_enc = query_pos_enc.to(dtype=torch.float32)

        label_gt, _ = scale_in_last_dim(query_rss)
        pred_rss, _ = estimator(sample_rss, sample_pos_enc, query_pos_enc)
        loss, as_loss_db = self._loss_fun(label_gt, pred_rss)
        return loss, as_loss_db

    def _loss_fun(self, tensor_gt, tensor_pred):
        mse = torch.nn.MSELoss()
        mse_loss_as = mse(tensor_gt, tensor_pred)
        mse_loss_as_db = 10 * torch.log10(mse_loss_as + 1e-16)
        return mse_loss_as, mse_loss_as_db

    def train(self):
        if self._check_model_exist(self.config.training.model_save_path):
            raise FileExistsError("You Use An Existing Folder as Save Path")

        generator = self.initialize_generator()
        estimator = self.initialize_estimator()
        generator.train()
        estimator.train()

        optimizer_estimator = torch.optim.AdamW(
            estimator.parameters(), lr=self.config.training.lr_estimator, weight_decay=0.01
        )
        optimizer_generator = torch.optim.AdamW(
            generator.parameters(), lr=self.config.training.lr_generator, weight_decay=0.01
        )

        total_steps = self.config.training.epochs * len(self.train_dataloader) / self.config.training.gpu_num
        warmup_steps = int(self.config.training.warmup_ratio * total_steps)
        scheduler_estimator = get_scheduler_by_type(optimizer_estimator, warmup_steps, total_steps)
        scheduler_generator = get_scheduler_by_type(optimizer_generator, warmup_steps, total_steps)

        train_dataloader, optimizer_generator, optimizer_estimator, generator, estimator = self.accelerator.prepare(
            self.train_dataloader, optimizer_generator, optimizer_estimator, generator, estimator
        )

        dp = load_data_process(self.config, device=self.device)

        for epoch in range(self.config.training.epochs):
            epoch_loss = 0.0
            for batch_idx, (csi, _) in enumerate(train_dataloader):
                loss, as_loss_db = self._co_train_step(
                    csi, generator, estimator, dp,
                    optimizer_estimator, scheduler_estimator,
                    optimizer_generator, scheduler_generator,
                )
                epoch_loss += loss
                self.log(
                    f"Epoch [{epoch+1}] Batch [{batch_idx}]"
                    f" | EstLR: {scheduler_estimator.get_last_lr()[0]:.2e}"
                    f" | GenLR: {scheduler_generator.get_last_lr()[0]:.2e}"
                    f" | Loss: {loss:.6f} | AS Loss (dB): {as_loss_db:.3f}"
                )

            avg_loss = epoch_loss / (len(self.train_dataloader) * self.config.training.gpu_num)
            self.log(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.6f}")

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                epoch_tag = "epoch_final" if (epoch + 1) == self.config.training.epochs else f"epoch{epoch + 1}"
                self.save_model(self.accelerator.unwrap_model(estimator), f"estimator_{epoch_tag}")
                self.save_model(self.accelerator.unwrap_model(generator), f"generator_{epoch_tag}")

        print("Training complete.")

    def _co_train_step(self, csi, generator, estimator, dp,
                       optimizer_estimator, scheduler_estimator,
                       optimizer_generator, scheduler_generator):
        z_dim = self.config.assumption.sample_num * self.config.dataset.M * self.config.dataset.N
        z = torch.randn(self.config.training.batch_size, z_dim).to(self.device)
        loss, as_loss_db = self._forward_pass_and_loss(csi, generator, estimator, dp, z)
        optimizer_estimator.zero_grad()
        optimizer_generator.zero_grad()
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(estimator.parameters(), 0.5)
        optimizer_estimator.step()
        optimizer_generator.step()
        scheduler_estimator.step()
        scheduler_generator.step()
        return loss.item(), as_loss_db.item()

    def _set_seeds(self):
        seed = self.config.training.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if self.accelerator.is_main_process:
            config_dict = namespace_to_dict(self.config)
            config_str = pprint.pformat(config_dict, indent=4, width=100)
            self.log("========== Configuration ==========\n" + config_str + "\n==================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training Script with Config Support")
    parser.add_argument("--config", type=str, default=None, help="name of config file under configs")
    args = parser.parse_args()
    print(f"You're using config: {args.config}")
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.train()
