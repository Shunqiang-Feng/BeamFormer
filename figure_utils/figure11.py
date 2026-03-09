"""
Figure 11: Visualization of layer-wise beam spectrum generation.
Polar heatmap showing how the model generates beam spectra at each processing layer.
"""
import os
import sys
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

REQUIRED_CONFIGS = []  # special case - uses model directly

DEFAULT_INDICES = [6, 204]


def plot(data, save_path):
    """
    Plot Figure 11: Multi-row polar heatmap of beam spectra at each processing layer.

    Args:
        data: list of (gt_tensor, [8 layer tensors]) - one tuple per sample row
        save_path: path to save the resulting figure
    """
    # Import here to avoid top-level import failures if dependencies are missing
    from beamformer.info_flow import plot_multi_row_comparison

    plot_multi_row_comparison(
        data_rows=data,
        col_title="Normalized RSS",
        show_angle_lines=False,
        show_peak_coordinates=False,
        save_path=save_path,
        show_colorbar=False,
        width_ratio=4
    )


def generate_spectra(indices=None):
    """
    Generate beam spectra by running model inference.

    Args:
        indices: list of dataset indices to process. Defaults to DEFAULT_INDICES.

    Returns:
        list of (gt_tensor, [8 layer tensors])
    """
    if indices is None:
        indices = DEFAULT_INDICES

    from beamformer.dataset import load_datasets, load_data_process
    from beamformer.utils import load_config
    from beamformer.weight_generator import ParametricGenerator, transform_weights
    import beamformer.utils as utils

    from beamformer.info_flow import TransformerModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = load_config("pi_param_co_train_indoor")
    dp = load_data_process(config, device)
    _, csi_dataset = load_datasets(config)

    # Resolve saved_models path
    saved_models_dir = os.path.join(_PROJECT_ROOT, "saved_models", "pi_param_co_train_indoor")
    generator_path = os.path.join(saved_models_dir, "generator_epoch_final.pth")
    estimator_path = os.path.join(saved_models_dir, "estimator_epoch_final.pth")

    if not os.path.exists(generator_path):
        raise FileNotFoundError(
            f"Generator model not found: {generator_path}\n"
            f"Please ensure saved_models/pi_param_co_train_indoor/ contains the model files."
        )
    if not os.path.exists(estimator_path):
        raise FileNotFoundError(
            f"Estimator model not found: {estimator_path}\n"
            f"Please ensure saved_models/pi_param_co_train_indoor/ contains the model files."
        )

    generator = ParametricGenerator(64, 16, 16)
    generator.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
    generator.eval()

    model = TransformerModel(config.estimator)
    model.load_state_dict(torch.load(estimator_path, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    z = torch.randn(1, 64 * 16 * 16).to(device)
    raw_weights = generator(z)
    weight, _ = transform_weights(raw_weights)
    weight = weight.to(device)

    all_rows_data = []
    for _, idx in enumerate(indices):
        print(f"\n--- Processing index: {idx} ---")

        csi = csi_dataset[idx][0].unsqueeze(0).to(device)

        sample_rss = dp.generate_sample_rss(csi, weight).to(dtype=torch.float32)
        sample_pos_enc = dp.generate_sample_position_encoding(weight).to(dtype=torch.float32)
        query_pos_enc = dp.generate_query_position_encoding(1).to(dtype=torch.float32)
        scale = torch.max(sample_rss, dim=1, keepdim=True).values
        sample_rss = sample_rss / scale

        # Ground truth
        print(f"Generating GT for idx={idx}...")
        full_angles_spectrum = dp.generate_query_rss(csi)
        full_angles_spectrum = full_angles_spectrum / scale
        full_angles_spectrum, _ = utils.scale_in_last_dim(full_angles_spectrum)
        gt_tensor_for_plot = full_angles_spectrum.reshape(80, 20).detach().cpu()

        # Layer-wise processed spectra
        print(f"Generating 8 processed spectra for idx={idx}...")
        processed_tensors_list = []
        for eff_layers in range(1, 9):
            angles_spectrum, _ = model(sample_rss, sample_pos_enc, query_pos_enc, eff_layers=eff_layers)
            current_spectrum = angles_spectrum.reshape(80, 20).detach().cpu()
            processed_tensors_list.append(current_spectrum)

        all_rows_data.append((gt_tensor_for_plot, processed_tensors_list))

    return all_rows_data
