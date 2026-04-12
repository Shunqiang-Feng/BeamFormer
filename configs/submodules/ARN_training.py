from types import SimpleNamespace
import os
import copy


def add_train_ARN(config_original, config_name=None):
    """
    Add training configuration for ARN model

    Args:
        config_original: Original configuration object
        config_name: Optional config name to use for model_save_path.
                    If provided, uses config_name directly.
                    If None, generates name from generator and assumption.

    Returns:
        Updated config with training parameters
    """
    config = copy.deepcopy(config_original)

    # Use config_name if provided, otherwise generate from generator and assumption
    if config_name is not None:
        model_name = config_name
    else:
        raise ValueError("You need input config name")

    config.training = SimpleNamespace(
        batch_size = 400,
        warmup_ratio=0.05,
        learning_rate=1e-5,
        random_seed=42,
        model_save_path=os.path.join("ARN_saved_models", model_name),
        epochs=10,
        num_workers=0,
        gpu_num=4,
    )
    return config




