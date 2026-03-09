from types import SimpleNamespace
import os

def pre_defined(local_config_name):
    training=SimpleNamespace(
        batch_size=120,
        warmup_ratio=0.05,
        learning_rate=1e-5,
        lr_generator=1e-4,
        lr_estimator=1e-5,
        random_seed=42,
        model_save_path=os.path.join("saved_models", local_config_name),
        epochs=30,
        scheme='pre-defined', # co-train, pre-defined        
        num_workers=0,
        gpu_num=4,
    )

    return training

def co_train(local_config_name):
    training=SimpleNamespace(
        batch_size=120,
        warmup_ratio=0.05,
        learning_rate=1e-5,
        lr_generator=1e-4,
        lr_estimator=1e-5,
        random_seed=42,
        model_save_path=os.path.join("saved_models", local_config_name),
        epochs=30,
        scheme='co-train', # co-train, pre-defined
        num_workers=0,
        gpu_num=4,
    )

    return training

