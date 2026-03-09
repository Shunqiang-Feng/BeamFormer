import os
from types import SimpleNamespace
from configs.submodules import ARN_model, ARN_training, assumption, dataset, estimator, generator


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam16_sivers(),
    dataset = dataset.sivers(add_noise = True,  snr_min = -3),
    generator = generator.parametric_generator("./saved_models/sivers_pi_param_co_train_16_snr_worse/generator_epoch_final.pth"),
    arn_model = ARN_model.typical_ARN(),
)

config = ARN_training.add_train_ARN(config)
config.training.batch_size *= 2