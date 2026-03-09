import os
from types import SimpleNamespace
from configs.submodules import ARN_model, ARN_training, assumption, dataset, estimator, generator


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam64(),
    dataset = dataset.uva_communication_28g(),
    generator = generator.uniform_16x16(),
    arn_model = ARN_model.typical_ARN(),
)

config = ARN_training.add_train_ARN(config)
config.training.batch_size *= 2