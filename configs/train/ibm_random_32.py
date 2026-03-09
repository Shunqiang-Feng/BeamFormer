import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam32(),
    dataset = dataset.ibm_indoor(),
    estimator = estimator.PerceiverIO(),
    generator = generator.ibm_random_8x8(),
    training = training.pre_defined(config_name),
)
config.training.epochs = 3
