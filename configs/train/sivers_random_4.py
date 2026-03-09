import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam4_sivers(),
    dataset = dataset.sivers(),
    estimator = estimator.PerceiverIO(),
    generator = generator.sivers_random_4x4(),
    training = training.pre_defined(config_name),
)
config.training.epochs = 3
