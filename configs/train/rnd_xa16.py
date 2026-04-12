import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam64(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g(),
    estimator = estimator.PerceiverIO(depth=4, dim=512, latent_dim=512),
    generator = generator.random_16x16(),
    training = training.pre_defined(config_name),
)
config.training.epochs = 3
