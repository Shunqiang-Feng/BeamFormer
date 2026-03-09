import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam64(),
    dataset = dataset.uva_communication_28g(), # uva_communication_28g(),
    estimator = estimator.PerceiverIO(dim=512, latent_dim = 512), 
    generator = generator.random_16x16(),
    training = training.pre_defined(config_name),
)
config.training.epochs = 3