import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam64(),
    dataset = dataset.homeoffice_communication_28g(),
    estimator = estimator.PerceiverIO(estimator_pretrained_model= "saved_models/pi_random_16x16_indoor/estimator_epoch3.pth"),
    generator = generator.uniform_16x16(),
    training = training.pre_defined(config_name),
)
