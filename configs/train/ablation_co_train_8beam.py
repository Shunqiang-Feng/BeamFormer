import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training



config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam8(),
    dataset = dataset.uva_communication_28g(),
    estimator = estimator.PerceiverIO(estimator_pretrained_model= "saved_models/pi_random_16x16_indoor/estimator_epoch3.pth"),
    generator = generator.parametric_generator(),
    training = training.co_train(config_name),
)

