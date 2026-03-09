import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training



config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam64(),
    dataset = dataset.homeoffice_communication_28g(),
    estimator = estimator.PerceiverIO(depth=2, dim=256, num_latents=16, estimator_pretrained_model= "saved_models/pi_param_co_train_indoor_mini_random/estimator_epoch_final.pth"),
    generator = generator.parametric_generator(),
    training = training.co_train(config_name),
)
