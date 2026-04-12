import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam64(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g(),
    estimator = estimator.PerceiverIO(depth=4, dim=512, latent_dim=512, estimator_pretrained_model="saved_models/rnd_xa16/estimator_epoch_final.pth"),
    generator = generator.parametric_generator(),
    training = training.co_train(config_name),
)
