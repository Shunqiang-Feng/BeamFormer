import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam64(),
    dataset = dataset.uva_communication_28g(), 
    estimator = estimator.PerceiverIO(dim=256, latent_dim = 256, estimator_pretrained_model="saved_models/ablation_dim_256_random/estimator_epoch_final.pth"),
    generator = generator.parametric_generator(),
    training = training.co_train(config_name),
)

