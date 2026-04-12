import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training, array_adapter


config_name = os.path.splitext(os.path.basename(__file__))[0]

# 20x20: array_factor_steps_theta=20, array_factor_steps_phi=80 -> input_array_sampling=1600
# ArrayAdapter projects 1600 -> 1024 to match baseline estimator's array_factor_len=1024
config = SimpleNamespace(
    assumption = assumption.beam100_20x20(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g_20x20(),
    estimator = estimator.PerceiverIO(depth=4, dim=640, latent_dim=640),
    generator = generator.random_20x20(),
    training = training.pre_defined(config_name),
    array_adapter = array_adapter.array_adapter(input_array_sampling=1600, output_array_sampling=1024),
)
config.training.epochs = 1
