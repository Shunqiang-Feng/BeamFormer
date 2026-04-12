import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training, array_adapter


config_name = os.path.splitext(os.path.basename(__file__))[0]

# 28x28: array_factor_steps_theta=28, array_factor_steps_phi=112 -> input_array_sampling=3136
# ArrayAdapter projects 3136 -> 1024 to match baseline estimator's array_factor_len=1024
config = SimpleNamespace(
    assumption = assumption.beam196_28x28(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g_28x28(),
    estimator = estimator.PerceiverIO(depth=4, dim=896, latent_dim=896),
    generator = generator.random_28x28(),
    training = training.pre_defined(config_name),
    array_adapter = array_adapter.array_adapter(input_array_sampling=3136, output_array_sampling=1024),
)
config.training.epochs = 1
config.training.batch_size = 72
