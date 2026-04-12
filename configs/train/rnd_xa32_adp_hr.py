import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training, array_adapter


config_name = os.path.splitext(os.path.basename(__file__))[0]

# high_res: angle_steps_theta=80, angle_steps_phi=320 -> angle_spectrum_length=25600
# ArrayAdapter projects 4096 -> 1024 (same array structure as beam256_32x32)
config = SimpleNamespace(
    assumption = assumption.beam256_32x32_high_res(phi_endpoint=False, query_valid_num=1600),
    dataset = dataset.homeoffice_communication_28g_32x32(),
    estimator = estimator.PerceiverIO(depth=4, dim=1024, latent_dim=1024),
    generator = generator.random_32x32(),
    training = training.pre_defined(config_name),
    array_adapter = array_adapter.array_adapter(input_array_sampling=4096, output_array_sampling=1024),
)
config.training.epochs = 3
config.training.batch_size = 36
