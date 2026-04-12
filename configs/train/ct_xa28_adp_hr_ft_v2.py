import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training, array_adapter


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam196_28x28_hr(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g_28x28(),
    estimator = estimator.PerceiverIO(depth=4, dim=896, latent_dim=896),
    generator = generator.parametric_generator(),
    training = training.co_train(config_name, ckpt_path=None),
    array_adapter = array_adapter.array_adapter(
        input_array_sampling=3136,
        output_array_sampling=1024,
    ),
)
config.training.batch_size = 10
config.training.epochs = 10
