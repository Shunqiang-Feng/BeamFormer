import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training, array_adapter


config_name = os.path.splitext(os.path.basename(__file__))[0]

# 28x28: array_factor_steps_theta=28, array_factor_steps_phi=112 -> input_array_sampling=3136
# ArrayAdapter projects 3136 -> 1024 to match baseline estimator's array_factor_len=1024
config = SimpleNamespace(
    assumption = assumption.beam196_28x28(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g_28x28(),
    estimator = estimator.PerceiverIO(depth=4, dim=896, latent_dim=896,
        estimator_pretrained_model="saved_models/rnd_xa28_adp/estimator_epoch_final.pth"),
    generator = generator.parametric_generator(),
    training = training.co_train(config_name),
    array_adapter = array_adapter.array_adapter(
        input_array_sampling=3136,
        output_array_sampling=1024,
        pretrained_model="saved_models/rnd_xa28_adp/array_adapter_epoch_final.pth",
    ),
)
config.training.batch_size = 72
