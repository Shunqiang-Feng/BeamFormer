import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training, array_adapter


config_name = os.path.splitext(os.path.basename(__file__))[0]

# 24x24: array_factor_steps_theta=24, array_factor_steps_phi=96 -> input_array_sampling=2304
# ArrayAdapter projects 2304 -> 1024 to match baseline estimator's array_factor_len=1024
config = SimpleNamespace(
    assumption = assumption.beam144_24x24(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g_24x24(),
    estimator = estimator.PerceiverIO(depth=4, dim=768, latent_dim=768,
        estimator_pretrained_model="saved_models/rnd_xa24_adp/estimator_epoch_final.pth"),
    generator = generator.parametric_generator(),
    training = training.co_train(config_name),
    array_adapter = array_adapter.array_adapter(
        input_array_sampling=2304,
        output_array_sampling=1024,
        pretrained_model="saved_models/rnd_xa24_adp/array_adapter_epoch_final.pth",
    ),
)
