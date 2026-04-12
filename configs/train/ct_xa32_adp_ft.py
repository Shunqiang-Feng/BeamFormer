import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training, array_adapter


config_name = os.path.splitext(os.path.basename(__file__))[0]

_ckpt_prefix = "saved_models/ct_xa32_adp"
_epoch_tag = "epoch26"

config = SimpleNamespace(
    assumption = assumption.beam256_32x32(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g_32x32(),
    estimator = estimator.PerceiverIO(depth=4, dim=1024, latent_dim=1024,
        estimator_pretrained_model=f"{_ckpt_prefix}/estimator_{_epoch_tag}.pth"),
    generator = generator.parametric_generator(
        generator_pretrained_model=f"{_ckpt_prefix}/generator_{_epoch_tag}.pth"),
    training = training.co_train(config_name),
    array_adapter = array_adapter.array_adapter(
        input_array_sampling=4096,
        output_array_sampling=1024,
        pretrained_model=f"{_ckpt_prefix}/array_adapter_{_epoch_tag}.pth",
    ),
)
config.training.batch_size = 72
config.training.epochs = 15