import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training, array_adapter


config_name = os.path.splitext(os.path.basename(__file__))[0]

_ckpt_prefix = "saved_models/ct_xa28_adp"
_epoch_tag = "epoch22"

config = SimpleNamespace(
    assumption = assumption.beam196_28x28(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g_28x28(),
    estimator = estimator.PerceiverIO(depth=4, dim=896, latent_dim=896,
        estimator_pretrained_model=f"{_ckpt_prefix}/estimator_{_epoch_tag}.pth"),
    generator = generator.parametric_generator(
        generator_pretrained_model=f"{_ckpt_prefix}/generator_{_epoch_tag}.pth"),
    training = training.co_train(config_name),
    array_adapter = array_adapter.array_adapter(
        input_array_sampling=3136,
        output_array_sampling=1024,
        pretrained_model=f"{_ckpt_prefix}/array_adapter_{_epoch_tag}.pth",
    ),
)
config.training.batch_size = 72
config.training.epochs = 15
