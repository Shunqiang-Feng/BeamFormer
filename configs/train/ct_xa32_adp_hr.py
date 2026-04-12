import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training, array_adapter


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam256_32x32_high_res(phi_endpoint=False, query_valid_num=1600),
    dataset = dataset.homeoffice_communication_28g_32x32(),
    estimator = estimator.PerceiverIO(depth=4, dim=1024, latent_dim=1024,
        estimator_pretrained_model="saved_models/rnd_xa32_adp_hr/estimator_epoch_final.pth"),
    generator = generator.parametric_generator(),
    training = training.co_train(config_name),
    array_adapter = array_adapter.array_adapter(
        input_array_sampling=4096,
        output_array_sampling=1024,
        pretrained_model="saved_models/rnd_xa32_adp_hr/array_adapter_epoch_final.pth",
    ),
)
config.training.batch_size = 36
