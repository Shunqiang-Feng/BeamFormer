import os
from types import SimpleNamespace
from configs.submodules import ARN_model, ARN_training, assumption, dataset, estimator, generator


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam144_24x24(phi_endpoint=False),
    dataset = dataset.homeoffice_communication_28g_24x24(),
    generator = generator.parametric_generator(
        "saved_models/ct_xa24_adp/generator_epoch_final.pth"),
    arn_model = ARN_model.typical_ARN(),
)

config = ARN_training.add_train_ARN(config, config_name=config_name)
