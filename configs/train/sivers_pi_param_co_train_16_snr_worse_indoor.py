import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, training


config_name = os.path.splitext(os.path.basename(__file__))[0]

config = SimpleNamespace(
    assumption = assumption.beam16_sivers(),
    dataset = dataset.sivers_indoor(add_noise = True,  snr_min = -3),
    estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/sivers_random_16/estimator_epoch_final.pth"),
    generator = generator.parametric_generator(),
    training = training.co_train(config_name),
)



