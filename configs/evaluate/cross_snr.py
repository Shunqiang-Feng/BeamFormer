from types import SimpleNamespace
import os
from configs.submodules import assumption, dataset, estimator, generator, training, ARN_model

# Base configuration for evaluation settings
request = SimpleNamespace(
    phase_constraint = False, 
    results_folder = "./.eval_tmp_folder/cross-snr",
    figures = 10,
    batch_size = 100, # batch size for evaluation
    test_batch_length = 1000, # if None, use all the test dataset
)


models = [

    SimpleNamespace(
        name = "SNR-n5",
        dataset = dataset.uva_communication_28g(add_noise=True, snr_min=-5),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_co_train_snr_n5/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/ablation_co_train_snr_n5/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),

    SimpleNamespace(
        name = "SNR-n10",
        dataset = dataset.uva_communication_28g(add_noise=True, snr_min=-10),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_co_train_snr_n10/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/ablation_co_train_snr_n10/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),

    SimpleNamespace(
        name = "SNR-n15",
        dataset = dataset.uva_communication_28g(add_noise=True, snr_min=-15),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_co_train_snr_n15/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/ablation_co_train_snr_n15/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),
]

include_related_work = [
    # SimpleNamespace(
    #     name = "2ACE",
    #     description = None,
    #     dataset = dataset.uva_communication_28g(add_noise=True, snr_min=3),
    #     assumption = assumption.beam512(),
    # ),
    # SimpleNamespace(
    #     name = "AgileLink",
    #     description = None,
    #     dataset = dataset.uva_communication_28g(add_noise=True, snr_min=3),
    #     assumption = assumption.beam384(),
    # ),
    # SimpleNamespace(
    #     name = "SectorSweep",
    #     description = None,
    #     dataset = dataset.uva_communication_28g(add_noise=True, snr_min=3),
    #     assumption = assumption.beam64(),
    # ),
    # SimpleNamespace(
    #     name = "Hierarchical",
    #     description = None,
    #     dataset = dataset.uva_communication_28g(add_noise=True, snr_min=3),
    #     assumption = assumption.beam64(),
    # ),
    # SimpleNamespace(
    #     name = "802ad",
    #     description = None,
    #     dataset = dataset.uva_communication_28g(add_noise=True, snr_min=3),
    #     assumption = assumption.beam64(),
    # ),
]

plot_combinations = [
]

# Main configuration
config = SimpleNamespace(
    request = request,
    models = models,
    include_related_work = include_related_work,
    plot_combinations = plot_combinations,
) 