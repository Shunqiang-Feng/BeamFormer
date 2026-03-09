from types import SimpleNamespace
import os
from configs.submodules import assumption, dataset, estimator, generator, training, ARN_model

# Base configuration for evaluation settings
request = SimpleNamespace(
    phase_constraint = False, 
    results_folder = "./.eval_tmp_folder/cross-pattern",
    figures = 10,
    batch_size = 100, # batch size for evaluation
    test_batch_length = 1000, # if None, use all the test dataset
)


models = [

    SimpleNamespace(
        name = "Isotropic",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),

    SimpleNamespace(
        name = "Patch",
        dataset = dataset.uva_communication_28g_patch(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),
]

include_related_work = [
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