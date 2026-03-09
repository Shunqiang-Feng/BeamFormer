from types import SimpleNamespace
import os
from configs.submodules import assumption, dataset, estimator, generator, ARN_model

config_name = os.path.splitext(os.path.basename(__file__))[0]

# Base configuration for evaluation settings
request = SimpleNamespace(
    phase_constraint = False, 
    results_folder = f"./.eval_tmp_folder/{config_name}",
    figures = 10,
    batch_size = 100, # batch size for evaluation
    test_batch_length = 1000, # if None, use all the test dataset
)

models = [
    SimpleNamespace(
        name = "indoor-model-homeoffice",
        dataset = dataset.homeoffice_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN('./ARN_saved_models/PARAM-64/model_epoch_final.pth'),
    ),
    
    SimpleNamespace(
        name = "indoor-model-classroom",
        dataset = dataset.classroom_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN('./ARN_saved_models/PARAM-64/model_epoch_final.pth'),
    ),
    
    SimpleNamespace(
        name = "indoor-model-uva-comm",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN('./ARN_saved_models/PARAM-64/model_epoch_final.pth'),
    ),
    SimpleNamespace(
        name = "indoor-model-rural-comm",
        dataset = dataset.rural_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN('./ARN_saved_models/PARAM-64/model_epoch_final.pth'),
    ),
    SimpleNamespace(
        name = "indoor-model-nyc-comm",
        dataset = dataset.nyc_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN('./ARN_saved_models/PARAM-64/model_epoch_final.pth'),
    ),
    
]

include_related_work = []

plot_combinations = []

# Main configuration
config = SimpleNamespace(
    request = request,
    models = models,
    include_related_work = include_related_work,
    plot_combinations = plot_combinations,
)