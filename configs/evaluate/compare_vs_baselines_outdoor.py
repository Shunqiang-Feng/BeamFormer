from types import SimpleNamespace
import os
from configs.submodules import assumption, dataset, estimator, generator, training, ARN_model

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
        name = "Our Method",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN('ARN_saved_models/arn_IndoorModel_OutdoorSce/model_epoch_final.pth'),
    ),
    SimpleNamespace(
        name = "MLP",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/mlpae_param_co_train_indoor_mini/generator_epoch_final.pth"),
        estimator = estimator.MLPAutoEncoder(estimator_pretrained_model="saved_models/mlpae_param_co_train_indoor_mini/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),
    SimpleNamespace(
        name = "CNN",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "pre-defined",
        generator = generator.single_arm_16x16(),
        estimator = estimator.CNNGenerator(estimator_pretrained_model="saved_models/cnn_single_16x16_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),
]

include_related_work = [
    # 2ACE baseline

    # AgileLink baseline
    SimpleNamespace(
        name = "AgileLink",
        description = None,
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam384(),
    ),
    
    # SectorSweep baseline
    SimpleNamespace(
        name = "SectorSweep",
        description = None,
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
    ),
    
    # Hierarchical baseline
    SimpleNamespace(
        name = "Hierarchical",
        description = None,
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
    ),
    
    # 802.11ad baseline
    SimpleNamespace(
        name = "802ad",
        description = None,
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
    ),

    SimpleNamespace(
        name = "2ACE",
        description = None,
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam512(),
    ),
    
]

plot_combinations = []

# Main configuration
config = SimpleNamespace(
    request = request,
    models = models,
    include_related_work = include_related_work,
    plot_combinations = plot_combinations,
)