from types import SimpleNamespace
import os
from configs.submodules import assumption, dataset, estimator, generator, ARN_model

config_name = os.path.splitext(os.path.basename(__file__))[0]

# Base configuration for evaluation settings
request = SimpleNamespace(
    phase_constraint = False, 
    results_folder = f"./eval_results/{config_name}",
    figures = 10,
    batch_size = 100, # batch size for evaluation
    test_batch_length = 100, # if None, use all the test dataset
)

# 我们的方法在Home Office数据集上的表现
models = [
    SimpleNamespace(
        name = "Our Method",
        dataset = dataset.homeoffice_communication_28g_nlos(),
        assumption = assumption.beam64_res4(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN('./ARN_saved_models/PARAM-64/model_epoch_final.pth'),
    ),
    SimpleNamespace(
        name = "MLP",
        dataset = dataset.homeoffice_communication_28g_nlos(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/mlpae_param_co_train_indoor_mini/generator_epoch_final.pth"),
        estimator = estimator.MLPAutoEncoder(estimator_pretrained_model="saved_models/mlpae_param_co_train_indoor_mini/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),
    SimpleNamespace(
        name = "CNN",
        dataset = dataset.homeoffice_communication_28g_nlos(),
        assumption = assumption.beam64(),
        scheme = "pre-defined",
        generator = generator.single_arm_16x16(),
        estimator = estimator.CNNGenerator(estimator_pretrained_model="saved_models/cnn_single_16x16_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),
]

# 所有baseline方法的对比
include_related_work = [

    
    # AgileLink baseline
    SimpleNamespace(
        name = "AgileLink",
        description = None,
        dataset = dataset.homeoffice_communication_28g_nlos(),
        assumption = assumption.beam384(),
    ),
    
    # SectorSweep baseline
    SimpleNamespace(
        name = "SectorSweep",
        description = None,
        dataset = dataset.homeoffice_communication_28g_nlos(),
        assumption = assumption.beam64(),
    ),
    
    # Hierarchical baseline
    SimpleNamespace(
        name = "Hierarchical",
        description = None,
        dataset = dataset.homeoffice_communication_28g_nlos(),
        assumption = assumption.beam64(),
    ),
    
    # 802.11ad baseline
    SimpleNamespace(
        name = "802ad",
        description = None,
        dataset = dataset.homeoffice_communication_28g_nlos(),
        assumption = assumption.beam64(),
    ),

    # 2ACE baseline
    SimpleNamespace(
        name = "2ACE",
        description = None,
        dataset = dataset.homeoffice_communication_28g_nlos(),
        assumption = assumption.beam512(),
    ),
]

plot_combinations = [
    {
        "name": "homeoffice-28g-baseline-comparison",
        "components": ["Our Method", "2ACE", "AgileLink", "SectorSweep", "Hierarchical", "802ad"],
        "metrics": "loss",
        "format": "cdf",
    },
]

# Main configuration
config = SimpleNamespace(
    request = request,
    models = models,
    include_related_work = include_related_work,
    plot_combinations = plot_combinations,
)