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
        name = "Directional Beam Weight [Full Array]",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "pre-defined",
        generator = generator.uniform_16x16(),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_uniform_16x16_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="./ARN_saved_models/uniform_16x16-64/model_epoch_final.pth"),
    ),
    SimpleNamespace(
        name = "Directional Beam Weight [4x4 Subarray]",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "pre-defined",
        generator = generator.uniform_4x4(),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_uniform_4x4_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="./ARN_saved_models/uniform_4x4-64/model_epoch_final.pth"),
    ),
    SimpleNamespace(
        name = "ML-Optimized Weight",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="./ARN_saved_models/arn_IndoorModel_OutdoorSce/model_epoch_final.pth"),
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