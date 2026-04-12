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

# Baseline model (pi_param_co_train_outdoor)
baseline_model = SimpleNamespace(
    name = "Baseline",
    dataset = dataset.uva_communication_28g(),
    assumption = assumption.beam64(),
    scheme = "co-train",
    generator = generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_outdoor/generator_epoch_final.pth"),
    estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_outdoor/estimator_epoch_final.pth"),
    arn_model = ARN_model.typical_ARN('ARN_saved_models/arn_IndoorModel_OutdoorSce/model_epoch_final.pth'),
)

# Models for ablation study
models = [
    # Baseline model
    baseline_model,
    
    # Dimension ablation (dim=1024 is baseline, testing 256 and 512)
    SimpleNamespace(
        name = "Dim-256",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_dim_256_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(dim=256, latent_dim=256, estimator_pretrained_model="saved_models/ablation_dim_256_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_dim_256_cotrain/model_epoch_final.pth"),
    ),
    
    SimpleNamespace(
        name = "Dim-512",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_dim_512_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(dim=512, latent_dim=512, estimator_pretrained_model="saved_models/ablation_dim_512_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_dim_512_cotrain/model_epoch_final.pth"),
    ),

    SimpleNamespace(
        name = "Dim-128",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_dim_128_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(dim=128, latent_dim=128, estimator_pretrained_model="saved_models/ablation_dim_128_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_dim_128_cotrain/model_epoch_final.pth"),
    ),

    SimpleNamespace(
        name = "Dim-64",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_dim_64_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(dim=64, latent_dim=64, estimator_pretrained_model="saved_models/ablation_dim_64_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_dim_64_cotrain/model_epoch_final.pth"),
    ),
    
    # Depth ablation (depth=8 is baseline, testing 1, 2, 4)
    SimpleNamespace(
        name = "Depth-1",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_depth_1_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(depth=1, estimator_pretrained_model="saved_models/ablation_depth_1_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_depth_1_cotrain/model_epoch_final.pth"),
    ),
    
    SimpleNamespace(
        name = "Depth-2",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_depth_2_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(depth=2, estimator_pretrained_model="saved_models/ablation_depth_2_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_depth_2_cotrain/model_epoch_final.pth"),
    ),
    
    SimpleNamespace(
        name = "Depth-4",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_depth_4_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(depth=4, estimator_pretrained_model="saved_models/ablation_depth_4_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_depth_4_cotrain/model_epoch_final.pth"),
    ),
    
    # Latent number ablation (num_latents=64 is baseline, testing 16, 32)
    SimpleNamespace(
        name = "Latent-16",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_latentnum_16_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(num_latents=16, estimator_pretrained_model="saved_models/ablation_latentnum_16_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_latentnum_16_cotrain/model_epoch_final.pth"),
    ),
    
    SimpleNamespace(
        name = "Latent-32",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_latentnum_32_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(num_latents=32, estimator_pretrained_model="saved_models/ablation_latentnum_32_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_latentnum_32_cotrain/model_epoch_final.pth"),
    ),

    SimpleNamespace(
        name = "Latent-8",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_latentnum_8_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(num_latents=8, estimator_pretrained_model="saved_models/ablation_latentnum_8_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_latentnum_8_cotrain/model_epoch_final.pth"),
    ),

    SimpleNamespace(
        name = "Latent-4",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_latentnum_4_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(num_latents=4, estimator_pretrained_model="saved_models/ablation_latentnum_4_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_latentnum_4_cotrain/model_epoch_final.pth"),
    ),

    SimpleNamespace(
        name = "Latent-2",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam64(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_latentnum_2_cotrain/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(num_latents=2, estimator_pretrained_model="saved_models/ablation_latentnum_2_cotrain/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_latentnum_2_cotrain/model_epoch_final.pth"),
    ),
    
    # Sample number ablation (sample_num=64 is baseline, testing 8, 16, 32)
    SimpleNamespace(
        name = "Sample-8",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam8(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_co_train_8beam/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/ablation_co_train_8beam/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_co_train_8beam/model_epoch_final.pth"),
    ),
    
    SimpleNamespace(
        name = "Sample-16",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam16(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_co_train_16beam/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/ablation_co_train_16beam/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_co_train_16beam/model_epoch_final.pth"),
    ),
    
    SimpleNamespace(
        name = "Sample-32",
        dataset = dataset.uva_communication_28g(),
        assumption = assumption.beam32(),
        scheme = "co-train",
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ablation_co_train_32beam/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/ablation_co_train_32beam/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(ARN_model_pretrained_model="ARN_saved_models/arn_ablation_co_train_32beam/model_epoch_final.pth"),
    ),
]

include_related_work = []

# Plotting combinations for each ablation dimension
plot_combinations = [
    
]

# Main configuration
config = SimpleNamespace(
    request = request,
    models = models,
    include_related_work = include_related_work,
    plot_combinations = plot_combinations,
)