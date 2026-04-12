from types import SimpleNamespace
import os
from configs.submodules import assumption, dataset, estimator, generator, ARN_model, array_adapter

config_name = os.path.splitext(os.path.basename(__file__))[0]

# Base configuration for evaluation settings
request = SimpleNamespace(
    phase_constraint=False,
    results_folder=f"./.eval_tmp_folder/{config_name}",
    figures=10,
    batch_size=1,
    test_batch_length=10,
)

models = [

    SimpleNamespace(
        name = "4x4",
        dataset = dataset.sivers_indoor(add_noise=True, snr_min=-3),
        assumption = assumption.beam16_sivers(),
        scheme = "co-train",
        hardware = True,
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/sivers/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/sivers/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),

    SimpleNamespace(
        name = "8x8",
        dataset = dataset.ibm_indoor(add_noise=True, snr_min=-3),
        assumption = assumption.beam32(),
        scheme = "co-train",
        hardware = True,
        generator = generator.parametric_generator(generator_pretrained_model="saved_models/ibm/generator_epoch_final.pth"),
        estimator = estimator.PerceiverIO(estimator_pretrained_model="saved_models/ibm/estimator_epoch_final.pth"),
        arn_model = ARN_model.typical_ARN(),
    ),

    # 16x16: baseline model, no array adapter
    # array_factor_steps: 16x64=1024
    SimpleNamespace(
        name="16x16",
        dataset=dataset.homeoffice_communication_28g(),
        assumption=assumption.beam64_hr(phi_endpoint=False),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model="saved_models/ct_xa16/generator_epoch_final.pth"
        ),
        estimator=estimator.PerceiverIO(
            depth=4, dim=512, latent_dim=512,
            estimator_pretrained_model="saved_models/ct_xa16/estimator_epoch_final.pth",
        ),
        arn_model=ARN_model.typical_ARN(None),
    ),

    # 20x20: array_factor_steps: 20x80=1600 -> adapter projects to 1024
    SimpleNamespace(
        name="20x20",
        dataset=dataset.homeoffice_communication_28g_20x20(),
        assumption=assumption.beam100_20x20_hr(phi_endpoint=False),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model="saved_models/ct_xa20_adp/generator_epoch_final.pth"
        ),
        estimator=estimator.PerceiverIO(
            depth=4, dim=640, latent_dim=640,
            estimator_pretrained_model="saved_models/ct_xa20_adp/estimator_epoch_final.pth",
        ),
        arn_model=ARN_model.typical_ARN(None),
        array_adapter=array_adapter.array_adapter(
            input_array_sampling=1600,
            output_array_sampling=1024,
            pretrained_model="saved_models/ct_xa20_adp/array_adapter_epoch_final.pth",
        ),
    ),

    # 24x24: array_factor_steps: 24x96=2304 -> adapter projects to 1024
    SimpleNamespace(
        name="24x24",
        dataset=dataset.homeoffice_communication_28g_24x24(),
        assumption=assumption.beam144_24x24_hr(phi_endpoint=False),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model="saved_models/ct_xa24_adp/generator_epoch_final.pth"
        ),
        estimator=estimator.PerceiverIO(
            depth=4, dim=768, latent_dim=768,
            estimator_pretrained_model="saved_models/ct_xa24_adp/estimator_epoch_final.pth",
        ),
        arn_model=ARN_model.typical_ARN(None),
        array_adapter=array_adapter.array_adapter(
            input_array_sampling=2304,
            output_array_sampling=1024,
            pretrained_model="saved_models/ct_xa24_adp/array_adapter_epoch_final.pth",
        ),
    ),

    # 28x28: array_factor_steps: 28x112=3136 -> adapter projects to 1024
    SimpleNamespace(
        name="28x28",
        dataset=dataset.homeoffice_communication_28g_28x28(),
        assumption=assumption.beam196_28x28_hr(phi_endpoint=False),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model="saved_models/ct_xa28_adp_hr_ft/generator_epoch2.pth"
        ),
        estimator=estimator.PerceiverIO(
            depth=4, dim=896, latent_dim=896,
            estimator_pretrained_model="saved_models/ct_xa28_adp_hr_ft/estimator_epoch2.pth",
        ),
        arn_model=ARN_model.typical_ARN(None),
        array_adapter=array_adapter.array_adapter(
            input_array_sampling=3136,
            output_array_sampling=1024,
            pretrained_model="saved_models/ct_xa28_adp_hr_ft/array_adapter_epoch2.pth",
        ),
    ),


    # 32x32: array_factor_steps: 32x128=4096 -> adapter projects to 1024
    SimpleNamespace(
        name="32x32",
        dataset=dataset.homeoffice_communication_28g_32x32(),
        assumption=assumption.beam256_32x32_hr(phi_endpoint=False),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model="saved_models/ct_xa32_adp_ft/generator_epoch20.pth"
        ),
        estimator=estimator.PerceiverIO(
            depth=4, dim=1024, latent_dim=1024,
            estimator_pretrained_model="saved_models/ct_xa32_adp_ft/estimator_epoch20.pth",
        ),
        arn_model=ARN_model.typical_ARN(None),
        array_adapter=array_adapter.array_adapter(
            input_array_sampling=4096,
            output_array_sampling=1024,
            pretrained_model="saved_models/ct_xa32_adp_ft/array_adapter_epoch20.pth",
        ),
    ),
]

include_related_work = []

plot_combinations = []

# Main configuration
config = SimpleNamespace(
    request=request,
    models=models,
    include_related_work=include_related_work,
    plot_combinations=plot_combinations,
)
