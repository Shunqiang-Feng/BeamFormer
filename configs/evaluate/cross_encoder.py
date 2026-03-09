from types import SimpleNamespace
import os
from configs.submodules import assumption, dataset, estimator, generator, training, ARN_model

config_name = os.path.splitext(os.path.basename(__file__))[0]

request = SimpleNamespace(
    phase_constraint=False,
    results_folder=f"./.eval_tmp_folder/{config_name}",
    figures=1,
    batch_size=100,
    test_batch_length=1000,
)

models = [
    SimpleNamespace(
        name="Array Factor",
        scheme="co-train",
        assumption=assumption.beam64(position_encoding_type='array_factor', phi_endpoint=False),
        dataset=dataset.homeoffice_communication_28g(),
        estimator=estimator.PerceiverIO(estimator_pretrained_model="saved_models/pi_param_co_train_indoor/estimator_epoch_final.pth"),
        generator=generator.parametric_generator(generator_pretrained_model="saved_models/pi_param_co_train_indoor/generator_epoch_final.pth"),
        arn_model=ARN_model.typical_ARN("ARN_saved_models/PARAM-64/model_epoch_final.pth"),
    ),
    SimpleNamespace(
        name="2D Positional",
        scheme="co-train",
        assumption=assumption.beam64(position_encoding_type='2d_positional', phi_endpoint=False),
        dataset=dataset.homeoffice_communication_28g(),
        estimator=estimator.PerceiverIO(estimator_pretrained_model="saved_models/co_train_indoor_2d_positional/estimator_epoch_final.pth"),
        generator=generator.parametric_generator(generator_pretrained_model="saved_models/co_train_indoor_2d_positional/generator_epoch_final.pth"),
        arn_model=ARN_model.typical_ARN("ARN_saved_models/arn_co_train_indoor_2d_positional/model_epoch_final.pth"),
    ),
    SimpleNamespace(
        name="Concat",
        scheme="co-train",
        assumption=assumption.beam64(position_encoding_type='concat', phi_endpoint=False),
        dataset=dataset.homeoffice_communication_28g(),
        estimator=estimator.PerceiverIO(array_factor_len=512, estimator_pretrained_model="saved_models/co_train_indoor_concat/estimator_epoch_final.pth"),
        generator=generator.parametric_generator(generator_pretrained_model="saved_models/co_train_indoor_concat/generator_epoch_final.pth"),
        arn_model=ARN_model.typical_ARN("ARN_saved_models/arn_co_train_indoor_concat/model_epoch_final.pth"),
    ),
]

include_related_work = []
plot_combinations = []

config = SimpleNamespace(
    request=request,
    models=models,
    include_related_work=include_related_work,
    plot_combinations=plot_combinations,
)
