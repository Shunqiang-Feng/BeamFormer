import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, ARN_model

config_name = os.path.splitext(os.path.basename(__file__))[0]

request = SimpleNamespace(
    phase_constraint=False,
    results_folder="./eval_results/visualize_ct_xa16_large_ft",
    figures=96,
    batch_size=5,
    test_batch_length=None,
)

models = [
    SimpleNamespace(
        name="ct_xa16_large_ft/",
        dataset=dataset.homeoffice_communication_28g(),
        assumption=assumption.beam64_hr(phi_endpoint=False),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model="saved_models/ct_xa16_large_ft/best_generator.pth"
        ),
        estimator=estimator.PerceiverIO(
            estimator_pretrained_model="saved_models/ct_xa16_large_ft/best_estimator.pth",
        ),
        arn_model=ARN_model.typical_ARN(
            ARN_model_pretrained_model='ARN_saved_models/PARAM-64/model_epoch_final.pth'
        ),
    ),
]

include_related_work = []

config = SimpleNamespace(
    request=request,
    models=models,
    include_related_work=include_related_work,
)
