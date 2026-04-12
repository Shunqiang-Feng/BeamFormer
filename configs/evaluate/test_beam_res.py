import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, ARN_model

config_name = os.path.splitext(os.path.basename(__file__))[0]

request = SimpleNamespace(
    phase_constraint=False,
    results_folder="./eval_results/test_beam_res",
    figures=1000,          # generate all test samples
    batch_size=10,
    test_batch_length=1,
)

models = [
    SimpleNamespace(
        name="ct_xa16",
        dataset=dataset.diy_28G_csi_16x16(),
        assumption=assumption.beam64_hr(phi_endpoint=False),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model="saved_models/ct_xa16_large_ft/best_generator.pth"
        ),
        estimator=estimator.PerceiverIO(
            estimator_pretrained_model="saved_models/ct_xa16_large_ft/best_estimator.pth",
        ),
        arn_model=ARN_model.typical_ARN(None),  # no ARN for ct_xa16
    ),
]

include_related_work = []

config = SimpleNamespace(
    request=request,
    models=models,
    include_related_work=include_related_work,
)
