import os
from types import SimpleNamespace
from configs.submodules import assumption, dataset, estimator, generator, ARN_model

config_name = os.path.splitext(os.path.basename(__file__))[0]

request = SimpleNamespace(
    phase_constraint=False,
    results_folder="./eval_results/visualize_ct_xa16",
    figures=10,
    batch_size=5,
    test_batch_length=1,
    noise_gt=True,  # replace GT tensor with zeros during visualization (renders as flat dark disk)
)

models = [
    SimpleNamespace(
        name="ct_xa16",
        dataset=dataset.homeoffice_communication_28g(add_noise=True, snr_min=-100),
        assumption=assumption.beam64(phi_endpoint=False),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model="saved_models/ct_xa16/generator_epoch_final.pth"
        ),
        estimator=estimator.PerceiverIO(
            depth=4,
            dim=512,
            latent_dim=512,
            estimator_pretrained_model="saved_models/ct_xa16/estimator_epoch_final.pth",
        ),
        arn_model=ARN_model.typical_ARN(None),
    ),
]

include_related_work = []

config = SimpleNamespace(
    request=request,
    models=models,
    include_related_work=include_related_work,
)
