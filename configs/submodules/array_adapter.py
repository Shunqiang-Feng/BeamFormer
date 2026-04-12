from types import SimpleNamespace


def array_adapter(input_array_sampling, output_array_sampling, pretrained_model=None):
    return SimpleNamespace(
        input_array_sampling=input_array_sampling,
        output_array_sampling=output_array_sampling,
        pretrained_model=pretrained_model,
    )
