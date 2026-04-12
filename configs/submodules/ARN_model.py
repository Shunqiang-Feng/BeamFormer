from types import SimpleNamespace

def typical_ARN(ARN_model_pretrained_model = None,):
    return SimpleNamespace(
        output_num = 16, # this parameter is useless
        hidden_dims = 512,
        ARN_model_pretrained_model = ARN_model_pretrained_model,
    )