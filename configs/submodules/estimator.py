from types import SimpleNamespace


def PerceiverIO(depth = 8, array_factor_len = 1024, dim=1024, num_latents = 64, latent_dim = 1024, cross_heads = 8, latent_heads = 8, decoder_ff = True, optimizer = "adamw", 
                    estimator_pretrained_model = None, estimator_lr_scheduler_type = "cosine", seq_dropout_prob = 0):
    config = SimpleNamespace(
        type = "perceiver_io",
        depth = depth,
        array_factor_len = array_factor_len,
        dim=dim,
        num_latents = num_latents, 
        latent_dim = latent_dim, 
        cross_heads = cross_heads,
        latent_heads = latent_heads, 
        decoder_ff = decoder_ff,
        estimator_pretrained_model = estimator_pretrained_model,
        estimator_lr_scheduler_type = estimator_lr_scheduler_type, # ["cosine", "step", "flat_cosine"]
        optimizer = optimizer,
        seq_dropout_prob = seq_dropout_prob,
    )
    config.dim_feedforward = config.array_factor_len * 2
    config.queries_dim = config.dim
    config.cross_dim_head = config.queries_dim // config.cross_heads
    config.latent_dim_head = config.latent_dim // config.latent_heads

    return config

def PerceiverIO_deep(estimator_pretrained_model = None, estimator_lr_scheduler_type = "cosine"):
    return PerceiverIO(depth = 16, estimator_pretrained_model = estimator_pretrained_model, estimator_lr_scheduler_type = estimator_lr_scheduler_type)

def PerceiverIO_shallow(estimator_pretrained_model = None, estimator_lr_scheduler_type = "cosine"):
    return PerceiverIO(depth = 4, estimator_pretrained_model = estimator_pretrained_model, estimator_lr_scheduler_type = estimator_lr_scheduler_type)
