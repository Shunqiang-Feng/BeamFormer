from types import SimpleNamespace

def uniform_4x4():

    generator=SimpleNamespace(
        name = "uniform_4x4",
        type='uniform', 
        M_act = 4,
        N_act = 4,
        phase_constraint=False,
    )

    return generator

def uniform_16x16():

    generator=SimpleNamespace(
        name = "uniform_16x16",
        type='uniform',
        M_act = 16,
        N_act = 16,
        phase_constraint=False,
    )

    return generator


def single_arm_16x16():

    generator=SimpleNamespace(
        name = "single-arm_16x16",
        type='single-arm',
        M_act = 16,
        N_act = 16,
        phase_constraint=False,
    )

    return generator

def random_16x16():
    generator=SimpleNamespace(
        name = "random_16x16",
        type='random', 
        M_act = 16,
        N_act = 16,
        phase_constraint=False,
    )

    return generator


def parametric_generator(generator_pretrained_model = None, generator_lr_scheduler_type = "cosine"):

    generator=SimpleNamespace(
        name = 'PARAM',
        type='PARAM', 
        generator_pretrained_model=generator_pretrained_model,
        generator_lr_scheduler_type = generator_lr_scheduler_type,
        phase_constraint=False,
    )

    return generator

def vit_generator(generator_pretrained_model = None, generator_lr_scheduler_type = "cosine"):

    generator=SimpleNamespace(
        name = 'VIT',
        type='VIT', 
        generator_pretrained_model=generator_pretrained_model,
        generator_lr_scheduler_type = generator_lr_scheduler_type,
        phase_constraint=False,
    )

    return generator

def sivers_random_4x4():
    generator=SimpleNamespace(
        name = "sivers_random_4x4",
        type='random', 
        M_act = 4,
        N_act = 4,
        phase_constraint=False,

    )

    return generator

def sivers_uniform_2x2():
    generator=SimpleNamespace(
        name = "sivers_uniform_2x2",
        type='uniform', 
        M_act = 2,
        N_act = 2,
        phase_constraint=False,

    )

    return generator

def sivers_uniform_4x4():
    generator=SimpleNamespace(
        name = "sivers_uniform_4x4",
        type='uniform', 
        M_act = 4,
        N_act = 4,
        phase_constraint=False,

    )

    return generator


def ibm_random_8x8():
    generator=SimpleNamespace(
        name = "ibm_random_8x8",
        type='random', 
        M_act = 8,
        N_act = 8,
        phase_constraint=False,

    )

    return generator

def ibm_uniform_8x8():
    generator=SimpleNamespace(
        name = "ibm_uniform_8x8",
        type='uniform',
        M_act = 8,
        N_act = 8,
        phase_constraint=False,

    )


def random_20x20():
    generator=SimpleNamespace(
        name = "random_20x20",
        type='random',
        M_act = 20,
        N_act = 20,
        phase_constraint=False,
    )

    return generator


def random_28x28():
    generator=SimpleNamespace(
        name = "random_28x28",
        type='random',
        M_act = 28,
        N_act = 28,
        phase_constraint=False,
    )

    return generator


def random_24x24():
    generator=SimpleNamespace(
        name = "random_24x24",
        type='random',
        M_act = 24,
        N_act = 24,
        phase_constraint=False,
    )

    return generator


def random_32x32():
    generator=SimpleNamespace(
        name = "random_32x32",
        type='random',
        M_act = 32,
        N_act = 32,
        phase_constraint=False,
    )

    return generator

    return generator