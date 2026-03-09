from types import SimpleNamespace


def beam64(position_encoding_type='array_factor', phi_endpoint=True):
    config = SimpleNamespace(
        sample_num=64,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

def beam64_res2():
    config = SimpleNamespace(
        sample_num=64,
        angle_steps_theta=20*2,
        angle_steps_phi=80*2,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


def beam64_res4():
    config = SimpleNamespace(
        sample_num=64,
        angle_steps_theta=20*4,
        angle_steps_phi=80*4,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


def beam32():
    config = SimpleNamespace(
        sample_num=32,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config



def beam16():
    config = SimpleNamespace(
        sample_num=16,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

def beam8():
    config = SimpleNamespace(
        sample_num=8,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

def beam512():
    config = SimpleNamespace(
        sample_num=512,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

def beam384():
    config = SimpleNamespace(
        sample_num=384,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

def beam4_sivers():
    config = SimpleNamespace(
        sample_num=4,
        # angle_steps_theta=10,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

def beam8_sivers():
    config = SimpleNamespace(
        sample_num=8,
        # angle_steps_theta=10,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

def beam16_sivers():
    config = SimpleNamespace(
        sample_num=16,
        # angle_steps_theta=10,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

