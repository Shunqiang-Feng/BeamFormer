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

def beam64_hr(position_encoding_type='array_factor', phi_endpoint=False):
    config = SimpleNamespace(
        sample_num=64,
        angle_steps_theta=91,
        angle_steps_phi=360,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


def beam64_res2(position_encoding_type='array_factor', phi_endpoint=True):
    config = SimpleNamespace(
        sample_num=64,
        angle_steps_theta=20*2,
        angle_steps_phi=80*2,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


def beam64_res4(position_encoding_type='array_factor', phi_endpoint=True):
    config = SimpleNamespace(
        sample_num=64,
        angle_steps_theta=20*4,
        angle_steps_phi=80*4,
        array_factor_steps_theta=16,
        array_factor_steps_phi=64,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
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


def beam100_20x20_hr(position_encoding_type='array_factor', phi_endpoint=False):
    config = SimpleNamespace(
        sample_num=100,
        angle_steps_theta=91,
        angle_steps_phi=360,
        array_factor_steps_theta=20,
        array_factor_steps_phi=80,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


def beam100_20x20(position_encoding_type='array_factor', phi_endpoint=False):
    # 20x20: array_factor_steps_theta=20, phi=80 -> input_array_sampling=1600
    # sample_num = (20/2)^2 = 100
    config = SimpleNamespace(
        sample_num=100,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=20,
        array_factor_steps_phi=80,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config



def beam144_24x24_hr(position_encoding_type='array_factor', phi_endpoint=False):
    config = SimpleNamespace(
        sample_num=144,
        angle_steps_theta=91,
        angle_steps_phi=360,
        array_factor_steps_theta=24,
        array_factor_steps_phi=96,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


def beam196_28x28_hr(position_encoding_type='array_factor', phi_endpoint=False):
    config = SimpleNamespace(
        sample_num=196,
        angle_steps_theta=91,
        angle_steps_phi=360,
        array_factor_steps_theta=28,
        array_factor_steps_phi=112,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


def beam196_28x28(position_encoding_type='array_factor', phi_endpoint=False):
    # 28x28: array_factor_steps_theta=28, phi=112 -> input_array_sampling=3136
    # sample_num = (28/2)^2 = 196
    config = SimpleNamespace(
        sample_num=196,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=28,
        array_factor_steps_phi=112,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config



def beam144_24x24(position_encoding_type='array_factor', phi_endpoint=False):
    config = SimpleNamespace(
        sample_num=144,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=24,
        array_factor_steps_phi=96,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


def beam256_32x32_hr(position_encoding_type='array_factor', phi_endpoint=False):
    config = SimpleNamespace(
        sample_num=256,
        angle_steps_theta=91,
        angle_steps_phi=360,
        array_factor_steps_theta=32,
        array_factor_steps_phi=128,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


def beam256_32x32(position_encoding_type='array_factor', phi_endpoint=False):
    config = SimpleNamespace(
        sample_num=256,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=32,
        array_factor_steps_phi=128,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

def beam256_32x32_sar(position_encoding_type='array_factor', phi_endpoint=False):
    config = SimpleNamespace(
        sample_num=256,
        angle_steps_theta=20,
        angle_steps_phi=80,
        array_factor_steps_theta=32 * 2,
        array_factor_steps_phi=128 * 2,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config

def beam256_32x32_high_res(position_encoding_type='array_factor', phi_endpoint=False, query_valid_num=None):
    config = SimpleNamespace(
        sample_num=256,
        angle_steps_theta=20*4,
        angle_steps_phi=80*4,
        array_factor_steps_theta=32,
        array_factor_steps_phi=128,
        position_encoding_type=position_encoding_type,
        phi_endpoint=phi_endpoint,
        query_valid_num=query_valid_num,
    )
    config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi
    return config


