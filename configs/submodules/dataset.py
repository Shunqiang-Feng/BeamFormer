from types import SimpleNamespace
import os
from scipy.constants import c as light_speed

DATA_FOLDER = "./csi-dataset"

def sivers(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'sivers',
        train_data_path=f"{DATA_FOLDER}/uva-communication-28G-csi-sivers/t4x4_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/uva-communication-28G-csi-sivers/t4x4_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 4,
        N_tx = 4,
        M_rx = 2,
        N_rx = 1,
        max_theta=40,
        # max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.76381174338948853996811294564395*light_speed/mid_freq
    ds.d_col = 0.76381174338948853996811294564395*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds


def homeoffice_communication_28g(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'homeoffice_communication_28g',
        train_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds


def homeoffice_communication_28g_24x24(add_noise = False,  snr_min = None):
    # Cropped from 32x32 data: keeps top-left 24x24 TX antennas
    ds = SimpleNamespace(
        name = 'homeoffice_communication_28g_24x24',
        train_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t32x32_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t32x32_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 24,
        N_tx = 24,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
        crop = SimpleNamespace(M_tx_original=32, N_tx_original=32),
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds


def homeoffice_communication_28g_32x32(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'homeoffice_communication_28g_32x32',
        train_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t32x32_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t32x32_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 32,
        N_tx = 32,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds


def classroom_communication_28g(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'classroom_communication_28g',
        train_data_path=f"{DATA_FOLDER}/classroom-communication-28G-csi/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/classroom-communication-28G-csi/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds

def nyc_communication_28g(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'nyc_communication_28g',
        train_data_path=f"{DATA_FOLDER}/nyc-communication-28G-csi/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/nyc-communication-28G-csi/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds

def uva_communication_28g(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'uva_communication_28g',
        train_data_path=f"{DATA_FOLDER}/uva-communication-28G-csi/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/uva-communication-28G-csi/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds

def uva_communication_28g_patch(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'uva_communication_28g_patch',
        train_data_path=f"{DATA_FOLDER}/uva-communication-28G-csi-patch/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/uva-communication-28G-csi-patch/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds

def rural_communication_28g(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'rural_communication_28g',
        train_data_path=f"{DATA_FOLDER}/rural-communication-28G-csi/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/rural-communication-28G-csi/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds


def uva_communication_60g(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'uva_communication_60g',
        train_data_path=f"{DATA_FOLDER}/uva-communication-60G-csi/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/uva-communication-60G-csi/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=59.6e9,
        end_freq=61.34625e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 60.48e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds



# 14 G
def sivers_indoor(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'sivers',
        train_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi-sivers-indoor-patch/t4x4_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi-sivers-indoor-patch/t4x4_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 4,
        N_tx = 4,
        M_rx = 2,
        N_rx = 1,
        max_theta=40,
        # max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx
    mid_freq = 27.925e9
    ds.d_row = 0.76381174338948853996811294564395*light_speed/mid_freq
    ds.d_col = 0.76381174338948853996811294564395*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)
    return ds


def ibm_indoor(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'ibm',
        train_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi-ibm-indoor/t8x8_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi-ibm-indoor/t8x8_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 8,
        N_tx = 8,
        M_rx = 2,
        N_rx = 1,
        max_theta=60,
        # max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx
    mid_freq = 27.925e9
    ds.d_row = 0.45931775908785538176104523699905*light_speed/mid_freq
    ds.d_col = 0.45931775908785538176104523699905*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)
    return ds


def uva_communication_14g(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'uva_communication_14g',
        train_data_path=f"{DATA_FOLDER}/uva-communication-14G-csi/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/uva-communication-14G-csi/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=14.8e9,
        end_freq=15.35e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 15.075e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds


def homeoffice_communication_28g_20x20(add_noise = False, snr_min = None):
    # Cropped from 32x32 data: keeps top-left 20x20 TX antennas
    ds = SimpleNamespace(
        name = 'homeoffice_communication_28g_20x20',
        train_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t32x32_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t32x32_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 20,
        N_tx = 20,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
        crop = SimpleNamespace(M_tx_original=32, N_tx_original=32),
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds


def homeoffice_communication_28g_28x28(add_noise = False, snr_min = None):
    # Cropped from 32x32 data: keeps top-left 28x28 TX antennas
    ds = SimpleNamespace(
        name = 'homeoffice_communication_28g_28x28',
        train_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t32x32_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t32x32_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 28,
        N_tx = 28,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
        crop = SimpleNamespace(M_tx_original=32, N_tx_original=32),
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds


def diy_28G_csi_16x16(add_noise=False, snr_min=None):
    ds = SimpleNamespace(
        name='diy_28G_csi_16x16',
        train_data_path=f"{DATA_FOLDER}/diy_28G_csi/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/diy_28G_csi/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx=16,
        N_tx=16,
        M_rx=2,
        N_rx=1,
        max_theta=90,
        add_noise=add_noise,
        snr_min=snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5 * light_speed / mid_freq
    ds.d_col = 0.5 * light_speed / mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq) / (ds.freq_num - 1)

    return ds


def homeoffice_communication_28g_nlos(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'homeoffice_communication_28g_nlos',
        train_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi-nlos/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi-nlos/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds

def uva_communication_28g_nlos(add_noise = False,  snr_min = None):
    ds = SimpleNamespace(
        name = 'uva_communication_28g_nlos',
        train_data_path=f"{DATA_FOLDER}/uva-communication-28G-csi-nlos/t16x16_r2x1_train",
        test_data_path=f"{DATA_FOLDER}/uva-communication-28G-csi-nlos/t16x16_r2x1_test_small",
        mode="rx_act1",
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx = 16,
        N_tx = 16,
        M_rx = 2,
        N_rx = 1,
        max_theta=90,
        add_noise = add_noise,
        snr_min = snr_min,
    )
    if ds.mode == "rx_act1":
        ds.M = ds.M_tx
        ds.N = ds.N_tx
    else:
        ds.M = ds.M_rx
        ds.N = ds.N_rx

    mid_freq = 27.925e9
    ds.d_row = 0.5*light_speed/mid_freq
    ds.d_col = 0.5*light_speed/mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq)/(ds.freq_num-1)

    return ds
