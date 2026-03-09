import torch
import numpy as np
import h5py
import math
from config import cfg


def load_tm():
    """加载传输矩阵，保持原始 GitHub 版本语义。"""
    print(f"Loading TM from {cfg.TM_PATH} ...")
    with h5py.File(cfg.TM_PATH, 'r') as f:
        keys = list(f.keys())
        tm_key = [k for k in keys if 'Tm' in k or 'TM' in k][0]
        TM_raw = f[tm_key]
        TM_np = np.array(TM_raw)

        if getattr(TM_np, 'dtype', None) is not None and TM_np.dtype.names is not None and 'real' in TM_np.dtype.names:
            B = np.zeros(TM_np.shape, dtype=np.complex64)
            B.real = TM_np['real']
            B.imag = TM_np['imag']
            TM = torch.from_numpy(B)
        else:
            TM = torch.from_numpy(TM_np.astype(np.complex64))

        # 保持原始版本行为：转置一次
        TM = TM.transpose(0, 1)
        TM = TM.to(cfg.DEVICE)

    print(f"TM Loaded. Shape: {TM.shape}")
    return TM


def phase_encoding(phase_img):
    """相位编码。

    Input:  [B, 1, cfg.IMG_SIZE, cfg.IMG_SIZE]
    Output: [B, cfg.IMG_SIZE*cfg.IMG_SIZE, 1]
    """
    b, c, h, w = phase_img.shape
    phase_flat = phase_img.view(b, -1).unsqueeze(-1)
    complex_vec = torch.exp(1j * phase_flat * math.pi * cfg.PHASE_RANGE)
    return complex_vec


def physics_forward(tm, phase_img):
    """物理前向传播，保持原始 GitHub 版本语义：默认返回已归一化 speckle。"""
    complex_vec = phase_encoding(phase_img)
    scattered_field = torch.matmul(tm, complex_vec)

    b = phase_img.shape[0]
    scattered_field = scattered_field.view(b, cfg.SPECKLE_SIZE, cfg.SPECKLE_SIZE)
    scattered_field = scattered_field.permute(0, 2, 1)
    scattered_field = scattered_field.unsqueeze(1)

    speckle = torch.abs(scattered_field) ** 2

    # 保持原始版本行为：每张图单独归一化
    sp_max = speckle.amax(dim=(1, 2, 3), keepdim=True)
    speckle = speckle / (sp_max + 1e-8)
    return speckle
