# utils_physics.py
import torch
import numpy as np
import h5py
import math
from config import cfg


def load_tm():
    """
    加载传输矩阵 - 严格复刻 test-cwd-ctm.py 的加载逻辑
    """
    print(f"Loading TM from {cfg.TM_PATH} ...")
    with h5py.File(cfg.TM_PATH, 'r') as f:
        # 1. 寻找 Key (兼容不同的命名)
        keys = list(f.keys())
        tm_key = [k for k in keys if 'Tm' in k or 'TM' in k][0]
        TM_raw = f[tm_key]
        TM_np = np.array(TM_raw)

        # 2. 构建复数矩阵 (复刻原代码: B.real = TM['real']...)
        # 即使 h5py 读出来是结构体，这样写最稳
        if 'real' in TM_np.dtype.names:
            B = np.zeros(TM_np.shape, dtype=np.complex64)
            B.real = TM_np['real']
            B.imag = TM_np['imag']
            TM = torch.from_numpy(B)
        else:
            # 如果已经是 complex
            TM = torch.from_numpy(TM_np.astype(np.complex64))

        # 3. 转置 (复刻原代码: TM = TM.transpose(0, 1))
        # 你的验证代码里加了这一句，所以这里必须加
        TM = TM.transpose(0, 1)
        TM = TM.to(cfg.DEVICE)

    print(f"TM Loaded. Shape: {TM.shape}")
    return TM


def phase_encoding(phase_img):
    """
    相位编码
    Input: [B, 1, 64, 64]
    Output: [B, 4096, 1]
    """
    b, c, h, w = phase_img.shape

    # === 复刻 test-cwd-ctm.py 逻辑 ===
    # 原代码: complex_label = complex_number.reshape((4096, 1))
    # 说明: 直接 Reshape (Row-Major)，不需要 Permute

    phase_flat = phase_img.view(b, -1).unsqueeze(-1)

    # 复数运算: exp(1j * phase * pi)
    complex_vec = 1 * torch.exp(1j * phase_flat * math.pi * cfg.PHASE_RANGE)

    return complex_vec


def physics_forward(tm, phase_img):
    """
    物理前向传播
    严格复刻 test-cwd-ctm.py 中的 train_one_epoch 逻辑
    """
    # 1. 编码
    complex_vec = phase_encoding(phase_img)  # [B, 4096, 1]

    # 2. 矩阵乘法
    # c_tensor = torch.mm(TM, complex_label)
    scattered_field = torch.matmul(tm, complex_vec)

    # 3. Reshape & Transpose (核心关键点!)
    b = phase_img.shape[0]

    # 原代码逻辑:
    # c_tensor = c_tensor.reshape(1, size, size)
    # c_tensor = c_tensor.transpose(1, 2) <--- 必须保留

    # 适配 Batch:
    # 先 reshape 成 [B, 384, 384]
    scattered_field = scattered_field.view(b, cfg.SPECKLE_SIZE, cfg.SPECKLE_SIZE)

    # 再 transpose (交换最后两个维度 H 和 W)
    scattered_field = scattered_field.permute(0, 2, 1)

    # 恢复通道维度 [B, 1, 384, 384]
    scattered_field = scattered_field.unsqueeze(1)

    # 4. 取强度
    speckle = torch.abs(scattered_field) ** 2

    # 5. 归一化 (复刻原代码: c_tensor = abs_square_matrix / max_value)
    # 为了梯度稳定，通常在 Loss 中做归一化，但这里也可以做
    # 注意: 原代码是每张图单独归一化
    sp_max = speckle.amax(dim=(1, 2, 3), keepdim=True)
    speckle = speckle / (sp_max + 1e-8)

    return speckle