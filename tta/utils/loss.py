# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from piq import SSIMLoss
from .physics import physics_forward


class FrequencyLoss(nn.Module):
    """
    频谱损失 (Spectral Loss)
    用于压制 FNO 的高频网格伪影，替代 DCT 的功能
    """

    def __init__(self, weight=1.0):
        super(FrequencyLoss, self).__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # 1. FFT 变换到频域
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # 2. 取幅度谱 (我们只关心能量分布，消除高频噪点)
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        # 3. Log 变换 (平衡高低频能量差异)
        pred_log = torch.log(pred_amp + 1e-8)
        target_log = torch.log(target_amp + 1e-8)

        # 4. 频域 L1
        return self.weight * self.l1(pred_log, target_log)


class HybridLoss(nn.Module):
    def __init__(self, tm, lambda_data=1.0, lambda_ssim=0.5, lambda_phy=0.5, lambda_freq=0.1, lambda_tv=0.01):
        super(HybridLoss, self).__init__()
        self.tm = tm

        # 基础损失
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss(data_range=1.0)  # SSIM 越大越好，Loss = 1 - SSIM
        self.freq_loss = FrequencyLoss(weight=1.0)

        # 权重配置
        self.w_data = lambda_data  # 像素级 L1
        self.w_ssim = lambda_ssim  # 结构级 SSIM
        self.w_phy = lambda_phy  # 物理级 Consistency
        self.w_freq = lambda_freq  # 频域级 (去网格)
        self.w_tv = lambda_tv  # 空间级 (去噪)

    def tv_loss(self, img):
        """全变分损失"""
        b, c, h, w = img.size()
        h_tv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (h_tv + w_tv) / (b * c * h * w)

    def forward(self, pred_obj, gt_obj, input_speckle):
        """
        Input:
            pred_obj: [B, 1, 64, 64] 预测物体
            gt_obj:   [B, 1, 64, 64] 真实物体 (预训练时必须有)
            input_speckle: [B, 1, 384, 384] 真实/模拟散斑
        """
        loss_dict = {}
        total_loss = 0

        # === 1. 有监督部分 (Object Domain) ===
        if gt_obj is not None:
            # A. L1 Loss (像素准确)
            l_data = self.l1(pred_obj, gt_obj)
            total_loss += self.w_data * l_data
            loss_dict['l1'] = l_data.item()

            # B. SSIM Loss (结构准确) [你要求的]
            l_ssim = self.ssim_loss(pred_obj, gt_obj)
            total_loss += self.w_ssim * l_ssim
            loss_dict['ssim'] = l_ssim.item()

            # C. Frequency Loss (去网格) [你要求的]
            l_freq = self.freq_loss(pred_obj, gt_obj)
            total_loss += self.w_freq * l_freq
            loss_dict['freq'] = l_freq.item()

        # === 2. 自监督部分 (Physics Domain) ===
        # D. Physics Loss (TM 约束)
        pred_speckle = physics_forward(self.tm, pred_obj)

        # 归一化对齐
        if pred_speckle.max() > 1e-8:
            pred_speckle = pred_speckle / pred_speckle.max()
        if input_speckle.max() > 1e-8:
            target_speckle = input_speckle / input_speckle.max()
        else:
            target_speckle = input_speckle

        l_phy = self.l1(pred_speckle, target_speckle)
        total_loss += self.w_phy * l_phy
        loss_dict['phy'] = l_phy.item()

        # === 3. 正则化部分 ===
        # E. TV Loss (平滑约束)
        l_tv = self.tv_loss(pred_obj)
        total_loss += self.w_tv * l_tv
        loss_dict['tv'] = l_tv.item()

        return total_loss, loss_dict