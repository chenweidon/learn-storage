import torch
import torch.nn as nn
from piq import SSIMLoss
from .physics import physics_forward


class FrequencyLoss(nn.Module):
    """频谱损失，用于压制高频网格伪影。"""

    def __init__(self, weight=1.0):
        super(FrequencyLoss, self).__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        pred_log = torch.log(pred_amp + 1e-8)
        target_log = torch.log(target_amp + 1e-8)
        return self.weight * self.l1(pred_log, target_log)


class HybridLoss(nn.Module):
    def __init__(self, tm, lambda_data=1.0, lambda_ssim=0.5, lambda_phy=0.5, lambda_freq=0.1, lambda_tv=0.01):
        super(HybridLoss, self).__init__()
        self.tm = tm
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss(data_range=1.0)
        self.freq_loss = FrequencyLoss(weight=1.0)

        self.w_data = lambda_data
        self.w_ssim = lambda_ssim
        self.w_phy = lambda_phy
        self.w_freq = lambda_freq
        self.w_tv = lambda_tv

    def tv_loss(self, img):
        b, c, h, w = img.size()
        h_tv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (h_tv + w_tv) / (b * c * h * w)

    def forward(self, pred_obj, gt_obj, input_speckle):
        """
        pred_obj: [B, 1, cfg.IMG_SIZE, cfg.IMG_SIZE]
        gt_obj:   [B, 1, cfg.IMG_SIZE, cfg.IMG_SIZE] 或 None
        input_speckle: [B, 1, cfg.SPECKLE_SIZE, cfg.SPECKLE_SIZE]
        """
        loss_dict = {}
        total_loss = 0

        if gt_obj is not None:
            l_data = self.l1(pred_obj, gt_obj)
            total_loss += self.w_data * l_data
            loss_dict['l1'] = l_data.item()

            l_ssim = self.ssim_loss(pred_obj, gt_obj)
            total_loss += self.w_ssim * l_ssim
            loss_dict['ssim'] = l_ssim.item()

            l_freq = self.freq_loss(pred_obj, gt_obj)
            total_loss += self.w_freq * l_freq
            loss_dict['freq'] = l_freq.item()

        pred_speckle = physics_forward(self.tm, pred_obj)

        # 保持当前仓库行为：再做一次整体 max 对齐
        if pred_speckle.max() > 1e-8:
            pred_speckle = pred_speckle / pred_speckle.max()
        if input_speckle.max() > 1e-8:
            target_speckle = input_speckle / input_speckle.max()
        else:
            target_speckle = input_speckle

        l_phy = self.l1(pred_speckle, target_speckle)
        total_loss += self.w_phy * l_phy
        loss_dict['phy'] = l_phy.item()

        l_tv = self.tv_loss(pred_obj)
        total_loss += self.w_tv * l_tv
        loss_dict['tv'] = l_tv.item()

        return total_loss, loss_dict
