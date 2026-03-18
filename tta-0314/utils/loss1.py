import torch
import torch.nn as nn
from piq import SSIMLoss
from .physics import physics_forward
import torch.nn.functional as F


def pcc_loss(pred, target, eps=1e-8):
    pred_mean = pred.mean(dim=(1, 2, 3), keepdim=True)
    target_mean = target.mean(dim=(1, 2, 3), keepdim=True)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    numerator = (pred_centered * target_centered).sum(dim=(1, 2, 3))
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=(1, 2, 3)) *
        (target_centered ** 2).sum(dim=(1, 2, 3)) + eps
    )
    pcc = numerator / (denominator + eps)
    return (1.0 - pcc).mean()


def sobel_edge_map(x):
    kernel_x = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        dtype=x.dtype, device=x.device
    ).unsqueeze(0)

    kernel_y = torch.tensor(
        [[[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]],
        dtype=x.dtype, device=x.device
    ).unsqueeze(0)

    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def edge_loss(pred, target):
    return F.l1_loss(sobel_edge_map(pred), sobel_edge_map(target))


def physics_data_loss(tm, pred_obj, input_speckle, pcc_weight=0.2):
    pred_speckle = physics_forward(tm, pred_obj)
    loss_l1 = F.l1_loss(pred_speckle, input_speckle)
    loss_pcc = pcc_loss(pred_speckle, input_speckle)
    return loss_l1 + pcc_weight * loss_pcc

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



class WarmRefineLoss(nn.Module):
    """
    对 x0 做弱监督，对 xK 做强监督，再加 refined physics consistency
    """
    def __init__(
        self,
        ssim_loss_fn,
        w_direct=0.2,
        w_refined=1.0,
        w_phy=0.3,
        direct_ssim_weight=0.2,
        refined_ssim_weight=0.8,
        refined_edge_weight=0.2,
        phy_pcc_weight=0.2,
    ):
        super().__init__()
        self.ssim_loss_fn = ssim_loss_fn
        self.w_direct = w_direct
        self.w_refined = w_refined
        self.w_phy = w_phy
        self.direct_ssim_weight = direct_ssim_weight
        self.refined_ssim_weight = refined_ssim_weight
        self.refined_edge_weight = refined_edge_weight
        self.phy_pcc_weight = phy_pcc_weight

    def forward(self, tm, x0, xk, gt_obj, input_speckle):
        # direct 弱监督
        loss_direct = (
            F.l1_loss(x0, gt_obj)
            + self.direct_ssim_weight * self.ssim_loss_fn(x0, gt_obj)
        )

        # refined 强监督
        loss_refined = (
            F.l1_loss(xk, gt_obj)
            + self.refined_ssim_weight * self.ssim_loss_fn(xk, gt_obj)
            + self.refined_edge_weight * edge_loss(xk, gt_obj)
        )

        # refined physics consistency
        loss_phy = physics_data_loss(
            tm, xk, input_speckle, pcc_weight=self.phy_pcc_weight
        )

        total = (
            self.w_direct * loss_direct
            + self.w_refined * loss_refined
            + self.w_phy * loss_phy
        )

        parts = {
            "loss_direct": loss_direct.detach().item(),
            "loss_refined": loss_refined.detach().item(),
            "loss_phy": loss_phy.detach().item(),
            "loss_total": total.detach().item(),
        }
        return total, parts
