# inference.py
import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F  # 新增
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from piq import ssim as compute_ssim
from scipy.ndimage import median_filter

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 引入两个模型
from model import Turbo_LightFNO, UNet
from utils import load_tm, physics_forward
from config import cfg


# ==========================================
# 指标计算工具
# ==========================================
def compute_pcc(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return pcc.item()


# ==========================================
# 绘图工具 (自动亮度增强)
# ==========================================
def save_comparison_plot(save_path, input_sp, gt_obj, pred_obj, pred_sp_phy, ssim_val, pcc_val):
    def to_numpy(tensor):
        return tensor.squeeze().cpu().detach().numpy()

    img_in_sp = to_numpy(input_sp)
    img_gt = to_numpy(gt_obj)
    img_pred = to_numpy(pred_obj)
    img_pred_sp = to_numpy(pred_sp_phy)

    # 视觉增强：拉伸到 0-1 显示，方便人眼观察结构
    p_min, p_max = img_pred.min(), img_pred.max()
    if p_max - p_min > 1e-6:
        img_pred_vis = (img_pred - p_min) / (p_max - p_min)
    else:
        img_pred_vis = img_pred

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 标题显示当前模型
    model_name = getattr(cfg, 'MODEL_TYPE', 'FNO').upper()

    axes[0].imshow(img_in_sp, cmap='gray')
    axes[0].set_title(f"Input Speckle (384)")
    axes[0].axis('off')

    axes[1].imshow(img_gt, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("GT Object")
    axes[1].axis('off')

    axes[2].imshow(img_pred_vis, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Pred ({model_name})\nSSIM: {ssim_val:.4f} | PCC: {pcc_val:.4f}")
    axes[2].axis('off')

    axes[3].imshow(img_pred_sp, cmap='gray')
    axes[3].set_title("Pred Speckle (Physics 384)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==========================================
# TTA 核心逻辑 (Force 64 Input + 384 Loss)
# ==========================================
def tta_process(model, base_state_dict, tm, input_speckle_384, max_steps=100, tol=5e-2, patience=3):
    """
    input_speckle_384:[1, 1, 384, 384] 原始高清散斑
    max_steps: 允许的最大迭代次数 (放宽到50，由自适应机制决定何时停)
    tol: 相对 loss 变化的阈值参数
    patience: 容忍 loss 不降或下降微小的步数
    """
    # 1. 恢复权重
    model.load_state_dict(base_state_dict)

    # 2. 全参数解冻
    for param in model.parameters():
        param.requires_grad = True

    # 3. 优化器 & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.L1Loss()

    target_384 = input_speckle_384
    input_64 = F.interpolate(input_speckle_384, size=(64, 64), mode='bilinear', align_corners=False)

    prev_loss = float('inf')
    plateau_count = 0
    actual_steps = 0

    for step in range(max_steps):
        # A. 网络预测 (输入 64 -> 输出 64)
        pred_obj = model(input_64)

        # B. 物理前向 (输入 64 -> 输出 384)
        pred_speckle_sim_384 = physics_forward(tm, pred_obj)

        # C. 线性强度对齐 (物理标量对齐)
        numerator = torch.sum(pred_speckle_sim_384 * target_384)
        denominator = torch.sum(pred_speckle_sim_384 ** 2)
        alpha = numerator / (denominator + 1e-8)
        alpha = torch.clamp(alpha, min=1e-3, max=1e3)
        scaled_sim = pred_speckle_sim_384 * alpha.detach()

        # D. Loss 计算
        loss = criterion(scaled_sim, target_384)
        current_loss = loss.item()

        # E. 自适应停止逻辑 (Early Stopping)
        if step > 0:
            rel_change = abs(prev_loss - current_loss) / (prev_loss + 1e-8)
            # print(f"  Step {step}: Loss={current_loss:.6f} | Rel_Change={rel_change:.6f}")
            if rel_change < tol:
                plateau_count += 1
            else:
                plateau_count = 0  # 重新计算

        # 如果连续 patience 次 loss 变化小于阈值，说明物理逼近已达极限
        if plateau_count >= patience:
            # print(f"    [TTA Early Stopping] Converged at step {step + 1} (Rel change < {tol})")
            break

        prev_loss = current_loss
        actual_steps = step + 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 可以选择打印每张图实际的迭代步数，方便你写 Paper 时做统计分析
    # print(f"    -> Actual TTA steps: {actual_steps}")

    # 返回原始预测 (Raw Data)
    return pred_obj.detach(),actual_steps


# ==========================================
# 主程序
# ==========================================
def load_real_data():
    input_dir = os.path.join(cfg.TEST_DATA_PATH, 'input')
    label_dir = os.path.join(cfg.TEST_DATA_PATH, 'label')
    valid_ext = ('.png', '.jpg', '.bmp', '.tif')

    if not os.path.exists(input_dir):
        return [], [], [], []

    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)])
    label_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(valid_ext)])
    return input_dir, input_files, label_dir, label_files


def main():
    # 获取模型类型
    model_type = getattr(cfg, 'MODEL_TYPE').lower()
    print(f"=== Start TTA Inference (Mode: {model_type.upper()} | Input: 64x64) ===")

    tm = load_tm()

    # 1. 根据配置加载模型
    if model_type == 'unet':
        print("Loading U-Net (64x64)...")
        model = UNet(n_channels=1, n_classes=1).to(cfg.DEVICE)
    else:
        print("Loading Turbo_LightFNO (Force 64x64)...")
        model = Turbo_LightFNO().to(cfg.DEVICE)

    # 2. 自动匹配权重文件
    weights_path = os.path.join(cfg.weight_path, f'pretrained_{model_type}.pth')

    if not os.path.exists(weights_path):
        print(f"Error: 权重文件未找到: {weights_path}")
        print(f"请先运行 pre_train.py 并确保 MODEL_TYPE 设置正确。")
        return

    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    base_state_dict = copy.deepcopy(state_dict)
    print(f"Model loaded from {weights_path}")

    # 3. 数据预处理
    # 始终加载 384 的散斑 (transform_in)，降维操作在 tta_process 内部做
    transform_in = transforms.Compose([
        transforms.Resize((cfg.SPECKLE_SIZE, cfg.SPECKLE_SIZE)),  # 384
        transforms.ToTensor()
    ])
    transform_gt = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),  # 64
        transforms.ToTensor()
    ])

    input_dir, input_files, label_dir, label_files = load_real_data()

    # 结果保存路径
    save_vis_dir = os.path.join(cfg.RESULT_DIR, f'tta_vis_{model_type}\\5')
    os.makedirs(save_vis_dir, exist_ok=True)
    print(f"Results will be saved to: {save_vis_dir}")

    total_time = 0
    total_ssim = 0
    total_pcc = 0
    total_step = 0
    for i, (f_in, f_gt) in enumerate(zip(input_files, label_files)):
        try:
            img_in = Image.open(os.path.join(input_dir, f_in)).convert('L')
            img_gt = Image.open(os.path.join(label_dir, f_gt)).convert('L')

            # 输入 384x384
            ts_in = transform_in(img_in).to(cfg.DEVICE).unsqueeze(0)
            ts_gt = transform_gt(img_gt).to(cfg.DEVICE).unsqueeze(0)

            # 简单归一化
            ts_in = (ts_in - ts_in.min()) / (ts_in.max() - ts_in.min() + 1e-8)

            start_time = time.time()
            # === TTA 过程 ===
            # 这里传入的是 384 的 ts_in，函数内部会负责缩放到 64 给网络
            pred_obj,acc_step = tta_process(model, base_state_dict, tm, ts_in)
            ##修改
            # pred_np = pred_obj.squeeze().cpu().numpy()
            # pred_np_clean = median_filter(pred_np, size=3)
            # pred_obj = torch.from_numpy(pred_np_clean).view_as(pred_obj).to(cfg.DEVICE)

            dur = time.time() - start_time
            total_time += dur

            # 物理回推 (用于绘图，确认物理一致性)
            with torch.no_grad():
                pred_speckle_phy = physics_forward(tm, pred_obj)

            # === 评估 (基于 Raw Output) ===
            val_ssim = compute_ssim(pred_obj, ts_gt, data_range=1.0).item()
            val_pcc = compute_pcc(pred_obj, ts_gt)

            total_ssim += val_ssim
            total_pcc += val_pcc
            total_step += acc_step
            print(f"[{i + 1}/{len(input_files)}] {f_in} | SSIM: {val_ssim:.4f} | PCC: {val_pcc:.4f} | Time: {dur:.3f}s | Epoch: {acc_step}")

            if i < 5:  # 保存前20张
                save_name = os.path.join(save_vis_dir, f"vis_{f_in.split('.')[0]}.png")
                save_comparison_plot(save_name, ts_in, ts_gt, pred_obj, pred_speckle_phy, val_ssim, val_pcc)

        except Exception as e:
            print(f"Error processing {f_in}: {e}")
            import traceback
            traceback.print_exc()

    avg_time = total_time / len(input_files)
    avg_ssim = total_ssim / len(input_files)
    avg_pcc = total_pcc / len(input_files)
    avg_step = total_step / len(input_files)

    print("=" * 40)
    print(
        f"Model: {model_type.upper()} | Avg SSIM: {avg_ssim:.4f} | Avg PCC: {avg_pcc:.4f} | Avg Time: {avg_time:.4f}s | Avg Step: {avg_step}")


if __name__ == '__main__':
    main()