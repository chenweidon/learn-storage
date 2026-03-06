# pre_train.py
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import time

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import *

# === 导入你的模块 ===
# 请确保 unet.py 已经创建
from model import Turbo_LightFNO, UNet
from utils import HybridLoss,load_tm, physics_forward
from config import cfg


# ==========================================
# 可视化工具函数
# ==========================================
def save_visual_check(epoch, model, tm, speckle, gt_obj, save_dir):
    """
    可视化检查：
    1. Input Speckle (384或64)
    2. GT Object
    3. Pred Object
    4. Pred Speckle (Physics, 384)
    """
    model.eval()
    with torch.no_grad():
        # === 统一缩放输入到 64x64 ===
        # 无论 FNO 还是 UNet，都强制使用低分辨率输入以提升速度和公平性
        # 如果你后续想换回 384，只需注释掉这一行逻辑即可
        if speckle.shape[-1] != 64:
            net_input = F.interpolate(speckle, size=(64, 64), mode='bilinear', align_corners=False)
        else:
            net_input = speckle

        # 预测
        pred_obj = model(net_input)

        # 物理回推 (用于检查是否符合物理规律)
        # 注意：这里输出的一定是 384x384，由 TM 决定
        pred_speckle_phy = physics_forward(tm, pred_obj)

        # 转 Numpy 用于绘图
        def to_numpy(tensor):
            img = tensor[0, 0].cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img

        vis_in_sp = to_numpy(speckle)  # 显示原始高清散斑
        vis_gt_obj = to_numpy(gt_obj)
        vis_pred_obj = to_numpy(pred_obj)
        vis_pred_sp = to_numpy(pred_speckle_phy)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 标题显示当前模型
        model_name = getattr(cfg, 'MODEL_TYPE', 'FNO').upper()

        axes[0].imshow(vis_in_sp, cmap='gray')
        axes[0].set_title(f'Raw Input (384)\nEpoch {epoch}')
        axes[0].axis('off')

        axes[1].imshow(vis_gt_obj, cmap='gray')
        axes[1].set_title('GT Object')
        axes[1].axis('off')

        axes[2].imshow(vis_pred_obj, cmap='gray')
        axes[2].set_title(f'Pred ({model_name})\n(Min:{pred_obj.min():.2f} Max:{pred_obj.max():.2f})')
        axes[2].axis('off')

        axes[3].imshow(vis_pred_sp, cmap='gray')
        axes[3].set_title('Pred Speckle (Physics)')
        axes[3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

    model.train()


# ==========================================
# 主程序
# ==========================================
def main():
    # 获取模型类型，默认为 FNO
    model_type = getattr(cfg, 'MODEL_TYPE').lower()
    print(f"=== Start Pre-training (Mode: {model_type.upper()} | Input: 64x64) ===")
    start_time = time.time()
    # 1. 准备保存路径
    # 自动根据模型名创建不同文件夹，方便你对比
    save_vis_dir = os.path.join(cfg.RESULT_DIR, f'pre_train_{model_type}')
    os.makedirs(save_vis_dir, exist_ok=True)

    if not os.path.exists(cfg.SYN_DATA_PATH):
        print(f"Dataset not found at {cfg.SYN_DATA_PATH}")
        return

    # 2. 加载数据
    print("Loading dataset...")
    data_dict = torch.load(cfg.SYN_DATA_PATH)
    # dataset 里的 speckle 是 384 的，我们在取出来后实时 resize
    dataset = TensorDataset(data_dict['speckle'], data_dict['object'])
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # 3. 加载物理环境 (TM)
    tm = load_tm()

    # 4. 初始化模型
    if model_type == 'unet':
        print("Initializing U-Net (64x64)...")
        # U-Net 输入 64
        model = UNet(n_channels=1, n_classes=1).to(cfg.DEVICE)
    elif model_type == 'fno':
        print("Initializing Turbo_LightFNO (Force 64x64 input)...")
        # FNO 原本设计是任意尺寸，这里我们喂 64x64 给它
        model = Turbo_LightFNO().to(cfg.DEVICE)

    # 5. Loss 与 优化器
    # 注意：HybridLoss 需要 TM 来计算物理一致性
    loss_fn = HybridLoss(tm, lambda_data=1.0, lambda_phy=0.5, lambda_tv=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_PRETRAIN)

    # 6. 训练循环
    model.train()
    total_epochs = cfg.PRETRAIN_EPOCHS

    for epoch in trange(total_epochs, desc="Training", unit="epoch"):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch", leave=False)

        last_batch_data = None

        for speckle, obj in pbar:
            speckle = speckle.to(cfg.DEVICE)
            obj = obj.to(cfg.DEVICE)

            # 归一化散斑
            sp_max = speckle.amax(dim=(1, 2, 3), keepdim=True)
            speckle = speckle / (sp_max + 1e-8)

            # === 关键修改：强制缩放到 64x64 ===
            # 这会让 FNO 速度飞快，且与 U-Net 公平对比
            net_input = F.interpolate(speckle, size=(64, 64), mode='bilinear', align_corners=False)

            # Forward
            pred_obj = model(net_input)

            # Loss 计算
            # 注意：loss_fn 内部的 physics loss 需要用 pred_obj 算出 384 的散斑
            # 然后和原始的 384 speckle (target) 进行比较
            # 所以这里传给 loss_fn 的第三个参数必须是原始的高清 speckle
            loss, _ = loss_fn(pred_obj, obj, speckle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
            last_batch_data = (speckle, obj)

        # 定期可视化与保存
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(loader)
            if last_batch_data is not None:
                save_visual_check(epoch + 1, model, tm, last_batch_data[0], last_batch_data[1], save_vis_dir)

            # # 每 50 轮保存一次 checkpoint（防止断电白跑）
            # if (epoch + 1) % 50 == 0:
            #     ckpt_name = f'ckpt_{model_type}_epoch{epoch + 1}.pth'
            #     torch.save(model.state_dict(), os.path.join(cfg.RESULT_DIR, ckpt_name))

    # 7. 保存最终模型
    # 文件名区分 model_type
    elapsed = time.time() - start_time
    final_save_name = f'pretrained_{model_type}.pth'
    save_path = os.path.join(cfg.RESULT_DIR, final_save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Training Finished! Model saved to {save_path}")
    print(f"Total training time: {elapsed:.2f} seconds")


if __name__ == '__main__':
    main()