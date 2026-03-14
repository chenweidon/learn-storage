##输入为散斑原尺寸
# pre_train.py
import sys
import os
import matplotlib.pyplot as plt
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import *

from model import Turbo_LightFNO1, UNet1
from utils import HybridLoss, load_tm, physics_forward
from config import cfg


# ==========================================
# 可视化工具函数
# ==========================================
def save_visual_check(epoch, model, tm, speckle, gt_obj, save_dir):
    """
    可视化检查：
    1. Input Speckle (native)
    2. GT Object
    3. Pred Object
    4. Pred Speckle (Physics, native)
    """
    model.eval()
    with torch.no_grad():
        # 关键修改：网络直接接收 native speckle 尺寸
        net_input = speckle

        pred_obj = model(net_input)
        pred_speckle_phy = physics_forward(tm, pred_obj)

        def to_numpy(tensor):
            img = tensor[0, 0].cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img

        vis_in_sp = to_numpy(speckle)
        vis_gt_obj = to_numpy(gt_obj)
        vis_pred_obj = to_numpy(pred_obj)
        vis_pred_sp = to_numpy(pred_speckle_phy)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        model_name = getattr(cfg, 'MODEL_TYPE', 'FNO').upper()

        axes[0].imshow(vis_in_sp, cmap='gray')
        axes[0].set_title(f'Raw Input ({speckle.shape[-1]})\nEpoch {epoch}')
        axes[0].axis('off')

        axes[1].imshow(vis_gt_obj, cmap='gray')
        axes[1].set_title('GT Object')
        axes[1].axis('off')

        axes[2].imshow(vis_pred_obj, cmap='gray')
        axes[2].set_title(f'Pred ({model_name})\n(Min:{pred_obj.min():.2f} Max:{pred_obj.max():.2f})')
        axes[2].axis('off')

        axes[3].imshow(vis_pred_sp, cmap='gray')
        axes[3].set_title(f'Pred Speckle (Physics {pred_speckle_phy.shape[-1]})')
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
    model_type = getattr(cfg, 'MODEL_TYPE').lower()
    print(f"=== Start Pre-training | Model={model_type.upper()} | Object={cfg.IMG_SIZE} | Speckle={cfg.SPECKLE_SIZE} ===")
    start_time = time.time()

    save_vis_dir = os.path.join(cfg.RESULT_DIR, f'pre_train_{model_type}')
    os.makedirs(save_vis_dir, exist_ok=True)

    if not os.path.exists(cfg.SYN_DATA_PATH):
        print(f"Dataset not found at {cfg.SYN_DATA_PATH}")
        return

    print("Loading dataset...")
    data_dict = torch.load(cfg.SYN_DATA_PATH)
    dataset = TensorDataset(data_dict['speckle'], data_dict['object'])
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    tm = load_tm()

    if model_type == 'unet':
        print(f"Initializing U-Net (native speckle -> {cfg.IMG_SIZE}x{cfg.IMG_SIZE})...")
        model = UNet1(n_channels=1, n_classes=1).to(cfg.DEVICE)
    elif model_type == 'fno':
        print(f"Initializing Turbo_LightFNO (native speckle -> {cfg.IMG_SIZE}x{cfg.IMG_SIZE})...")
        model = Turbo_LightFNO1().to(cfg.DEVICE)
    else:
        raise ValueError(f'Unsupported MODEL_TYPE: {model_type}')

    loss_fn = HybridLoss(tm, lambda_data=1.0, lambda_phy=0.5, lambda_tv=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_PRETRAIN)

    model.train()
    total_epochs = cfg.PRETRAIN_EPOCHS

    for epoch in trange(total_epochs, desc="Training", unit="epoch"):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch", leave=False)
        last_batch_data = None

        for speckle, obj in pbar:
            speckle = speckle.to(cfg.DEVICE)
            obj = obj.to(cfg.DEVICE)

            sp_max = speckle.amax(dim=(1, 2, 3), keepdim=True)
            speckle = speckle / (sp_max + 1e-8)

            # 关键修改：网络直接接收 native speckle 尺寸
            net_input = speckle

            pred_obj = model(net_input)
            loss, _ = loss_fn(pred_obj, obj, speckle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
            last_batch_data = (speckle, obj)

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch [{epoch + 1}/{total_epochs}] Avg Loss: {avg_loss:.6f}")
            if last_batch_data is not None:
                save_visual_check(epoch + 1, model, tm, last_batch_data[0], last_batch_data[1], save_vis_dir)

    elapsed = time.time() - start_time
    final_save_name = f'pretrained_{model_type}.pth'
    save_path = os.path.join(cfg.RESULT_DIR, final_save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Training Finished! Model saved to {save_path}")
    print(f"Total training time: {elapsed:.2f} seconds")


if __name__ == '__main__':
    main()
