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
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import *

from model import Turbo_LightFNO1, UNet1
from utils import HybridLoss, load_tm, physics_forward,WarmRefineLoss, SSIMLoss,unrolled_refine
from config import cfg




def normalize_speckle_max(x):
    x = x.float()
    x = x.clamp_min(0.0)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    return x / (x_max + 1e-8)


def simulate_camera_io(x, levels=255):
    """
    用于在训练时模拟 jpg/png/相机 8-bit 输入链路。
    先按样本 max 归一化，再量化，再还原为 float。
    """
    x = normalize_speckle_max(x)
    x = torch.round(x * levels) / levels
    return x


# @torch.no_grad()
def evaluate_loader(model,tm, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for speckle, obj in loader:
        speckle = speckle.to(cfg.DEVICE)
        obj = obj.to(cfg.DEVICE)

        speckle = normalize_speckle_max(speckle)
        if getattr(cfg, 'SIMULATE_CAMERA_IO', False):
            speckle = simulate_camera_io(speckle)

        # pred_obj = model(speckle)
        # loss, _ = loss_fn(pred_obj, obj, speckle)
        with torch.no_grad():
         pred_obj0 = model(speckle)
        pred_objK = unrolled_refine(
            tm=tm,
            x0=pred_obj0,
            input_speckle=speckle,
            num_steps=3,
            step_size=0.1,
        )

        loss, parts = loss_fn(
            tm=tm,
            x0=pred_obj0,
            xk=pred_objK,
            gt_obj=obj,
            input_speckle=speckle,
        )
        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


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
    print(
        f"=== Start Pre-training | Model={model_type.upper()} | Object={cfg.IMG_SIZE} | Speckle={cfg.SPECKLE_SIZE} ===")
    start_time = time.time()

    save_vis_dir = os.path.join(cfg.RESULT_DIR, f'pre_train_{model_type}')
    os.makedirs(save_vis_dir, exist_ok=True)

    if not os.path.exists(cfg.SYN_DATA_PATH):
        print(f"Dataset not found at {cfg.SYN_DATA_PATH}")
        return

    print("Loading dataset...")
    data_dict = torch.load(cfg.SYN_DATA_PATH)
    full_dataset = TensorDataset(data_dict['speckle'], data_dict['object'])

    val_size = max(1, int(len(full_dataset) * cfg.VAL_RATIO))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(getattr(cfg, 'SEED', 42))
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False)

    print(f"Train size: {len(train_set)} | Val size: {len(val_set)}")
    tm = load_tm()

    if model_type == 'unet':
        print(f"Initializing U-Net (native speckle -> {cfg.IMG_SIZE}x{cfg.IMG_SIZE})...")
        model = UNet1(n_channels=1, n_classes=1).to(cfg.DEVICE)
    elif model_type == 'fno':
        print(f"Initializing Turbo_LightFNO (native speckle -> {cfg.IMG_SIZE}x{cfg.IMG_SIZE})...")
        model = Turbo_LightFNO1().to(cfg.DEVICE)
    else:
        raise ValueError(f'Unsupported MODEL_TYPE: {model_type}')

    # loss_fn = HybridLoss(tm, lambda_data=1.0, lambda_phy=0.5, lambda_tv=0.01)
    ssim_criterion = SSIMLoss()
    loss_fn = WarmRefineLoss(
        ssim_loss_fn=ssim_criterion,
        w_direct=0.2,
        w_refined=1.0,
        w_phy=0.3,
        direct_ssim_weight=0.2,
        refined_ssim_weight=0.8,
        refined_edge_weight=0.2,
        phy_pcc_weight=0.2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_PRETRAIN)

    model.train()
    total_epochs = cfg.PRETRAIN_EPOCHS
    best_val_loss = float('inf')
    best_save_path = os.path.join(cfg.RESULT_DIR, getattr(cfg, 'BEST_MODEL_NAME', 'best_pretrained_fno.pth'))

    # 记录损失
    train_losses = []
    val_losses = []

    # 使用tqdm显示epoch进度
    epoch_pbar = tqdm(range(total_epochs), desc="Training Progress", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        last_batch_data = None

        # 内部batch循环不使用tqdm，或者使用更简洁的方式
        for speckle, obj in train_loader:
            speckle = speckle.to(cfg.DEVICE)
            obj = obj.to(cfg.DEVICE)

            speckle = normalize_speckle_max(speckle)
            if getattr(cfg, 'SIMULATE_CAMERA_IO', False):
                speckle = simulate_camera_io(speckle)

            net_input = speckle
            # pred_obj = model(net_input)
            # loss, _ = loss_fn(pred_obj, obj, speckle)
            pred_obj0 = model(net_input)

            pred_objK = unrolled_refine(
                tm=tm,
                x0=pred_obj0,
                input_speckle=speckle,
                num_steps=3,
                step_size=0.1,
            )

            loss, parts = loss_fn(
                tm=tm,
                x0=pred_obj0,
                xk=pred_objK,
                gt_obj=obj,
                input_speckle=speckle,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            last_batch_data = (speckle.detach(), obj.detach())

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss = evaluate_loader(model,tm, val_loader, loss_fn)

        # 记录损失
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        # 更新epoch进度条描述
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.6f}',
            'Val Loss': f'{val_loss:.6f}',
            "Dir": f"{parts['loss_direct']:.4f}",
            "Ref": f"{parts['loss_refined']:.4f}",
            "Phy": f"{parts['loss_phy']:.4f}",
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_save_path)

        if (epoch + 1) % 10 == 0 and last_batch_data is not None:
            save_visual_check(epoch + 1, model, tm, last_batch_data[0], last_batch_data[1], save_vis_dir)

    epoch_pbar.close()

    # 计算总训练时间
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    # 保存最终模型
    final_save_name = f'pretrained_{model_type}.pth'
    final_save_path = os.path.join(cfg.RESULT_DIR, final_save_name)
    torch.save(model.state_dict(), final_save_path)

    # 绘制并保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_epochs + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, total_epochs + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Curves - {model_type.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存损失曲线图
    loss_curve_path = os.path.join(cfg.RESULT_DIR, f'loss_curve_{model_type}.png')
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

    # 保存损失数据到txt文件
    loss_data_path = os.path.join(cfg.RESULT_DIR, f'loss_data_{model_type}.txt')
    with open(loss_data_path, 'w') as f:
        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Total epochs: {total_epochs}\n")
        f.write(f"Training time: {hours}h {minutes}m {seconds:.2f}s\n\n")
        f.write("Epoch\tTrain Loss\tVal Loss\n")
        for epoch in range(total_epochs):
            f.write(f"{epoch + 1}\t{train_losses[epoch]:.6f}\t{val_losses[epoch]:.6f}\n")

    print(f"\n{'=' * 60}")
    print(f"Training Finished!")
    print(f"Total training time: {hours}h {minutes}m {seconds:.2f}s")
    print(f"Final model saved to: {final_save_path}")
    print(f"Best model saved to: {best_save_path}")
    print(f"Loss curve saved to: {loss_curve_path}")
    print(f"Loss data saved to: {loss_data_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
