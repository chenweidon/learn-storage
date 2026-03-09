import sys
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange, tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from model import Turbo_LightFNO, UNet
from utils import HybridLoss, load_tm, physics_forward
from config import cfg


def build_model(model_type: str):
    model_type = model_type.lower()
    if model_type == 'unet':
        return UNet(n_channels=1, n_classes=1).to(cfg.DEVICE)
    if model_type == 'fno':
        return Turbo_LightFNO().to(cfg.DEVICE)
    raise ValueError(f'Unsupported MODEL_TYPE: {model_type}')


def save_visual_check(epoch, model, tm, speckle_native, gt_obj, save_dir):
    """保持原始预训练流程，只把标题/变量名改成动态尺寸。"""
    model.eval()
    with torch.no_grad():
        if speckle_native.shape[-1] != cfg.IMG_SIZE:
            net_input = F.interpolate(
                speckle_native,
                size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
                mode='bilinear',
                align_corners=False,
            )
        else:
            net_input = speckle_native

        pred_obj = model(net_input)
        pred_speckle_phy = physics_forward(tm, pred_obj)

        def to_numpy(tensor):
            img = tensor[0, 0].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img

        vis_in_sp = to_numpy(speckle_native)
        vis_gt_obj = to_numpy(gt_obj)
        vis_pred_obj = to_numpy(pred_obj)
        vis_pred_sp = to_numpy(pred_speckle_phy)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        model_name = getattr(cfg, 'MODEL_TYPE', 'FNO').upper()

        axes[0].imshow(vis_in_sp, cmap='gray')
        axes[0].set_title(f'Raw Input ({speckle_native.shape[-1]})\nEpoch {epoch}')
        axes[0].axis('off')

        axes[1].imshow(vis_gt_obj, cmap='gray')
        axes[1].set_title(f'GT Object ({cfg.IMG_SIZE})')
        axes[1].axis('off')

        axes[2].imshow(vis_pred_obj, cmap='gray')
        axes[2].set_title(f'Pred ({model_name})')
        axes[2].axis('off')

        axes[3].imshow(vis_pred_sp, cmap='gray')
        axes[3].set_title(f'Pred Speckle (Physics {cfg.SPECKLE_SIZE})')
        axes[3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
    model.train()


def main():
    model_type = getattr(cfg, 'MODEL_TYPE').lower()
    print(f'=== Start Pre-training | Model={model_type.upper()} | Object={cfg.IMG_SIZE} | Speckle={cfg.SPECKLE_SIZE} ===')
    start_time = time.time()

    save_vis_dir = os.path.join(cfg.RESULT_DIR, f'pre_train_{model_type}')
    os.makedirs(save_vis_dir, exist_ok=True)

    if not os.path.exists(cfg.SYN_DATA_PATH):
        print(f'Dataset not found at {cfg.SYN_DATA_PATH}')
        return

    print('Loading dataset...')
    data_dict = torch.load(cfg.SYN_DATA_PATH)
    dataset = TensorDataset(data_dict['speckle'], data_dict['object'])
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    tm = load_tm()
    model = build_model(model_type)

    loss_fn = HybridLoss(tm, lambda_data=1.0, lambda_phy=0.5, lambda_tv=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_PRETRAIN)

    model.train()
    total_epochs = cfg.PRETRAIN_EPOCHS

    for epoch in trange(total_epochs, desc='Training', unit='epoch'):
        epoch_loss = 0.0
        last_batch_data = None
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}', unit='batch', leave=False)

        for speckle_native, obj in pbar:
            speckle_native = speckle_native.to(cfg.DEVICE)
            obj = obj.to(cfg.DEVICE)

            # 保持原始预训练行为：per-image / max
            sp_max = speckle_native.amax(dim=(1, 2, 3), keepdim=True)
            speckle_native = speckle_native / (sp_max + 1e-8)

            net_input = F.interpolate(
                speckle_native,
                size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
                mode='bilinear',
                align_corners=False,
            )

            pred_obj = model(net_input)
            loss, _ = loss_fn(pred_obj, obj, speckle_native)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            last_batch_data = (speckle_native, obj)

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(len(loader), 1)
            print(f'Epoch [{epoch + 1}/{total_epochs}] Avg Loss: {avg_loss:.6f}')
            if last_batch_data is not None:
                save_visual_check(epoch + 1, model, tm, last_batch_data[0], last_batch_data[1], save_vis_dir)

    final_save_name = f'pretrained_{model_type}.pth'
    final_save_path = os.path.join(cfg.RESULT_DIR, final_save_name)
    torch.save(model.state_dict(), final_save_path)

    total_time = time.time() - start_time
    print(f'Pre-training complete. Model saved to: {final_save_path}')
    print(f'Total training time: {total_time:.2f}s')


if __name__ == '__main__':
    main()
