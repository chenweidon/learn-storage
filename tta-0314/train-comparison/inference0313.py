##动态控制是否要推理，将网络的输入改回散斑尺寸
import os
import sys
import copy
import time
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from piq import ssim as compute_ssim

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from model import Turbo_LightFNO, UNet
from utils import load_tm, physics_forward
from config import cfg


def compute_pcc(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    pcc = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
    )
    return pcc.item()


def save_comparison_plot(save_path, input_speckle, gt_obj, pred_obj, pred_speckle_phy, ssim_val, pcc_val):
    def to_numpy(tensor):
        return tensor.squeeze().detach().cpu().numpy()

    img_in_sp = to_numpy(input_speckle)
    img_gt = to_numpy(gt_obj)
    img_pred = to_numpy(pred_obj)
    img_pred_sp = to_numpy(pred_speckle_phy)

    p_min, p_max = img_pred.min(), img_pred.max()
    if p_max - p_min > 1e-6:
        img_pred_vis = (img_pred - p_min) / (p_max - p_min)
    else:
        img_pred_vis = img_pred

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    model_name = getattr(cfg, 'MODEL_TYPE', 'FNO').upper()

    axes[0].imshow(img_in_sp, cmap='gray')
    axes[0].set_title(f'Input Speckle ({input_speckle.shape[-1]})')
    axes[0].axis('off')

    axes[1].imshow(img_gt, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'GT Object ({gt_obj.shape[-1]})')
    axes[1].axis('off')

    axes[2].imshow(img_pred_vis, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Pred ({model_name})\nSSIM={ssim_val:.4f} | PCC={pcc_val:.4f}')
    axes[2].axis('off')

    axes[3].imshow(img_pred_sp, cmap='gray')
    axes[3].set_title(f'Pred Speckle (Physics {pred_speckle_phy.shape[-1]})')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def normalize_real_input(ts_in):
    # 保持你原始 GitHub 推理行为：real input 用 min-max
    return (ts_in - ts_in.min()) / (ts_in.max() - ts_in.min() + 1e-8)


def ensure_obj_size(pred_obj):
    if pred_obj.shape[-2:] != (cfg.IMG_SIZE, cfg.IMG_SIZE):
        pred_obj = F.interpolate(
            pred_obj,
            size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
            mode='bilinear',
            align_corners=False
        )
    return pred_obj


def prepare_model_input(input_speckle_native):
    # 保持你当前版本的行为：网络输入仍然压到 IMG_SIZE
    input_low = F.interpolate(
        input_speckle_native,
        size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
        mode='bilinear',
        align_corners=False,
    )
    return input_low


def direct_predict(model, base_state_dict, input_speckle_native):
    model.load_state_dict(base_state_dict)
    model.eval()

    with torch.no_grad():
        input_low = prepare_model_input(input_speckle_native)
        pred_obj = model(input_low)
        pred_obj = ensure_obj_size(pred_obj)

    return pred_obj.detach()


def compute_initial_physics_residual(tm, pred_obj, target_native):
    with torch.no_grad():
        pred_speckle_sim = physics_forward(tm, pred_obj)
        numerator = torch.sum(pred_speckle_sim * target_native)
        denominator = torch.sum(pred_speckle_sim ** 2)
        alpha = numerator / (denominator + 1e-8)
        alpha = torch.clamp(alpha, min=1e-3, max=1e3)
        scaled_sim = pred_speckle_sim * alpha
        r0 = F.l1_loss(scaled_sim, target_native).item()
    return r0


def tta_process(model, base_state_dict, tm, input_speckle_native, max_steps=100, tol=5e-2, patience=3):
    """
    尽量保持你原始 GitHub 版 TTA 行为。
    """
    model.load_state_dict(base_state_dict)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    target_native = input_speckle_native
    input_low = prepare_model_input(input_speckle_native)

    prev_loss = float('inf')
    plateau_count = 0
    actual_steps = 0

    for step in range(max_steps):
        pred_obj = model(input_low)
        pred_obj = ensure_obj_size(pred_obj)

        pred_speckle_sim = physics_forward(tm, pred_obj)

        numerator = torch.sum(pred_speckle_sim * target_native)
        denominator = torch.sum(pred_speckle_sim ** 2)
        alpha = numerator / (denominator + 1e-8)
        alpha = torch.clamp(alpha, min=1e-3, max=1e3)
        scaled_sim = pred_speckle_sim * alpha.detach()

        loss = criterion(scaled_sim, target_native)
        current_loss = loss.item()

        if step > 0:
            rel_change = abs(prev_loss - current_loss) / (prev_loss + 1e-8)
            if rel_change < tol:
                plateau_count += 1
            else:
                plateau_count = 0
            if plateau_count >= patience:
                break

        prev_loss = current_loss
        actual_steps = step + 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return pred_obj.detach(), actual_steps


def normalize_stem(filename: str) -> str:
    stem = os.path.splitext(filename)[0].lower().strip()

    prefixes = ['input_', 'in_', 'img_', 'speckle_', 'label_', 'gt_', 'target_', 'obj_']
    suffixes = ['_input', '_in', '_img', '_speckle', '_label', '_gt', '_target', '_obj']

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if stem.startswith(p):
                stem = stem[len(p):]
                changed = True
        for s in suffixes:
            if stem.endswith(s):
                stem = stem[:-len(s)]
                changed = True
    return stem


def build_pairs(input_dir, label_dir, valid_ext):
    input_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(valid_ext)
    ]
    label_files = [
        f for f in os.listdir(label_dir)
        if os.path.isfile(os.path.join(label_dir, f)) and f.lower().endswith(valid_ext)
    ]

    input_map = {normalize_stem(f): f for f in input_files}
    label_map = {normalize_stem(f): f for f in label_files}
    common_keys = sorted(set(input_map) & set(label_map))
    pairs = [(input_map[k], label_map[k]) for k in common_keys]

    if len(pairs) == 0 and len(input_files) == len(label_files) and len(input_files) > 0:
        print('[Warn] No same-stem pairs found. Fallback to sorted zip pairing.')
        pairs = list(zip(sorted(input_files), sorted(label_files)))

    return pairs


def load_real_pairs():
    input_dir = os.path.join(cfg.TEST_DATA_PATH, 'input')
    label_dir = os.path.join(cfg.TEST_DATA_PATH, 'label')
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    if not os.path.exists(input_dir) or not os.path.exists(label_dir):
        return input_dir, label_dir, []

    pairs = build_pairs(input_dir, label_dir, valid_ext)
    return input_dir, label_dir, pairs


def build_model(model_type: str):
    model_type = model_type.lower()
    if model_type == 'unet':
        return UNet(n_channels=1, n_classes=1).to(cfg.DEVICE)
    if model_type == 'fno':
        return Turbo_LightFNO().to(cfg.DEVICE)
    raise ValueError(f'Unsupported MODEL_TYPE: {model_type}')


def run_direct_mode(model, base_state_dict, tm, ts_in):
    pred_obj = direct_predict(model, base_state_dict, ts_in)
    r0 = compute_initial_physics_residual(tm, pred_obj, ts_in)
    return pred_obj, 0, r0, 'direct'


def run_all_tta_mode(model, base_state_dict, tm, ts_in):
    max_steps = int(getattr(cfg, 'EVAL_ALL_TTA_STEPS', 100))
    tol = float(getattr(cfg, 'TTA_TOL', 5e-2))
    patience = int(getattr(cfg, 'TTA_PATIENCE', 3))

    pred0 = direct_predict(model, base_state_dict, ts_in)
    r0 = compute_initial_physics_residual(tm, pred0, ts_in)

    pred_obj, acc_step = tta_process(
        model, base_state_dict, tm, ts_in,
        max_steps=max_steps, tol=tol, patience=patience
    )
    return pred_obj, acc_step, r0, 'tta'


def run_gated_mode(model, base_state_dict, tm, ts_in):
    gate_enable = getattr(cfg, 'TTA_GATE_ENABLE', True)

    tau_skip = float(getattr(cfg, 'TTA_GATE_TAU_SKIP', 0.03))
    tau_light = float(getattr(cfg, 'TTA_GATE_TAU_LIGHT', 0.08))
    light_steps = int(getattr(cfg, 'TTA_GATE_LIGHT_STEPS', 10))
    full_steps = int(getattr(cfg, 'TTA_GATE_FULL_STEPS', 100))
    tol = float(getattr(cfg, 'TTA_TOL', 5e-2))
    patience = int(getattr(cfg, 'TTA_PATIENCE', 3))

    pred0 = direct_predict(model, base_state_dict, ts_in)
    r0 = compute_initial_physics_residual(tm, pred0, ts_in)

    if not gate_enable:
        pred_obj, acc_step = tta_process(
            model, base_state_dict, tm, ts_in,
            max_steps=full_steps, tol=tol, patience=patience
        )
        return pred_obj, acc_step, r0, 'full'

    if r0 < tau_skip:
        return pred0, 0, r0, 'skip'

    if r0 < tau_light:
        pred_obj, acc_step = tta_process(
            model, base_state_dict, tm, ts_in,
            max_steps=light_steps, tol=tol, patience=patience
        )
        return pred_obj, acc_step, r0, 'light'

    pred_obj, acc_step = tta_process(
        model, base_state_dict, tm, ts_in,
        max_steps=full_steps, tol=tol, patience=patience
    )
    return pred_obj, acc_step, r0, 'full'


def run_one_sample(eval_mode, model, base_state_dict, tm, ts_in):
    eval_mode = eval_mode.lower()

    if eval_mode == 'direct':
        return run_direct_mode(model, base_state_dict, tm, ts_in)
    elif eval_mode == 'tta':
        return run_all_tta_mode(model, base_state_dict, tm, ts_in)
    elif eval_mode == 'gated':
        return run_gated_mode(model, base_state_dict, tm, ts_in)
    else:
        raise ValueError(f'Unsupported EVAL_MODE: {eval_mode}')


def main():
    model_type = getattr(cfg, 'MODEL_TYPE').lower()
    eval_mode = getattr(cfg, 'EVAL_MODE', 'gated').lower()

    print(f'=== Start Inference | Mode={eval_mode.upper()} | Model={model_type.upper()} | Object={cfg.IMG_SIZE} | Speckle={cfg.SPECKLE_SIZE} ===')

    tm_start = time.time()
    tm = load_tm()
    tm_load_time = time.time() - tm_start
    print(f'TM loading time: {tm_load_time:.4f} seconds')

    model = build_model(model_type)

    weights_path = os.path.join(cfg.weight_path, f'pretrained_{model_type}.pth')
    if not os.path.exists(weights_path):
        print(f'Error: 权重文件未找到: {weights_path}')
        return

    state_dict = torch.load(weights_path, map_location=cfg.DEVICE)
    model.load_state_dict(state_dict)
    base_state_dict = copy.deepcopy(state_dict)
    print(f'Model loaded from: {weights_path}')

    transform_in = transforms.Compose([
        transforms.Resize((cfg.SPECKLE_SIZE, cfg.SPECKLE_SIZE)),
        transforms.ToTensor(),
    ])
    transform_gt = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    input_dir, label_dir, pairs = load_real_pairs()
    if len(pairs) == 0:
        print(f'No valid input/label pairs found under: {cfg.TEST_DATA_PATH}')
        return

    save_vis_dir = os.path.join(cfg.RESULT_DIR, f'{eval_mode}_vis_{model_type}\\5')
    os.makedirs(save_vis_dir, exist_ok=True)
    print(f'Results will be saved to: {save_vis_dir}')

    total_time = 0.0
    total_ssim = 0.0
    total_pcc = 0.0
    total_step = 0.0
    total_r0 = 0.0

    route_count = {}

    for i, (f_in, f_gt) in enumerate(pairs):
        try:
            img_in = Image.open(os.path.join(input_dir, f_in)).convert('L')
            img_gt = Image.open(os.path.join(label_dir, f_gt)).convert('L')

            ts_in = transform_in(img_in).to(cfg.DEVICE).unsqueeze(0)
            ts_gt = transform_gt(img_gt).to(cfg.DEVICE).unsqueeze(0)

            ts_in = normalize_real_input(ts_in)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            pred_obj, acc_step, r0, route = run_one_sample(eval_mode, model, base_state_dict, tm, ts_in)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dur = time.time() - start_time

            total_time += dur
            total_step += acc_step
            total_r0 += r0
            route_count[route] = route_count.get(route, 0) + 1

            with torch.no_grad():
                pred_speckle_phy = physics_forward(tm, pred_obj)

            val_ssim = compute_ssim(pred_obj, ts_gt, data_range=1.0).item()
            val_pcc = compute_pcc(pred_obj, ts_gt)

            total_ssim += val_ssim
            total_pcc += val_pcc

            print(
                f'[{i + 1}/{len(pairs)}] {f_in} | '
                f'Route: {route} | r0: {r0:.6f} | '
                f'SSIM: {val_ssim:.4f} | PCC: {val_pcc:.4f} | '
                f'Time: {dur:.4f}s | Step: {acc_step}'
            )

            if i < 5:
                save_name = os.path.join(save_vis_dir, f"vis_{f_in.split('.')[0]}.png")
                save_comparison_plot(save_name, ts_in, ts_gt, pred_obj, pred_speckle_phy, val_ssim, val_pcc)

        except Exception as e:
            print(f"Error processing {f_in}: {e}")
            traceback.print_exc()

    num_samples = len(pairs)
    avg_time = total_time / num_samples
    avg_ssim = total_ssim / num_samples
    avg_pcc = total_pcc / num_samples
    avg_step = total_step / num_samples
    avg_r0 = total_r0 / num_samples

    print("=" * 60)
    print(
        f"Model: {model_type.upper()} | Mode: {eval_mode.upper()} | "
        f"Avg SSIM: {avg_ssim:.4f} | Avg PCC: {avg_pcc:.4f} | "
        f"Avg Time: {avg_time:.4f}s | Avg Step: {avg_step:.2f} | Avg r0: {avg_r0:.6f}"
    )

    route_info = " | ".join([f"{k}: {v}" for k, v in route_count.items()])
    print(f"Route Count | {route_info}")
    print(f"TM loading time: {tm_load_time:.4f} seconds")


if __name__ == '__main__':
    main()