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

from model import Turbo_LightFNO1, UNet1
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
    # 改成和训练一致：按样本 max 归一化
    ts_in = ts_in.float()
    ts_in = ts_in.clamp_min(0.0)
    return ts_in / (ts_in.max() + 1e-8)


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
    # 改成和训练一致：直接输入 native speckle 尺寸
    return input_speckle_native

# def configure_tta_scope(model, scope='head'):
#     # 先全部冻结
#     for p in model.parameters():
#         p.requires_grad = False
#
#     # 如果模型自己带 set_tta_mode，就优先用它
#     if hasattr(model, 'set_tta_mode'):
#         model.set_tta_mode(scope)
#     else:
#         if scope == 'full':
#             for p in model.parameters():
#                 p.requires_grad = True
#         elif scope == 'head':
#             for name, p in model.named_parameters():
#                 if any(key in name for key in ['fc0', 'fc1', 'fc2']):
#                     p.requires_grad = True
#         else:
#             raise ValueError(f"Unsupported TTA scope: {scope}")
#
#     trainable = [p for p in model.parameters() if p.requires_grad]
#     if len(trainable) == 0:
#         raise RuntimeError("No trainable parameters found for current TTA scope.")
#     return trainable

def pcc_loss_infer(pred, target, eps=1e-8):
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


def object_refine_loss(tm, pred_obj, target_native, pcc_weight=0.2, use_alpha=True):
    pred_speckle = physics_forward(tm, pred_obj)

    if use_alpha:
        numerator = torch.sum(pred_speckle * target_native)
        denominator = torch.sum(pred_speckle ** 2)
        alpha = numerator / (denominator + 1e-8)
        alpha = torch.clamp(alpha, min=1e-3, max=1e3)
        pred_speckle = pred_speckle * alpha

    loss_l1 = F.l1_loss(pred_speckle, target_native)
    loss_pcc = pcc_loss_infer(pred_speckle, target_native)
    return loss_l1 + pcc_weight * loss_pcc

def configure_tta_scope(model, scope='head'):
    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 1) 如果模型自己实现了 set_tta_mode，就优先走模型内部逻辑（主要给 FNO 用）
    if hasattr(model, 'set_tta_mode'):
        model.set_tta_mode(scope)

    # 2) 否则按模型结构名来开参数
    else:
        if scope == 'full':
            for p in model.parameters():
                p.requires_grad = True

        elif scope == 'head':
            # -------- FNO 风格 head --------
            fno_keys = ['fc0', 'fc1', 'fc2']
            has_fno_head = any(any(k in name for k in fno_keys) for name, _ in model.named_parameters())

            # -------- UNet 风格 head --------
            # 你的 UNet1 结构里最后几层是 up4 和 outc
            unet_keys = ['up4', 'outc']
            has_unet_head = any(any(k in name for k in unet_keys) for name, _ in model.named_parameters())

            if has_fno_head:
                for name, p in model.named_parameters():
                    if any(k in name for k in fno_keys):
                        p.requires_grad = True

            elif has_unet_head:
                for name, p in model.named_parameters():
                    if any(k in name for k in unet_keys):
                        p.requires_grad = True

            else:
                raise RuntimeError(
                    "TTA_SCOPE='head' but no recognized head layers were found. "
                    "Please check model parameter names or use scope='full'."
                )

        else:
            raise ValueError(f"Unsupported TTA scope: {scope}")

    trainable = [p for p in model.parameters() if p.requires_grad]

    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found for current TTA scope.")

    print(f"[DEBUG] TTA scope = {scope}, trainable params = {sum(p.numel() for p in trainable)}")
    return trainable

def direct_predict(model, base_state_dict, input_speckle_native):
    model.load_state_dict(base_state_dict)
    model.eval()

    with torch.no_grad():
        input_low = prepare_model_input(input_speckle_native)
        pred_obj = model(input_low)
        pred_obj = ensure_obj_size(pred_obj)

    return pred_obj.detach()

def refine_object(model, base_state_dict, tm, input_speckle_native,
                  num_steps=None, step_size=None):
    """
    不更新模型参数，只更新当前样本的 object 变量 x
    """
    if num_steps is None:
        num_steps = getattr(cfg, 'REFINE_STEPS', 3)
    if step_size is None:
        step_size = getattr(cfg, 'REFINE_LR', 0.1)

    model.load_state_dict(base_state_dict)
    model.eval()

    input_low = prepare_model_input(input_speckle_native)

    with torch.no_grad():
        x0 = model(input_low)
        x0 = ensure_obj_size(x0)

    x = x0.detach().clone()

    for _ in range(num_steps):
        x.requires_grad_(True)

        loss = object_refine_loss(
            tm=tm,
            pred_obj=x,
            target_native=input_speckle_native,
            pcc_weight=getattr(cfg, 'REFINE_PCC_WEIGHT', 0.2),
            use_alpha=getattr(cfg, 'REFINE_USE_ALPHA', True),
        )

        grad = torch.autograd.grad(loss, x, create_graph=False)[0]

        with torch.no_grad():
            x = x - step_size * grad
            x = torch.clamp(x, 0.0, 1.0)

    return x.detach(), num_steps


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


# def tta_process(model, base_state_dict, tm, input_speckle_native, max_steps=100, tol=5e-2, patience=3):
#     """
#     尽量保持你原始 GitHub 版 TTA 行为。
#     """
#     model.load_state_dict(base_state_dict)
#
#     for param in model.parameters():
#         param.requires_grad = True
#
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.L1Loss()
#
#     target_native = input_speckle_native
#     input_low = prepare_model_input(input_speckle_native)
#
#     prev_loss = float('inf')
#     plateau_count = 0
#     actual_steps = 0
#
#     for step in range(max_steps):
#         pred_obj = model(input_low)
#         pred_obj = ensure_obj_size(pred_obj)
#
#         pred_speckle_sim = physics_forward(tm, pred_obj)
#
#         numerator = torch.sum(pred_speckle_sim * target_native)
#         denominator = torch.sum(pred_speckle_sim ** 2)
#         alpha = numerator / (denominator + 1e-8)
#         alpha = torch.clamp(alpha, min=1e-3, max=1e3)
#         scaled_sim = pred_speckle_sim * alpha.detach()
#
#         loss = criterion(scaled_sim, target_native)
#         current_loss = loss.item()
#
#         if step > 0:
#             rel_change = abs(prev_loss - current_loss) / (prev_loss + 1e-8)
#             if rel_change < tol:
#                 plateau_count += 1
#             else:
#                 plateau_count = 0
#             if plateau_count >= patience:
#                 break
#
#         prev_loss = current_loss
#         actual_steps = step + 1
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     return pred_obj.detach(), actual_steps

def tta_process(model, base_state_dict, tm, input_speckle_native,
                max_steps=None, tol=None, patience=None):
    if max_steps is None:
        max_steps = getattr(cfg, 'TTA_MAX_STEPS', 20)
    if tol is None:
        tol = getattr(cfg, 'TTA_REL_TOL', 5e-3)
    if patience is None:
        patience = getattr(cfg, 'TTA_PATIENCE', 3)

    model.load_state_dict(base_state_dict)
    model.train()

    trainable_params = configure_tta_scope(model, getattr(cfg, 'TTA_SCOPE', 'head'))
    optimizer = optim.Adam(trainable_params, lr=getattr(cfg, 'TTA_LR', 5e-4))

    criterion = nn.L1Loss()
    target_native = input_speckle_native
    input_low = prepare_model_input(input_speckle_native)

    # 保存 direct 初值，作为 trust region 中心
    with torch.no_grad():
        x0 = model(input_low).detach()
        x0 = ensure_obj_size(x0)

    prev_loss = float('inf')
    plateau_count = 0
    actual_steps = 0
    best_pred = x0.clone()
    best_loss = float('inf')

    for step in range(max_steps):
        pred_obj = model(input_low)
        pred_obj = ensure_obj_size(pred_obj)

        pred_speckle_sim = physics_forward(tm, pred_obj)

        numerator = torch.sum(pred_speckle_sim * target_native)
        denominator = torch.sum(pred_speckle_sim ** 2)
        alpha = numerator / (denominator + 1e-8)
        alpha = torch.clamp(alpha, min=1e-3, max=1e3)

        scaled_sim = pred_speckle_sim * alpha.detach()

        loss_data = criterion(scaled_sim, target_native)
        loss_prior = F.l1_loss(pred_obj, x0)
        loss = loss_data + getattr(cfg, 'TTA_TRUST_WEIGHT', 0.05) * loss_prior

        current_loss = loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_pred = pred_obj.detach().clone()

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

    model.eval()
    return best_pred.detach(), actual_steps

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
        model = UNet1(n_channels=1, n_classes=1).to(cfg.DEVICE)
    elif model_type == 'fno':
        model = Turbo_LightFNO1().to(cfg.DEVICE)
    else:
        raise ValueError(f'Unsupported MODEL_TYPE: {model_type}')
    return model


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

def run_refine_mode(model, base_state_dict, tm, ts_in):
    # 先算 direct 初值对应的 r0，便于和其他模式统一日志
    pred0 = direct_predict(model, base_state_dict, ts_in)
    pred0_sim = physics_forward(tm, pred0)

    numerator = torch.sum(pred0_sim * ts_in)
    denominator = torch.sum(pred0_sim ** 2)
    alpha = numerator / (denominator + 1e-8)
    alpha = torch.clamp(alpha, min=1e-3, max=1e3)
    r0 = F.l1_loss(pred0_sim * alpha, ts_in).item()

    pred_obj, acc_step = refine_object(
        model=model,
        base_state_dict=base_state_dict,
        tm=tm,
        input_speckle_native=ts_in,
        num_steps=getattr(cfg, 'REFINE_STEPS', 3),
        step_size=getattr(cfg, 'REFINE_LR', 0.1),
    )

    return pred_obj, acc_step, r0, 'refine'

def run_one_sample(eval_mode, model, base_state_dict, tm, ts_in):
    eval_mode = eval_mode.lower()

    if eval_mode == 'direct':
        return run_direct_mode(model, base_state_dict, tm, ts_in)
    elif eval_mode == 'tta':
        return run_all_tta_mode(model, base_state_dict, tm, ts_in)
    elif eval_mode == 'gated':
        return run_gated_mode(model, base_state_dict, tm, ts_in)
    elif eval_mode == 'refine':
        return run_refine_mode(model, base_state_dict, tm, ts_in)
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

    # weights_path = os.path.join(cfg.weight_path, f'pretrained_{model_type}.pth')
    # if not os.path.exists(weights_path):
    #     print(f'Error: 权重文件未找到: {weights_path}')
    #     return

    best_model_path = os.path.join(cfg.weight_path, getattr(cfg, 'BEST_MODEL_NAME', 'best_pretrained_unet.pth'))
    fallback_model_path = os.path.join(cfg.RESULT_DIR, f'pretrained_{getattr(cfg, "MODEL_TYPE").lower()}.pth')

    if os.path.exists(best_model_path):
        model_path = best_model_path
    else:
        model_path = fallback_model_path

    print(f"Model loaded from: {model_path}")
    base_state_dict = torch.load(model_path, map_location=cfg.DEVICE)

    # state_dict = torch.load(weights_path, map_location=cfg.DEVICE)
    # model.load_state_dict(state_dict)
    # base_state_dict = copy.deepcopy(state_dict)
    # print(f'Model loaded from: {weights_path}')

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

    save_vis_dir = os.path.join(cfg.RESULT_DIR, f'vis_{model_type}\\3')
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
