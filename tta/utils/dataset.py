import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上溯到根目录
# 将根目录添加到Python路径
sys.path.append(project_root)
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from utils.physics import load_tm, physics_forward
from config import cfg
# ================= 配置区域 =================

# 1. 源图片路径
# SOURCE_IMG_DIR = r'F:\cwd\tta_mmf\data\label_mytest'
SOURCE_IMG_DIR = r'F:\cwd\tta_mmf\data\shengchenginput'
# 2. 输出总目录
OUTPUT_ROOT = r'F:\cwd\tta_mmf\data'

# --- 功能开关 ---
# [选项 A] 是否生成用于训练的 .pt 文件？ (训练时必须选 True)
SAVE_AS_PT = False
PT_FILENAME = 'custom_train_data-4.pt'

# [选项 B] 是否将生成的散斑保存为单独的图片？ (用于人工检查)
SAVE_AS_IMAGES = True
IMAGE_SAVE_FOLDER = 'speckle_check'  # 图片将保存在 data/speckle_check 下
MAX_IMAGE_COUNT = 50  # 保存多少张？ (-1 代表保存全部，建议先存几十张看看)


# ===========================================

def save_single_speckle(spk_tensor, save_dir, filename):
    """
    只保存单张散斑图片 (自动归一化)
    """
    # 1. 转为 Numpy [H, W]
    spk_np = spk_tensor.squeeze().cpu().numpy()

    # 2. 归一化到 0-255 (Min-Max)
    # 散斑通常很暗或动态范围大，必须拉伸才能看清结构
    min_val = spk_np.min()
    max_val = spk_np.max()

    if max_val > min_val:
        spk_norm = (spk_np - min_val) / (max_val - min_val)
    else:
        spk_norm = spk_np  # 全黑或全白

    spk_uint8 = (spk_norm * 255).astype(np.uint8)

    # 3. 保存为 PNG
    # 使用 PIL 保存，不带坐标轴，纯像素数据
    img = Image.fromarray(spk_uint8)

    # 确保文件名以 .png 结尾
    if not filename.lower().endswith('.png'):
        filename += '.png'

    img.save(os.path.join(save_dir, filename))


def main():
    print("=== 自定义散斑生成器 (独立图片版) ===")

    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"错误: 源图片路径不存在 -> {SOURCE_IMG_DIR}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 准备图片保存目录
    if SAVE_AS_IMAGES:
        img_out_dir = os.path.join(OUTPUT_ROOT, IMAGE_SAVE_FOLDER)
        os.makedirs(img_out_dir, exist_ok=True)
        print(f"散斑图片将保存至: {img_out_dir}")

    # 1. 准备物理环境
    tm = load_tm()

    # 2. 预处理
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor()
    ])

    # 3. 搜索图片
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(SOURCE_IMG_DIR, ext)))

    print(f"找到 {len(img_paths)} 张源图片")
    if len(img_paths) == 0: return

    # 4. 批量处理
    batch_size = 32
    all_objects = []
    all_speckles = []

    batch_imgs = []
    batch_names = []
    saved_img_count = 0

    print("正在计算物理传播...")
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(img_paths)):
            try:
                # 加载图片
                img = Image.open(img_path)
                img_tensor = preprocess(img)

                batch_imgs.append(img_tensor)
                # 获取原始文件名 (不带扩展名)，用于给散斑命名
                name_base = os.path.splitext(os.path.basename(img_path))[0]
                batch_names.append(name_base)

                # 处理 Batch
                if len(batch_imgs) == batch_size or i == len(img_paths) - 1:
                    batch_stack = torch.stack(batch_imgs).to(cfg.DEVICE)

                    # === 物理传播 (TM) ===
                    speckles = physics_forward(tm, batch_stack)

                    # 收集数据 (如果需要生成 PT)
                    if SAVE_AS_PT:
                        all_objects.append(batch_stack.cpu())
                        all_speckles.append(speckles.cpu())

                    # === 保存单独的散斑图片 ===
                    if SAVE_AS_IMAGES:
                        for j in range(len(batch_imgs)):
                            if MAX_IMAGE_COUNT == -1 or saved_img_count < MAX_IMAGE_COUNT:
                                # 命名格式: speckle_原文件名.png
                                save_name = f"speckle_{batch_names[j]}.png"
                                save_single_speckle(speckles[j], img_out_dir, save_name)
                                saved_img_count += 1

                    # 清空
                    batch_imgs = []
                    batch_names = []

            except Exception as e:
                print(f"跳过: {img_path} ({e})")

    # 5. 结果汇总
    print("-" * 30)

    if SAVE_AS_IMAGES:
        print(f"已保存 {saved_img_count} 张散斑图片到: {img_out_dir}")
        print("请打开文件夹检查散斑纹理是否正常（应为颗粒状）。")

    if SAVE_AS_PT:
        if len(all_objects) > 0:
            final_objects = torch.cat(all_objects, dim=0)
            final_speckles = torch.cat(all_speckles, dim=0)

            save_path = os.path.join(OUTPUT_ROOT, PT_FILENAME)
            torch.save({
                'object': final_objects,
                'speckle': final_speckles,
                'img_size': cfg.IMG_SIZE,
                'speckle_size': cfg.SPECKLE_SIZE,
                'tm_path': cfg.TM_PATH,
            }, save_path)

            print(f"训练数据集 (.pt) 已保存到: {save_path}")
            print(f"数据形状: {final_objects.shape}")
        else:
            print("未生成有效数据，跳过保存 .pt")


if __name__ == '__main__':
    main()