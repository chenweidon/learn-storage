import sys
import os
# 获取当前文件的目录，然后向上找到根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上溯到根目录
# 将根目录添加到Python路径
sys.path.append(project_root)
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import cfg


def inspect():
    print(f"正在加载数据: {cfg.SYN_DATA_PATH} ...")
    try:
        data_dict = torch.load(cfg.SYN_DATA_PATH)
        objects = data_dict['object']
        speckles = data_dict['speckle']
    except FileNotFoundError:
        print("错误：找不到文件，请先运行 dataset_gen.py")
        return

    print(f"\n=== 数据统计信息 ===")
    print(f"数据量: {objects.shape[0]}")
    print(f"物体维度: {objects.shape} (应为 [N, 1, 64, 64])")
    print(f"散斑维度: {speckles.shape} (应为 [N, 1, 384, 384])")

    # 检查数值范围
    obj_min, obj_max = objects.min().item(), objects.max().item()
    sp_min, sp_max = speckles.min().item(), speckles.max().item()
    sp_mean = speckles.mean().item()

    print(f"物体数值范围: {obj_min:.4f} ~ {obj_max:.4f} (应在 0~1 之间)")
    print(f"散斑数值范围: {sp_min:.4e} ~ {sp_max:.4e}")
    print(f"散斑平均强度: {sp_mean:.4e}")

    # === 致命错误预警 ===
    if sp_max == 0:
        print("\n[!!!! 致命错误 !!!!] 散斑全是黑的！TM 计算结果为 0。")
        print("可能原因：")
        print("1. TM 矩阵全是 0？")
        print("2. 相位编码后全是 0？")
        return

    if sp_max == sp_min:
        print("\n[!!!! 致命错误 !!!!] 散斑是一张纯色灰度图，没有任何纹理。")
        return

    # === 可视化检查 ===
    print("\n正在绘制前 3 对样本...")

    # 随机挑 3 张或者前 3 张
    indices = [0, 10, 20] if objects.shape[0] > 20 else [0]

    plt.figure(figsize=(12, 8))

    for i, idx in enumerate(indices):
        # 物体
        plt.subplot(3, 2, 2 * i + 1)
        obj_img = objects[idx].squeeze().cpu().numpy()
        plt.imshow(obj_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Sample {idx}: Object Phase")
        plt.colorbar(fraction=0.046)
        plt.axis('off')

        # 散斑
        plt.subplot(3, 2, 2 * i + 2)
        sp_img = speckles[idx].squeeze().cpu().numpy()
        plt.imshow(sp_img, cmap='gray')  # 散斑不需要固定 vmin/vmax，自动归一化看纹理
        plt.title(f"Sample {idx}: Calculated Speckle")
        plt.colorbar(fraction=0.046)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    inspect()