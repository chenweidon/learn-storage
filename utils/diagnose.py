import os
import sys
# 获取当前文件的目录，然后向上找到根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上溯到根目录
# 将根目录添加到Python路径
sys.path.append(project_root)
import torch
import numpy as np
import matplotlib.pyplot as plt
from physics import load_tm, phase_encoding


def diagnose():
    print("=== 开始散斑维度诊断 ===")

    # 1. 加载 TM
    tm = load_tm()  # [Out, In]
    device = tm.device

    # 2. 创建一个中心点光源 (Point Source)
    # 点光源最能反映系统的脉冲响应 (PSF)
    img_size = 64
    obj = torch.zeros(1, 1, img_size, img_size).to(device)
    obj[:, :, img_size // 2, img_size // 2] = 1.0  # 中心点亮

    # 3. 编码并物理传播
    # 注意：这里我们暂时不管输入端的编码顺序，只看输出端的解码
    complex_vec = phase_encoding(obj)
    scattered_vec = torch.matmul(tm, complex_vec)  # [1, 147456, 1]

    # 转为 Numpy CPU 处理，避免 PyTorch view 的歧义
    # 取出复数场数据
    field_data = scattered_vec.detach().cpu().numpy().flatten()
    # 取强度
    intensity_data = np.abs(field_data) ** 2

    print(f"数据总长度: {len(intensity_data)}")
    print(f"目标尺寸: 384 x 384 = {384 * 384}")

    # === 4. 生成 4 种视角的对比图 ===

    # A. PyTorch 默认 (Row-Major / C-Order)
    # 对应: torch.view(384, 384)
    img_A = intensity_data.reshape(384, 384)

    # B. MATLAB 默认 (Column-Major / F-Order)
    # 对应: torch.view(384, 384).permute(0, 2, 1) 的效果
    img_B = intensity_data.reshape(384, 384, order='F')

    # C. 假设 A 的转置 (看看是不是简单的旋转)
    img_C = img_A.T

    # D. 假设 B 的翻转 (MATLAB 数据有时会有上下翻转)
    img_D = np.flipud(img_B)

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    titles = [
        "A: Default (Row-Major)\nYour current output",
        "B: MATLAB Style (Col-Major)\nMost Likely Correct",
        "C: Transpose of A",
        "D: Flipped B"
    ]

    images = [img_A, img_B, img_C, img_D]

    for ax, img, title in zip(axes.flatten(), images, titles):
        # 局部放大看细节 (取中心 100x100)
        h, w = img.shape
        center_h, center_w = h // 2, w // 2
        crop = img[center_h - 50:center_h + 50, center_w - 50:center_w + 50]

        ax.imshow(crop, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('diagnosis_result.png', dpi=150)
    plt.show()
    print("诊断图已保存为 diagnosis_result.png")


if __name__ == '__main__':
    diagnose()