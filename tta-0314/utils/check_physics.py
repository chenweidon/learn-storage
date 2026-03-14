# check_tm.py
import os
import sys
# 获取当前文件的目录，然后向上找到根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上溯到根目录
# 将根目录添加到Python路径
sys.path.append(project_root)
from physics import load_tm, physics_forward
import torch
import matplotlib.pyplot as plt

tm = load_tm()
# 造一个“中心亮点”的物体
obj = torch.zeros(1, 1, 64, 64).to('cuda')
obj[:, :, 32, 32] = 1.0

# 造一个“全白”物体
obj_white = torch.ones(1, 1, 64, 64).to('cuda')

spk_point = physics_forward(tm, obj)
spk_white = physics_forward(tm, obj_white)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(spk_point[0,0].cpu().numpy(), cmap='gray')
plt.title("Point Source Response")

plt.subplot(1, 2, 2)
plt.imshow(spk_white[0,0].cpu().numpy(), cmap='gray')
plt.title("Full Field Response")
plt.show()