# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 标准初始化
        # self.scale = (1 / (in_channels * out_channels))
        self.scale = (1 / in_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)

        modes1 = min(self.modes1, x_ft.size(2))
        modes2 = min(self.modes2, x_ft.size(3))

        out_ft[:, :, :modes1, :modes2] = \
            self.compl_mul2d(x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2])
        out_ft[:, :, -modes1:, :modes2] = \
            self.compl_mul2d(x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2])

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class Turbo_LightFNO1(nn.Module):
    def __init__(self):
        super(Turbo_LightFNO1, self).__init__()
        self.modes1 = cfg.MODES
        self.modes2 = cfg.MODES
        self.width = cfg.WIDTH
        self.target_size = (cfg.IMG_SIZE, cfg.IMG_SIZE)  # 输出 64x64

        # Lifting Layer (FC0) - TTA时需要更新
        self.fc0 = nn.Linear(1, self.width)

        # Fourier Layers - TTA时冻结
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # 1x1 Convs (Skip connections)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Projection Layers (FC1, FC2) - TTA时需要更新
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)  # Output 1 channel

    def forward(self, x):
        # x: [B, 1, 384, 384]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)

        # 插值到 64x64
        x = F.interpolate(x, size=self.target_size, mode='bicubic', align_corners=False)
        x = torch.sigmoid(x)
        return x

def set_tta_mode(self, scope='head'):
    for p in self.parameters():
        p.requires_grad = False

    if scope in ['head', 'all']:
        for p in self.fc0.parameters():
            p.requires_grad = True
        for p in self.fc1.parameters():
            p.requires_grad = True
        for p in self.fc2.parameters():
            p.requires_grad = True

    if scope == 'all':
        for m in [self.conv0, self.conv1, self.conv2, self.conv3, self.w0, self.w1, self.w2, self.w3]:
            for p in m.parameters():
                p.requires_grad = True