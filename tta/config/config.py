# config.py
import torch


class Config:
    # --- 路径设置 ---
    # 你的 TM 路径
    TM_PATH = r'F:\cwd\ctm_test\origin_gama=36\Tm_filt_eff=4.mat'
    # 真实测试数据路径
    TEST_DATA_PATH = r'F:\cwd\tta_mmf\data\test\gama=4'
    # 结果保存路径
    RESULT_DIR = r'F:\cwd\tta_mmf\result\0306\gama=4'
    # 生成的虚拟数据集保存路径
    SYN_DATA_PATH = r'F:\cwd\tta_mmf\data\custom_train_data-4.pt'
    weight_path = r'F:\cwd\tta_mmf\weight\26-03\gama=4'
    MODEL_TYPE = 'fno'
    # --- 物理参数 ---
    PHASE_RANGE = 1.0  # 用户指定: 0 ~ 1.0 * pi
    IMG_SIZE = 64  # 物体尺寸 (对应 TM 输入)
    SPECKLE_SIZE = 128  # 散斑尺寸 (对应 TM 输出)

    # --- 训练参数 (预训练) ---
    PRETRAIN_EPOCHS = 300  # 预训练轮数 (建议多跑一点，让 Base Model 足够强)
    BATCH_SIZE = 16  # 批次大小
    LR_PRETRAIN = 1e-3  # 预训练学习率
    MODES = 32  # FNO 频率分量
    WIDTH = 64  # FNO 通道宽

    # --- TTA 参数 (测试时微调) ---
    TTA_STEPS = 20  # 单张图微调步数 (很少，所以快)
    LR_TTA = 5e-3  # 微调学习率 (稍微大点，加速收敛)

    # --- 设备 ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 实例化配置
cfg = Config()