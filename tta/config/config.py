# config.py
import torch

class Config:
    TM_PATH = r'F:\cwd\ctm_test\origin_gama=36\Tm_filt_eff=4.mat'
    TEST_DATA_PATH = r'F:\cwd\tta_mmf\data\test\gama=4'
    RESULT_DIR = r'F:\cwd\tta_mmf\result\0306\gama=4'
    SYN_DATA_PATH = r'F:\cwd\tta_mmf\data\custom_train_data-4.pt'
    weight_path = r'F:\cwd\tta_mmf\weight\26-03\26-03-06\gama=4'
    # ---------- 模型 ----------
    MODEL_TYPE = 'fno'  # 'fno' or 'unet'

    # ---------- 物理参数 ----------
    PHASE_RANGE = 1.0
    IMG_SIZE = 64
    SPECKLE_SIZE = 128

    # 是否检查 TM 形状
    ASSERT_TM_SHAPE = True

    # ---------- 训练参数 ----------
    PRETRAIN_EPOCHS = 300
    BATCH_SIZE = 16
    LR_PRETRAIN = 1e-3

    # ---------- FNO 参数 ----------
    MODES = 32
    WIDTH = 64

    # ---------- TTA 参数 ----------
    TTA_STEPS = 20
    LR_TTA = 5e-3
    TTA_MIN_STEPS = 3
    TTA_PATIENCE = 3
    TTA_REL_TOL = 5e-3
    TTA_ABS_TOL = 1e-4
    TTA_SCOPE = 'head'  # 'all' / 'head'

    # ---------- 其他 ----------
    NORMALIZE_MODE = 'max'
    SAVE_TOPK_VIS = 5

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()