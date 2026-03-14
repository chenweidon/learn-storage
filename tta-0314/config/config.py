# config.py
import torch

class Config:
    TM_PATH = r'F:\cwd\ctm_test\origin_gama=36\Tm_filt_eff=1.mat'
    # TM_PATH = r'H:\test\Tm_filt_eff=1.mat'
    TEST_DATA_PATH = r'F:\cwd\tta_mmf\data\test\gama=1'
    RESULT_DIR = r'F:\cwd\tta_mmf\result\0313\gama=1'
    SYN_DATA_PATH = r'F:\cwd\tta_mmf\data\custom_train_data-1-resize1.pt'
    weight_path = r'F:\cwd\tta_mmf\weight\26-03\26-03-13\gama=1'
    # ---------- 模型 ----------
    MODEL_TYPE = 'fno'  # 'fno' or 'unet'

    # ---------- 物理参数 ----------
    PHASE_RANGE = 1.0
    IMG_SIZE = 64
    SPECKLE_SIZE = 64

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
    TTA_GATE_ENABLE = True
    TTA_GATE_TAU_SKIP = 0.03
    TTA_GATE_TAU_LIGHT = 0.08
    TTA_GATE_LIGHT_STEPS = 50
    TTA_GATE_FULL_STEPS = 100
    TTA_STEPS = 20
    LR_TTA = 5e-3
    TTA_MIN_STEPS = 3
    TTA_PATIENCE = 3
    TTA_REL_TOL = 5e-3
    TTA_ABS_TOL = 1e-4
    TTA_SCOPE = 'head'  # 'all' / 'head'

    EVAL_MODE = 'gated'  # 'direct' / 'tta' / 'gated'

    # all-TTA 模式
    EVAL_ALL_TTA_STEPS = 100

    # ---------- 其他 ----------
    NORMALIZE_MODE = 'max'
    SAVE_TOPK_VIS = 5

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()