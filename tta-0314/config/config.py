# config.py
import torch

class Config:
    TM_PATH = r'F:\cwd\ctm_test\origin_gama=36\Tm_filt_eff=1.mat'
    # TM_PATH = r'H:\test\Tm_filt_eff=1.mat'
    TEST_DATA_PATH = r'F:\cwd\tta_mmf\data\test\gama=1'
    RESULT_DIR = r'F:\cwd\tta_mmf\result\0318\gama=1-200'
    # SYN_DATA_PATH = r'F:\cwd\tta_mmf\data\custom_train_data-1-resize1.pt'
    SYN_DATA_PATH = r'F:\cwd\tta_mmf\data\custom_train_data-1-resize1.pt'
    weight_path = r'F:\cwd\tta_mmf\weight\26-03\26-03-18\gama=1-200'
    # ---------- 模型 ----------
    MODEL_TYPE = 'fno'  # 'fno' or 'unet'

    # ---------- 物理参数 ----------
    PHASE_RANGE = 1.0
    IMG_SIZE = 64
    SPECKLE_SIZE = 64
    # ---------- Object refinement (inference) ----------
    REFINE_STEPS = 50  # 先和训练保持一致，可改成 10 做对比
    REFINE_LR = 0.1
    REFINE_PCC_WEIGHT = 0.2
    REFINE_USE_ALPHA = True  # 是否对预测 speckle 做 alpha 对齐
    # 是否检查 TM 形状
    ASSERT_TM_SHAPE = True

    # ---------- 训练参数 ----------
    PRETRAIN_EPOCHS = 300
    BATCH_SIZE = 16
    LR_PRETRAIN = 1e-3

    # ---------- FNO 参数 ----------
    MODES = 32
    WIDTH = 64

    # =========================
    # Training / validation
    # =========================
    SEED = 42
    VAL_RATIO = 0.1
    SAVE_BEST_ONLY = True
    BEST_MODEL_NAME = 'best_pretrained_fno.pth'
    SIMULATE_CAMERA_IO = True  # 训练时是否模拟 8-bit 图像量化

    # =========================
    # TTA
    # =========================
    TTA_SCOPE = 'full'  # 'head' or 'full'
    TTA_MAX_STEPS = 20
    TTA_LR = 5e-4
    # TTA_REL_TOL = 5e-3
    TTA_REL_TOL = 5e-3
    TTA_PATIENCE = 999
    TTA_TRUST_WEIGHT = 0.05  # 约束别偏离 direct 初值太远
    # ---------- TTA 参数 ----------

    TTA_GATE_ENABLE = False
    TTA_GATE_TAU_SKIP = 0.03
    TTA_GATE_TAU_LIGHT = 0.08
    TTA_GATE_LIGHT_STEPS = 50
    TTA_GATE_FULL_STEPS = 100
    TTA_STEPS = 20
    LR_TTA = 5e-3
    TTA_MIN_STEPS = 3
    # TTA_PATIENCE = 3
    # TTA_REL_TOL = 5e-3
    TTA_ABS_TOL = 1e-4
    # TTA_SCOPE = 'head'  # 'all' / 'head'

    EVAL_MODE = ('refine')  # 'direct' / 'tta' / 'gated'

    # all-TTA 模式
    EVAL_ALL_TTA_STEPS = 100

    # ---------- 其他 ----------
    NORMALIZE_MODE = 'max'
    SAVE_TOPK_VIS = 5

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()
