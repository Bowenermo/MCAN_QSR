# --------------------------------------------------------
# Config for MCAN + LVPruning (standalone package)
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.BBOXFEAT_EMB_SIZE = 2048
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.USE_AUX_FEAT = False
        self.USE_BBOX_FEAT = False
        self.BBOX_NORMALIZE = True

        # --- LVPruning-style gates (one optional gate per decoder layer after lang cross-attn) ---
        # Length must equal LAYER. Use 1.0 to disable pruning at that layer (no extra params).
        self.PRUNE_LAYER_RATIOS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.PRUNE_LAMBDA = 0.05
        self.PRUNE_GUMBEL_TAU = 1.0

        # Load backbone weights from a standard MCAN checkpoint (strict=False; new gate keys stay init).
        self.PRUNE_INIT_CKPT = ''

        # If True, only parameters whose name contains TRAIN_NAME_SUBSTR are trained (default: lv_gate).
        self.FREEZE_BACKBONE = False
        self.PRUNE_TRAIN_NAME_SUBSTR = 'lv_gate'
