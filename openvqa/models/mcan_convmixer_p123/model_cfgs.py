# --------------------------------------------------------
# OpenVQA
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        # MCAN language-fusion backbone
        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024

        # End-to-end visual branch from raw image
        self.USE_RAW_IMAGE_INPUT = True
        self.RAW_IMAGE_INPUT_SIZE = 224
        self.RAW_IMAGE_MEAN = [0.485, 0.456, 0.406]
        self.RAW_IMAGE_STD = [0.229, 0.224, 0.225]

        self.CMX_MODEL_NAME = 'convmixer_768_32'
        self.CMX_PRETRAINED = True
        self.CMX_GRAD_CHECKPOINT = True
        self.CMX_FUSE_POOL_SIZE = 16

        # Explicit 2D position embedding
        self.CMX_POS_DIM = 5
        self.CMX_POS_HIDDEN = 256

        # P3: question-guided visual token selection
        self.QG_TOPK = 64
        self.QG_HIDDEN = 512

        # Vision branch learns with smaller lr
        self.VISION_LR_SCALE = 0.1

        self.USE_BBOX_FEAT = False
        self.BBOX_NORMALIZE = False
