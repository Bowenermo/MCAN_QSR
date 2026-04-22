from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        # MCAN language-fusion backbone (kept identical to mcan_small)
        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024

        # Raw image input
        self.USE_RAW_IMAGE_INPUT = True
        self.RAW_IMAGE_INPUT_SIZE = 224
        self.RAW_IMAGE_MEAN = [0.485, 0.456, 0.406]
        self.RAW_IMAGE_STD = [0.229, 0.224, 0.225]

        # ConvMixer backbone
        self.CMX_MODEL_NAME = 'convmixer_768_32'
        self.CMX_PRETRAINED = True
        self.CMX_GRAD_CHECKPOINT = True

        # GridPPM-Lite tokenizer
        self.CMX_FUSE_POOL_SIZE = 14
        self.CMX_POS_DIM = 5
        self.CMX_POS_HIDDEN = 256
        self.CMX_USE_BN = True

        # Fast training controls
        self.CMX_FREEZE_BACKBONE_EPOCHS = 2
        self.VISION_LR_SCALE = 0.25

        self.USE_BBOX_FEAT = False
        self.BBOX_NORMALIZE = False
