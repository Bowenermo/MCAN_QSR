# --------------------------------------------------------
# OpenVQA
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        # Language / fusion backbone (MCAN-style)
        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024

        # Vision branch is fully end-to-end ConvMixer
        self.USE_RAW_IMAGE_INPUT = True
        self.RAW_IMAGE_INPUT_SIZE = 224
        self.RAW_IMAGE_MEAN = [0.485, 0.456, 0.406]
        self.RAW_IMAGE_STD = [0.229, 0.224, 0.225]

        # timm ConvMixer backbone
        self.CMX_MODEL_NAME = 'convmixer_768_32'
        self.CMX_PRETRAINED = True
        self.CMX_TOKEN_MIXER_K = 7
        self.CMX_GRAD_CHECKPOINT = True
        # Reduce image token count before multimodal fusion to save memory.
        # If <= 0, keep native HxW tokens from ConvMixer.
        self.CMX_FUSE_POOL_SIZE = 16

        # Optimization knobs for end-to-end V-L training
        self.VISION_LR_SCALE = 0.1
        self.USE_BBOX_FEAT = False
        self.BBOX_NORMALIZE = False
