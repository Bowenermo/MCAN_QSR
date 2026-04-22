from openvqa.core.base_dataset import BaseAdapter
import torch


class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def vqa_init(self, __C):
        return

    def gqa_init(self, __C):
        return

    def clevr_init(self, __C):
        return

    def vqa_forward(self, feat_dict):
        # In raw-image mode, VQA loader reuses FRCN_FEAT slot for image tensor [B, 3, H, W].
        image = feat_dict['FRCN_FEAT']
        img_mask = torch.zeros((image.size(0), 1, 1, 1), dtype=torch.bool, device=image.device)
        return image, img_mask

    def gqa_forward(self, feat_dict):
        raise RuntimeError('mcan_qsr currently supports VQA raw-image input only.')

    def clevr_forward(self, feat_dict):
        raise RuntimeError('mcan_qsr currently supports VQA raw-image input only.')
