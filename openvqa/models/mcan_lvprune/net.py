# --------------------------------------------------------
# MCAN + LVPruning — separate Net class (does not modify mcan/net.py)
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan_lvprune.mca import MCA_ED_LV
from openvqa.models.mcan.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        mask_fill_value = torch.finfo(att.dtype).min
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            mask_fill_value
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


def _maybe_freeze_backbone(net, __C):
    if not getattr(__C, 'FREEZE_BACKBONE', False):
        return
    substr = getattr(__C, 'PRUNE_TRAIN_NAME_SUBSTR', 'lv_gate')
    for name, p in net.named_parameters():
        if substr in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)
        self.backbone = MCA_ED_LV(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        _maybe_freeze_backbone(self, __C)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)

        lang_feat, img_feat, ratio_loss = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_vec = self.attflat_lang(lang_feat, lang_feat_mask)
        img_vec = self.attflat_img(img_feat, img_feat_mask)

        proj_feat = lang_vec + img_vec
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        pl = float(getattr(self.__C, 'PRUNE_LAMBDA', 0.0))
        if pl > 0.0 and self.training:
            return proj_feat, pl * ratio_loss
        return proj_feat
