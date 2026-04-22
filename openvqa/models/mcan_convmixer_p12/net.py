# --------------------------------------------------------
# OpenVQA
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan.mca import MCA_ED

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        neg_inf = torch.finfo(att.dtype).min
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            neg_inf
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i:i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


class ConvMixerVisionEncoder(nn.Module):
    def __init__(self, __C):
        super(ConvMixerVisionEncoder, self).__init__()
        self.__C = __C

        try:
            import timm
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "timm is required by mcan_convmixer_p12. Install with `pip install timm`."
            ) from exc

        self.backbone = timm.create_model(
            __C.CMX_MODEL_NAME,
            pretrained=__C.CMX_PRETRAINED,
        )
        if bool(getattr(__C, 'CMX_GRAD_CHECKPOINT', False)) and hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)

        in_channels = self.backbone.num_features
        self.proj = nn.Conv2d(in_channels, __C.HIDDEN_SIZE, kernel_size=1)
        self.pos_mlp = nn.Sequential(
            nn.Linear(__C.CMX_POS_DIM, __C.CMX_POS_HIDDEN),
            nn.GELU(),
            nn.Linear(__C.CMX_POS_HIDDEN, __C.HIDDEN_SIZE),
        )

    def _build_pos_feat(self, bsz, h, w, device, dtype):
        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        x1 = xx / float(w)
        y1 = yy / float(h)
        x2 = (xx + 1.0) / float(w)
        y2 = (yy + 1.0) / float(h)
        area = torch.full_like(x1, 1.0 / float(h * w))

        pos = torch.stack([x1, y1, x2, y2, area], dim=-1).view(1, h * w, 5)
        pos = pos.repeat(bsz, 1, 1)
        return pos

    def forward(self, image):
        feat = self.backbone.forward_features(image)  # [B, C, H, W]
        feat = self.proj(feat)  # [B, HIDDEN, H, W]

        pool_size = int(getattr(self.__C, 'CMX_FUSE_POOL_SIZE', 0))
        if pool_size > 0:
            feat = F.adaptive_avg_pool2d(feat, (pool_size, pool_size))

        bsz, _, h, w = feat.size()
        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, N, HIDDEN]

        pos = self._build_pos_feat(bsz, h, w, tokens.device, tokens.dtype)
        tokens = tokens + self.pos_mlp(pos)

        img_mask = torch.zeros((bsz, 1, 1, h * w), dtype=torch.bool, device=tokens.device)
        return tokens, img_mask


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(token_size, __C.WORD_EMBED_SIZE)
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.visual_encoder = ConvMixerVisionEncoder(__C)
        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def get_optim_groups(self, __C):
        vision_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('visual_encoder.backbone'):
                vision_params.append(param)
            else:
                other_params.append(param)

        return [
            {'params': other_params, 'lr_scale': 1.0},
            {'params': vision_params, 'lr_scale': __C.VISION_LR_SCALE},
        ]

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        image = frcn_feat  # raw image tensor from dataset

        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.visual_encoder(image)

        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(lang_feat, lang_feat_mask)
        img_feat = self.attflat_img(img_feat, img_feat_mask)

        proj_feat = self.proj_norm(lang_feat + img_feat)
        proj_feat = self.proj(proj_feat)
        return proj_feat
