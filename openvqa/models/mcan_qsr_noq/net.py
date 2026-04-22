from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan.mca import MCA_ED
from openvqa.models.mcan_qsr_noq.adapter import Adapter

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
        mask_fill_value = torch.finfo(att.dtype).min
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            mask_fill_value
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PPMLitePatchTokenizer(nn.Module):
    def __init__(self, __C):
        super(PPMLitePatchTokenizer, self).__init__()
        hidden = __C.HIDDEN_SIZE
        out_hw = int(__C.CMX_FUSE_POOL_SIZE)
        low_hw = max(1, out_hw // 2)

        self.pool14 = nn.AdaptiveAvgPool2d((out_hw, out_hw))
        self.pool7 = nn.AdaptiveAvgPool2d((low_hw, low_hw))
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False)
        self.dw_act = nn.GELU()
        self.fuse = nn.Conv2d(hidden * 2, hidden, kernel_size=1, bias=False)

        self.out_hw = out_hw

    def forward(self, feat):
        branch1 = self.pool14(feat)

        branch2 = self.pool7(feat)
        branch2 = self.dwconv(branch2)
        branch2 = self.dw_act(branch2)
        branch2 = F.interpolate(branch2, size=(self.out_hw, self.out_hw), mode='bilinear', align_corners=False)

        feat = torch.cat([branch1, branch2], dim=1)
        feat = self.fuse(feat)

        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        return tokens, self.out_hw, self.out_hw


class ConvMixerVisionEncoder(nn.Module):
    def __init__(self, __C):
        super(ConvMixerVisionEncoder, self).__init__()
        self.__C = __C
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "timm is required by mcan_qsr_noq. Install with `pip install timm`."
            ) from exc

        self.backbone = timm.create_model(
            __C.CMX_MODEL_NAME,
            pretrained=__C.CMX_PRETRAINED,
        )
        if bool(getattr(__C, 'CMX_GRAD_CHECKPOINT', False)) and hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)

        in_channels = self.backbone.num_features
        self.proj = nn.Conv2d(in_channels, __C.HIDDEN_SIZE, kernel_size=1, bias=False)
        if bool(getattr(__C, 'CMX_USE_BN', True)):
            self.proj_norm = nn.BatchNorm2d(__C.HIDDEN_SIZE)
        else:
            self.proj_norm = LayerNorm2d(__C.HIDDEN_SIZE)
        self.proj_act = nn.GELU()

        self.ppm_tokenizer = PPMLitePatchTokenizer(__C)

        self.freeze_backbone_epochs = int(getattr(__C, 'CMX_FREEZE_BACKBONE_EPOCHS', 0))
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _forward_backbone(self, image):
        if self.training and self.current_epoch < self.freeze_backbone_epochs:
            old_mode = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                feat = self.backbone.forward_features(image)
            if old_mode:
                self.backbone.train()
            return feat
        return self.backbone.forward_features(image)

    def _build_pos5d(self, bsz, h, w, device, dtype):
        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        x1 = xx / float(w)
        y1 = yy / float(h)
        x2 = (xx + 1.0) / float(w)
        y2 = (yy + 1.0) / float(h)
        area = torch.full_like(x1, 1.0 / float(h * w))

        pos = torch.stack([x1, y1, x2, y2, area], dim=-1).view(1, h * w, 5)
        return pos.repeat(bsz, 1, 1)

    def forward(self, image):
        feat = self._forward_backbone(image)
        feat = self.proj(feat)
        feat = self.proj_norm(feat)
        feat = self.proj_act(feat)

        tokens, h, w = self.ppm_tokenizer(feat)
        pos5d = self._build_pos5d(tokens.size(0), h, w, tokens.device, tokens.dtype)
        return tokens, pos5d


class VisionSoftRegionSelector(nn.Module):
    def __init__(self, __C):
        super(VisionSoftRegionSelector, self).__init__()
        hidden = int(getattr(__C, 'QSR_SELECTOR_HIDDEN', __C.HIDDEN_SIZE))
        num_regions = int(getattr(__C, 'QSR_NUM_REGIONS', 48))
        self.num_regions = num_regions

        self.v_proj = nn.Linear(__C.HIDDEN_SIZE, hidden)
        self.p_proj = nn.Linear(__C.CMX_POS_DIM, hidden)
        self.score_head = nn.Linear(hidden, num_regions)
        self.pos_mlp = nn.Sequential(
            nn.Linear(__C.CMX_POS_DIM, __C.CMX_POS_HIDDEN),
            nn.GELU(),
            nn.Linear(__C.CMX_POS_HIDDEN, __C.HIDDEN_SIZE),
        )

    def set_epoch(self, epoch):
        # Keep this hook for compatibility with train_engine.
        return

    def _diversity_loss(self, assign):
        # assign: [B, K, N]
        _, k_num, _ = assign.size()
        gram = torch.matmul(assign, assign.transpose(1, 2))  # [B, K, K]
        diag = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
        denom = float(max(k_num * max(k_num - 1, 1), 1))
        off_diag = (gram.sum(dim=(1, 2)) - diag) / denom
        return off_diag.mean()

    def forward(self, dense_tokens, patch_pos):
        # dense_tokens: [B, N, H], patch_pos: [B, N, 5]
        fused = F.gelu(self.v_proj(dense_tokens) + self.p_proj(patch_pos))
        score = self.score_head(fused)  # [B, N, K]

        # Softmax over patches to build assignment A: [B, K, N]
        assign = F.softmax(score.transpose(1, 2), dim=-1)

        region_tokens = torch.matmul(assign, dense_tokens)  # [B, K, H]
        region_pos = torch.matmul(assign, patch_pos)  # [B, K, 5]
        region_tokens = region_tokens + self.pos_mlp(region_pos)

        rescue = dense_tokens.mean(dim=1, keepdim=True)  # [B, 1, H]
        img_feat = torch.cat([region_tokens, rescue], dim=1)  # [B, K+1, H]
        img_mask = torch.zeros(
            (img_feat.size(0), 1, 1, img_feat.size(1)),
            dtype=torch.bool,
            device=img_feat.device
        )

        div_loss = self._diversity_loss(assign)
        return img_feat, img_mask, div_loss


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
        self.visual_encoder = ConvMixerVisionEncoder(__C)
        self.selector = VisionSoftRegionSelector(__C)
        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def on_train_epoch_start(self, epoch):
        self.visual_encoder.set_epoch(epoch)
        self.selector.set_epoch(epoch)

    def get_optim_groups(self, __C):
        vision_params = []
        other_params = []
        for name, param in self.named_parameters():
            if name.startswith('visual_encoder.backbone'):
                vision_params.append(param)
            else:
                other_params.append(param)

        return [
            {'params': other_params, 'lr_scale': 1.0},
            {'params': vision_params, 'lr_scale': __C.VISION_LR_SCALE},
        ]

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        image, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)
        dense_tokens, patch_pos = self.visual_encoder(image)
        img_feat, img_feat_mask, div_loss = self.selector(dense_tokens, patch_pos)

        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(lang_feat, lang_feat_mask)
        img_feat = self.attflat_img(img_feat, img_feat_mask)

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        if self.training:
            lambda_div = float(getattr(self.__C, 'QSR_LAMBDA_DIV', 1e-3))
            return proj_feat, lambda_div * div_loss
        return proj_feat
