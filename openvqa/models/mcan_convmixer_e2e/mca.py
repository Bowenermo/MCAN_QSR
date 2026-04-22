# --------------------------------------------------------
# OpenVQA
# --------------------------------------------------------

from openvqa.models.mcan.mca import MHAtt, FFN, SA
from openvqa.ops.layer_norm import LayerNorm

import torch
import torch.nn as nn


class ConvMixerTokenMixer(nn.Module):
    def __init__(self, hidden_size, kernel_size):
        super(ConvMixerTokenMixer, self).__init__()
        self.dwconv = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
        )
        self.dw_act = nn.GELU()
        self.dw_bn = nn.BatchNorm2d(hidden_size)

        self.pwconv = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.pw_act = nn.GELU()
        self.pw_bn = nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = x + self.dw_bn(self.dw_act(self.dwconv(x)))
        x = self.pw_bn(self.pw_act(self.pwconv(x)))
        return x


class CMXGuidedBlock(nn.Module):
    def __init__(self, __C):
        super(CMXGuidedBlock, self).__init__()
        self.__C = __C

        self.token_mixer = ConvMixerTokenMixer(__C.HIDDEN_SIZE, __C.CMX_TOKEN_MIXER_K)
        self.norm_img = LayerNorm(__C.HIDDEN_SIZE)

        self.mhatt_cross = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, y_mask, img_hw):
        # x: [B, N_img, H], y: [B, N_lang, H]
        bsz = x.size(0)
        h, w = img_hw

        x_img = x.transpose(1, 2).contiguous().view(bsz, self.__C.HIDDEN_SIZE, h, w)
        x_img = self.token_mixer(x_img)
        x = x_img.flatten(2).transpose(1, 2).contiguous()
        x = self.norm_img(x)

        x = self.norm1(x + self.dropout1(
            self.mhatt_cross(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


class MCA_CMX_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_CMX_ED, self).__init__()
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([CMXGuidedBlock(__C) for _ in range(__C.LAYER)])

    def forward(self, y, x, y_mask, img_hw):
        for enc in self.enc_list:
            y = enc(y, y_mask)

        for dec in self.dec_list:
            x = dec(x, y, y_mask, img_hw)

        return y, x
