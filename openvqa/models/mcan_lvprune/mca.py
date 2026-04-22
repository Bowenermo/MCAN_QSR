# --------------------------------------------------------
# MCA stack for MCAN + LVPruning (new file; does not modify mcan/mca.py)
# Reuses MHAtt / FFN / SA from original MCAN implementation.
# --------------------------------------------------------

from openvqa.models.mcan.mca import MHAtt, FFN, SA
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch


# -------------------------------
# Language-guided vision token gate (LVPruning-style)
# -------------------------------

class LVGate(nn.Module):
    """
    After language-conditioned cross-attention on image tokens, predict soft
    keep weights. Training: Gumbel-Softmax; eval: hard argmax keep/remove.
    Padding positions (from x_mask) are always kept and excluded from ratio loss.
    """
    def __init__(self, __C, layer_idx, target_ratio):
        super(LVGate, self).__init__()
        self.__C = __C
        self.layer_idx = layer_idx
        self.target_ratio = float(target_ratio)
        self.cross = MHAtt(__C)
        self.ffn_gate = nn.Sequential(
            LayerNorm(__C.HIDDEN_SIZE),
            nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(__C.DROPOUT_R),
            nn.Linear(__C.HIDDEN_SIZE * 2, __C.HIDDEN_SIZE),
        )
        self.logits = nn.Linear(__C.HIDDEN_SIZE, 2)
        self.tau = float(__C.PRUNE_GUMBEL_TAU)

    def forward(self, x, y, y_mask, x_mask):
        # Paper-style: vision query attends to language (see LVPruning Fig.2)
        o = self.cross(v=y, k=y, q=x, mask=y_mask)
        o = o + x
        o = self.ffn_gate(o)
        logit = self.logits(o)

        pad = x_mask.squeeze(1).squeeze(1)
        if self.training:
            g = F.gumbel_softmax(logit, tau=self.tau, hard=False, dim=-1)
            keep = g[:, :, 0]
        else:
            hard = torch.argmax(logit, dim=-1)
            keep = hard.eq(0).float()

        keep = keep.masked_fill(pad, 1.0)

        x_out = x * keep.unsqueeze(-1)

        valid = (~pad).float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        mean_keep = (keep * valid).sum(dim=1) / denom
        ratio_loss = ((mean_keep - self.target_ratio) ** 2).mean()

        return x_out, ratio_loss


class SGA_LV(nn.Module):
    """Self-Guided Attention block with optional LV gate after lang cross-attn."""
    def __init__(self, __C, layer_idx):
        super(SGA_LV, self).__init__()
        self.__C = __C
        self.layer_idx = layer_idx

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

        ratios = getattr(__C, 'PRUNE_LAYER_RATIOS', [1.0] * __C.LAYER)
        rho = float(ratios[layer_idx]) if layer_idx < len(ratios) else 1.0
        self.use_gate = rho < 0.999
        self.lv_gate = None
        if self.use_gate:
            self.lv_gate = LVGate(__C, layer_idx, target_ratio=rho)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        ratio_loss = x.new_zeros(())
        if self.lv_gate is not None:
            x, ratio_loss = self.lv_gate(x, y, y_mask, x_mask)

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x, ratio_loss


class MCA_ED_LV(nn.Module):
    def __init__(self, __C):
        super(MCA_ED_LV, self).__init__()
        self.__C = __C
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([
            SGA_LV(__C, li) for li in range(__C.LAYER)
        ])

    def forward(self, y, x, y_mask, x_mask):
        for enc in self.enc_list:
            y = enc(y, y_mask)

        ratio_acc = x.new_zeros(())
        for dec in self.dec_list:
            x, rl = dec(x, y, x_mask, y_mask)
            ratio_acc = ratio_acc + rl
        return y, x, ratio_acc
