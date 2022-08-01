""" Spatial and Temporal feature Mixer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from models.layers.mlp import Mlp


class STAttention(nn.Module):
    def __init__(self, dim, qk_bias=False, proj_drop=0.):
        super().__init__()
        self.qk = nn.Linear(dim, dim, bias=qk_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        xqk = self.qk(x)
        xlq, xgk, xtk = xqk[:,0], xqk[:,1], xqk[:,2]

        xlq = nn.functional.normalize(xlq, dim=1)
        xgk = nn.functional.normalize(xgk, dim=1)
        xtk = nn.functional.normalize(xtk, dim=1)
        
        att_s = torch.einsum('nc,nc->n', [xlq, xgk])
        att_t = torch.einsum('nc,nc->n', [xlq, xtk])

        x_att = x[:,1]*att_s[:,None] + x[:,2]*att_t[:,None]
        x_att = self.proj(x_att)
        x_att = self.proj_drop(x_att)
        return x_att


class STMixer(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qk_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = STAttention(dim, qk_bias=qk_bias, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:,0] + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
