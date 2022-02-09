# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

# from timm.models.vision_transformer import VisionTransformer
from timm.models.layers.helpers import to_3tuple
from timm.models.layers import PatchEmbed
from models.layers.vision_transformer import VisionTransformer


class VisionTransformerND(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, is_3d=False, use_learnable_pos_emb=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed sin-cos position embedding
        if not use_learnable_pos_emb:
            if is_3d:
                self.build_3d_sincos_position_embedding()
            else:
                self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_3d_sincos_position_embedding(self, temperature=10000.):
        d, h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_d = torch.arange(d, dtype=torch.float32)
        grid_w, grid_h, grid_d = torch.meshgrid(grid_w, grid_h, grid_d)
        assert self.embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
        pos_dim = self.embed_dim // 6
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h), torch.sin(out_d), torch.cos(out_d)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), self.pre_logits(x[:, 365]) #0, 365
        else:
            return x[:, 0], x[:, 1]
    

class ConvStem3D(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        if not isinstance(img_size, (tuple, list)):
            img_size = to_3tuple(img_size)
        if not isinstance(patch_size, (tuple, list)):
            patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        # self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = in_chans, 2*embed_dim // patch_size[0]
        conv_length = max(1, int(np.log2(patch_size[0])))
        for l in range(conv_length):
            stem.append(nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm3d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            # stem.append(nn.GroupNorm(num_groups=32, num_channels=output_dim))
            # stem.append(nn.GELU())
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv3d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D== self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNC
        x = self.norm(x)
        return x


def vit_base(**kwargs):
    model = VisionTransformerND(stop_grad_conv1=False, is_3d=True, use_learnable_pos_emb=False, img_size=72, in_chans=1,
        patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem3D, num_classes=3, drop_rate=0, drop_path_rate=0)
    model.img_size = model.patch_embed.img_size
    return model