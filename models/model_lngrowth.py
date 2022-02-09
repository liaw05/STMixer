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
import numpy as np
from models.vit_3d import VisionTransformerND, ConvStem3D
from models.layers.stmixer import STMixer


class LNGrowthNet(nn.Module):
    def __init__(self, args):

        super(LNGrowthNet, self).__init__()
        self.backbone = VisionTransformerND(
            stop_grad_conv1=False, is_3d=True, use_learnable_pos_emb=False, 
            img_size=args.input_size, in_chans=1,
            patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem3D, 
            num_classes=args.nb_classes, drop_rate=args.drop, drop_path_rate=args.drop_path)

        self.num_features = self.backbone.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        self.fuze_block = STMixer(
            dim=self.num_features, mlp_ratio=4., qk_bias=True, drop=0.,
            drop_path=0, norm_layer=norm_layer, act_layer=act_layer)

        self.head_h1 = nn.Linear(self.num_features, 2)
        self.head_h2 = nn.Linear(self.num_features, 3)
    
    def forward(self, x):
        x0, x1, mask = x
        x10, x11 = self.backbone(x1) # current, T ct
        batch_size = x1.size(0)
        prefeat_tokens = self.prefeat_token.expand(batch_size, -1, -1).clone()
        if mask.sum():  # previous, T-1 ct
            _, x01 = self.forward_features(x0[mask>0])
            prefeat_tokens[mask>0] = x01[:, None]
        else:
            _, x01 = self.forward_features(x0)
            prefeat_tokens = x01*0 + prefeat_tokens

        x = torch.cat([x11[:,None], x10[:,None], prefeat_tokens], dim=1)
        x = self.fuze_block(x)
        
        h1 = self.head_h1(x)
        h2 = self.head_h2(x)
        return h1, h2
