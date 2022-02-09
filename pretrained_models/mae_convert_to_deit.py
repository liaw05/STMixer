#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MAE Pre-Traind Model to DEiT')
    parser.add_argument('--input', default='/data/code/deit/pretrained_model/pretrain_mae_vit_base_mask_0.75_400e.pth', type=str, metavar='PATH',
                        help='path to moco pre-trained checkpoint')
    parser.add_argument('--output', default='/data/code/TS-Mixer/pretrained_models/mae_vit_base.pth', type=str, metavar='PATH',
                        help='path to output checkpoint in DEiT format')
    args = parser.parse_args()
    print(args)

    # load input
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('encoder') and not k.startswith('encoder.head'):
            # remove prefix
            state_dict[k.replace("encoder.", 'backbone.')] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # make output directory if necessary
    output_dir = os.path.dirname(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # save to output
    torch.save({'model': state_dict}, args.output)
