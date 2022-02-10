# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from timm.models import create_model
from models.model_lngrowth import LNGrowthNet

from datasets import build_dataset
from engine import evaluate_h


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--gpu', default='4', type=str,
                        help='GPU id to use.')

    # Model parameters
    parser.add_argument('--model', default='vit_stm_patch8_72', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=72, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Dataset parameters
    parser.add_argument('--data-path', default='/data_local/data/train_data/nodule_growth/dataset/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='lngrowth', choices=['cifar100', 'cifar10', 'lngrowth'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./output/lngrowth/lngrowth_maepre_p8_72_vit_stm_hloss0209',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_csv', default='./output/lngrowth/lngrowth_maepre_p8_72_vit_stm_hloss0209/results.csv',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    # resume checkpoint
    parser.add_argument('--resume', default="./output/lngrowth/lngrowth_maepre_p8_72_vit_stm_hloss0209/checkpoints/checkpoint_best.pth", help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")
    model = LNGrowthNet(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
   
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print('***load checkpoint at epoch {}***'.format(checkpoint['epoch']))

    start_time = time.time()
    test_stats = evaluate_h(data_loader_val, model, device, num_class=args.nb_classes, save_csv=args.save_csv)
    metric_name = 'macroauc'
    print(f"{metric_name} of the network on the {len(dataset_val)} test images: {test_stats[metric_name]:.3f}")
     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
