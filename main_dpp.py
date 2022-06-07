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
import random
import math
import warnings

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from torch.nn import SyncBatchNorm
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from tensorboardX import SummaryWriter

from datasets import build_dataset
from engine import train_one_epoch, evaluate_h
import torch.multiprocessing as mp
import torch.distributed as dist
from utilities import utils
from utilities.exp_utils import save_best_model, save_model
from models.model_lngrowth import LNGrowthNet
from losses.classification_loss import HierarchicalClass


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=60, type=int)

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

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=100, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--data-path', default="/data_local/data/train_data/nodule_growth/dataset/", type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='lngrowth', choices=['cifar100', 'cifar10', 'IMNET', 'lngrowth'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./output/lngrowth/lngrowth_maepre_p8_72_vit_stm_hloss0209',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # resume to pretrained model or trained checkpoint
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true', help='resume to checkpoint')
    parser.add_argument('--checkpoint-path', default='pretrained_models/mae_vit_base.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', default=False, help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--workers', default=4, type=int) #10
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--nodes', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--nr', type=int, default=0,
                        help='node rank.')
    parser.add_argument('--rank', type=int, default=0,
                        help='gpu rank.')
    parser.add_argument('--port', type=str, default='24455',
                        help='process rank.')
    parser.add_argument('--gpu', default='4', type=str,
                        help='GPU id to use.')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    return parser


def main():
    args = get_args_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.ngpus = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.world_size == -1:
        args.world_size = args.ngpus * args.nodes

    args.distributed = args.world_size > 1 or args.distributed

    log_dir = os.path.join(args.output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir
    save_model_dir = os.path.join(args.output_dir, 'checkpoints')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    args.save_model_dir = save_model_dir

    if args.distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        mp.spawn(main_worker, nprocs=args.ngpus, args=(args.ngpus, args))
    else:
        # Simply call main_worker function
        args.gpu = 0
        main_worker(args.gpu, args.ngpus, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        # global rank
        args.rank = args.nr * ngpus_per_node + gpu
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        dist.init_process_group("nccl", init_method='env://', rank=args.rank, world_size=args.world_size)
        torch.distributed.barrier()

        # compute per gpu
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else "cpu")
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    # dataset
    print('Create dataset')
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"Creating model: {args.model}")
    
    model = LNGrowthNet(args)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
    model_without_ddp = model

    if args.distributed:
        # sync_batchnorm and DistributedDataParallel
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 64
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)

    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = HierarchicalClass(device)

    output_dir = Path(args.output_dir)
    if args.resume:
    # if True:
        print("=> loading checkpoint '{}'".format(args.checkpoint_path))
        if args.checkpoint_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.checkpoint_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

        pretrain_dict = checkpoint['model']
        try:
            model_without_ddp.load_state_dict(pretrain_dict)
            print('***load parameters completely***')
        except:
            print('***load part parameters***')  
            model_dict = model_without_ddp.state_dict()
            pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
            print('Load params {}/{}'.format(len(pretrain_dict), len(model_dict)))
            model_dict.update(pretrain_dict)
            model_without_ddp.load_state_dict(model_dict)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    trained_epochs = []
    monitor_metrics = {}
    tensorboard_writer = SummaryWriter(args.log_dir)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn=None,
            set_training_mode=True
        )

        lr_scheduler.step(epoch)
        if (not args.distributed or (args.distributed and args.rank == 0)) and not epoch%1:
            print('Do validation')
            test_stats = evaluate_h(data_loader_val, model, device, criterion, num_class=args.nb_classes)
            metric_name = 'auc_agv' # auc, acc, accuracy, global_dice
            print(f"{metric_name} of the network on the {len(dataset_val)} test images: {test_stats[metric_name]:.3f}")
            # write tensorboard
            tensorboard_writer.add_scalar('macroauc', test_stats['macroauc'], epoch)
            tensorboard_writer.add_scalar('auc_agv', test_stats['auc_agv'], epoch)
            tensorboard_writer.add_scalar('auc0', test_stats['auc0'], epoch)
            tensorboard_writer.add_scalar('auc_h1', test_stats['auc_h1'], epoch)
            tensorboard_writer.add_scalar('auc1', test_stats['auc1'], epoch)
            tensorboard_writer.add_scalar('auc2', test_stats['auc2'], epoch)
            tensorboard_writer.add_scalar('val_loss', test_stats['loss'], epoch)
            tensorboard_writer.add_scalar('train_loss', train_stats['loss'], epoch)
            tensorboard_writer.add_scalar('lr', train_stats['lr'], epoch)

            if test_stats[metric_name] >= max_accuracy:
                checkpoint_path = os.path.join(args.save_model_dir, 'checkpoint_best.pth')
                save_model(model_without_ddp, model_ema, optimizer, lr_scheduler, epoch, loss_scaler, checkpoint_path)  

            max_accuracy = max(max_accuracy, test_stats[metric_name]) 
            print(f'Max {metric_name}: {max_accuracy:.3f}')     
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
            checkpoint_path = os.path.join(args.save_model_dir, 'checkpoint_lst.pth')
            save_model(model_without_ddp, model_ema, optimizer, lr_scheduler, epoch, loss_scaler, checkpoint_path)  

            trained_epochs.append(epoch)
            if metric_name not in monitor_metrics:
                monitor_metrics[metric_name] = []
            monitor_metrics[metric_name].append(test_stats[metric_name])
            # save best n checkpoint
            save_best_model(model_without_ddp, model_ema, optimizer, lr_scheduler, epoch, loss_scaler, trained_epochs, monitor_metrics, args.save_model_dir)
   
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()

