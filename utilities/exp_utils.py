import os
import sys
import csv
import logging
import subprocess
import glob

import torch
import numpy as np
import pandas as pd
import importlib
from collections import OrderedDict
from timm.utils import get_state_dict


def save_best_model(model, model_ema, optimizer, lr_scheduler, epoch, loss_scaler, trained_epochs, monitor_metrics, save_model_dir, bestk=5):
    keys = monitor_metrics.keys()
    for key in ['auc_agv', 'macroauc', 'auc', 'acc', 'dice', 'c-index', 'loss']:
        if key in keys:
            break
    val_losses = monitor_metrics[key]
    val_losses = np.array(val_losses)
    index_ranking = np.argsort(val_losses) if key=='loss' else np.argsort(-val_losses)
    epoch_ranking = np.array(trained_epochs)[index_ranking]
    
    # check if current epoch is among the best-k epchs.
    if epoch in epoch_ranking[:bestk]:
        checkpoint_path = os.path.join(save_model_dir, 'checkpoint_%03d.pth' % epoch)
        save_model(model, model_ema, optimizer, lr_scheduler, epoch, loss_scaler, checkpoint_path)

        # delete params of the epoch that just fell out of the best-k epochs.
        if len(epoch_ranking) > bestk:
            epoch_rm = epoch_ranking[bestk]
            subprocess.call('rm {}'.format(os.path.join(save_model_dir, 'checkpoint_%03d.pth' % epoch_rm)), shell=True)


def save_model(model, model_ema, optimizer, lr_scheduler, epoch, loss_scaler, checkpoint_path):    
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'model_ema': get_state_dict(model_ema),
        'scaler': loss_scaler.state_dict(),
    }, 
    checkpoint_path,
    _use_new_zipfile_serialization=False)


def load_checkpoint(checkpoint_path, net, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrain_dict = checkpoint['state_dict']
    try:
        if hasattr(net, 'module'):
            pretrain_dict = {'module.'+k:v for k,v in pretrain_dict.items()}
        net.load_state_dict(pretrain_dict)
        print('***load parameters completely***')
    except:
        print('***load part parameters***')
        model_dict = net.state_dict()
        pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
        print('Num of param:', len(pretrain_dict))
        model_dict.update(pretrain_dict)
        net.load_state_dict(model_dict)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']