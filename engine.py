"""
Train and eval functions used in main.py
"""
from enum import IntEnum
import math
import sys
from typing import Iterable, Optional

import torch
import csv
import numpy as np
from sklearn import metrics
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from torch.nn.utils import clip_grad_norm_

from losses.distillation_losses import DistillationLoss
from utilities import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cnn_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = tensor_to_cuda(samples, device)
        targets = tensor_to_cuda(targets, device)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(outputs)
            print(targets)
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        # torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(cnn_lr=optimizer.param_groups[-1]["lr"])

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion=None, num_class=2, save_csv=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    preds = []
    gts = []
    softmax = torch.nn.Softmax(dim=1)
    for images, target in metric_logger.log_every(data_loader, 10, header):
        # images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        images = tensor_to_cuda(images, device)
        target = tensor_to_cuda(target, device)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if criterion is not None:
                loss = criterion(images, output, target)
            else:
                loss = torch.tensor(0)

        metric_logger.update(loss=loss.item())
        # preds.append(output.sigmoid_())
        preds.append(softmax(output))
        gts.append(target)

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    preds = preds.cpu().detach().numpy()
    gts = gts.cpu().detach().numpy()
    if len(gts.shape)>1:
        gt_onehot = gts
        # gts = np.argmax(gts, axis=1)
        gts = gts[:,0]>0.5
    else:
        gt_onehot = np.eye(num_class)[gts]

    if save_csv is not None:
        csv_gt = save_csv[:-4] + '_gt.csv'
        csv_pred = save_csv[:-4] + '_pred.csv'
        header = ['level%d'%i for i in range(num_class)]
        write_csv(csv_gt, header, mul=False, mod='w')
        write_csv(csv_pred, header, mul=False, mod='w')
        write_csv(csv_gt, gt_onehot, mul=True, mod='a')
        write_csv(csv_pred, preds, mul=True, mod='a')

    # # multi-label metric auc
    auc_agv = 0
    macroauc = metrics.roc_auc_score(gt_onehot, preds, average='macro')
    weight = [0.25, 0.25, 0.5]
    for i in range(gt_onehot.shape[1]):
        auc = metrics.roc_auc_score(gt_onehot[:,i], preds[:,i])
        metric_logger.meters['auc{}'.format(i)].update(auc, n=len(preds))
        auc_agv += auc*weight[i]
    metric_logger.meters['auc_agv'].update(auc_agv, n=len(preds))
    metric_logger.meters['macroauc'].update(macroauc, n=len(preds))
    
    # # metric auc
    # auc = metrics.roc_auc_score(gt_onehot, preds, average='weighted')
    # metric_logger.meters['auc'].update(auc, n=len(preds))

    pred_y = np.argmax(preds, axis=1)
    # pred_y = preds[:,0]>0.5
    kp = metrics.cohen_kappa_score(gts, pred_y, weights='quadratic')
    metric_logger.meters['kp'].update(kp, n=len(preds))

    acc = metrics.accuracy_score(gts, pred_y)
    metric_logger.meters['acc'].update(acc, n=len(preds))

    print('* ACC {acc.global_avg:.3f} MacroAUC {macroauc.global_avg:.3f} AUC_Agv {auc_agv.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(acc=metric_logger.acc, macroauc=metric_logger.macroauc, auc_agv=metric_logger.auc_agv, losses=metric_logger.loss))
    print('* AUC0 {auc0.global_avg:.3f} AUC1 {auc1.global_avg:.3f} AUC2 {auc2.global_avg:.3f}'
          .format(auc0=metric_logger.auc0, auc1=metric_logger.auc1, auc2=metric_logger.auc2))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_h(data_loader, model, device, criterion=None, num_class=2, save_csv=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    preds_h1 = []
    preds_h2 = []
    gts = []
    softmax = torch.nn.Softmax(dim=1)
    for images, target in metric_logger.log_every(data_loader, 10, header):
        # images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        images = tensor_to_cuda(images, device)
        target = tensor_to_cuda(target, device)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if criterion is not None:
                # loss = criterion(images, output, target)
                loss = criterion(output, target)
            else:
                loss = torch.tensor(0)

        metric_logger.update(loss=loss.item())
        h1, h2 = output
        h1 = softmax(h1)
        h2 = softmax(h2)

        preds_h1.append(h1)
        preds_h2.append(h2)
        gts.append(target)

    preds_h1 = torch.cat(preds_h1, dim=0)
    preds_h2 = torch.cat(preds_h2, dim=0)
    gts = torch.cat(gts, dim=0)
    preds_h1 = preds_h1.cpu().detach().numpy()
    preds_h2 = preds_h2.cpu().detach().numpy()
    preds = preds_h2
    gts = gts.cpu().detach().numpy()
    gt_h1 = gts>1
    if len(gts.shape)>1:
        gt_onehot = gts
        # gts = np.argmax(gts, axis=1)
        gts = gts[:,0]>0.5
    else:
        gt_onehot = np.eye(num_class)[gts]

    if save_csv is not None:
        csv_gt = save_csv[:-4] + '_gt.csv'
        csv_pred = save_csv[:-4] + '_pred.csv'
        header = ['level%d'%i for i in range(num_class)]
        write_csv(csv_gt, header, mul=False, mod='w')
        write_csv(csv_pred, header, mul=False, mod='w')
        write_csv(csv_gt, gt_onehot, mul=True, mod='a')
        write_csv(csv_pred, preds, mul=True, mod='a')

    # # multi-label metric auc
    auc_agv = 0
    macroauc = metrics.roc_auc_score(gt_onehot, preds, average='macro')
    weight = [0.25, 0.25, 0.5]
    for i in range(gt_onehot.shape[1]):
        auc = metrics.roc_auc_score(gt_onehot[:,i], preds[:,i])
        metric_logger.meters['auc{}'.format(i)].update(auc, n=len(preds))
        auc_agv += auc*weight[i]
    metric_logger.meters['auc_agv'].update(auc_agv, n=len(preds))
    metric_logger.meters['macroauc'].update(macroauc, n=len(preds))

    auc_h1 = metrics.roc_auc_score(gt_h1, preds_h1[:,1])
    metric_logger.meters['auc_h1'].update(auc_h1, n=len(preds))
    
    # # metric auc
    # auc = metrics.roc_auc_score(gt_onehot, preds, average='weighted')
    # metric_logger.meters['auc'].update(auc, n=len(preds))

    pred_y = np.argmax(preds, axis=1)
    # pred_y = preds[:,0]>0.5
    kp = metrics.cohen_kappa_score(gts, pred_y, weights='quadratic')
    metric_logger.meters['kp'].update(kp, n=len(preds))

    acc = metrics.accuracy_score(gts, pred_y)
    metric_logger.meters['acc'].update(acc, n=len(preds))

    print('* ACC {acc.global_avg:.3f} MacroAUC {macroauc.global_avg:.3f} AUC_Agv {auc_agv.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(acc=metric_logger.acc, macroauc=metric_logger.macroauc, auc_agv=metric_logger.auc_agv, losses=metric_logger.loss))
    print('* AUC0 {auc0.global_avg:.3f} AUC1 {auc1.global_avg:.3f} AUC2 {auc2.global_avg:.3f} AUC_h1 {auc_h1.global_avg:.3f}'
          .format(auc0=metric_logger.auc0, auc1=metric_logger.auc1, auc2=metric_logger.auc2, auc_h1=metric_logger.auc_h1))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def write_csv(csv_name, content, mul=True, mod="w"):
    with open(csv_name, mod) as myfile:
        mywriter = csv.writer(myfile)
        if mul:
            mywriter.writerows(content)
        else:
            mywriter.writerow(content)


def tensor_to_cuda(tensor, device):
    if isinstance(tensor, dict):
        for key in tensor:
            tensor[key] = tensor_to_cuda(tensor[key], device)
        return tensor
    elif isinstance(tensor, (list, tuple)):
        tensor = [tensor_to_cuda(t, device) for t in tensor]
        return tensor
    else:
        return tensor.to(device)