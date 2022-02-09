#encoding=utf-8
import os
import time
import glob
import math
import random
import warnings

import numpy as np
import pandas as pd
import torch
from med_dataset import transforms as med_transforms


def build_transform_med(is_train, input_size=96):
    transform_train = med_transforms.Compose([ 
        # transforms.Pad(padding=8, pad_value=170),
        med_transforms.CenterPadCrop(input_size+16),
        med_transforms.RandomPadCrop(input_size),
        med_transforms.RandomXFlip(),
        # med_transforms.RandomYFlip(),
        med_transforms.RandomZFlip(),
        # med_transforms.Normalize(), #to_zero_mean=True
        med_transforms.Standardize(mean=-461.1, std=453.1), #for nodule-growth
        med_transforms.ZeroOut(4),
        med_transforms.ToTensor(),
    ])
    transform_test = med_transforms.Compose([
        med_transforms.CenterPadCrop(input_size),
        # med_transforms.Normalize(), #to_zero_mean=True
        med_transforms.Standardize(mean=-461.1, std=453.1),
        med_transforms.ToTensor(),
    ])
    return transform_train if is_train else transform_test


class LnGrowthDataset(torch.utils.data.Dataset):

    def __init__(self, root, train=True, input_size=72):

        super(LnGrowthDataset, self).__init__()
        self.input_size = input_size
        self.transform = build_transform_med(is_train=train, input_size=self.input_size)

        self.root = root
        print('Data path: ', self.root)
        self.train = train
        if self.train:
            csv_path = "/data/Metrics/nodulegrowth/growthlabel_csv/growth_nodule_train0208.csv"
        else:     
            csv_path = "/data/Metrics/nodulegrowth/growthlabel_csv/growth_nodule_valtest0208.csv"
            # csv_path = "/data/Metrics/nodulegrowth/growthlabel_csv/growth_nodule_val0208.csv"
            # csv_path = "/data/Metrics/nodulegrowth/growthlabel_csv/growth_nodule_test0208.csv"
            # csv_path = "/data/Metrics/nodulegrowth/growthlabel_csv/cvte_test.csv"

        df = pd.read_csv(csv_path)
        self.ids, self.labels = df['id'].values, df['Seg_label'].values
        path0, path1, path2 = df['path0'].values, df['path1'].values, df['path2'].values
        self.path_list = [(path1[i], path2[i]) if isinstance(path2[i], str) else (path0[i], path1[i]) for i in range(len(path1))]
        print('Num of {} train: {}'.format(self.train, len(self.path_list)))
        self.path0, self.path1, self.path2 = path0, path1, path2
    
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        # path0, path1 = self.path_list[index]

        path0, path1 = self.path0[index], self.path1[index]
        if self.train:
            t = time.time()
            np.random.seed(int(str(t % 1)[2:7]))
            if np.random.uniform()<0.4:
                path0 = None

        pre_img = 1
        if isinstance(path0, str):
            data_npz = np.load(os.path.join(self.root, path0))
            data_dict = dict(data_npz)
            img = data_dict['img'].astype(np.float32).squeeze()
            img = np.clip(img, -1000, 400)[None,] # C*D*H*W
            target = self.labels[index]
            if self.transform is not None:
                img = self.transform(data=img)['data']
        else:
            pre_img = 0
            img = np.zeros((self.input_size, self.input_size, self.input_size), dtype=np.float32)[None]
            img = torch.from_numpy(img)
        # return img, target
    
        data_npz = np.load(os.path.join(self.root, path1))
        data_dict = dict(data_npz)
        img1 = data_dict['img'].astype(np.float32).squeeze()
        img1 = np.clip(img1, -1000, 400)[None,] # C*D*H*W
        target = self.labels[index]
        if self.transform is not None:
            img1 = self.transform(data=img1)['data']

        return (img, img1, pre_img), target

    
