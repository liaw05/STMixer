import random
import numbers
import torch
import torchvision
from torchvision.transforms import functional as F
from scipy.ndimage import zoom
import numpy as np
from PIL import ImageFilter
import math
import torch
from torch import Tensor


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        '''
        data_dict: must have the key of 'data', other keys of ['mask', 'centerd'] are optional.
        data: shape of [C,D,H,W]
        mask: shape of [C,D,H,W]
        centerd: shape of [N,6]
        '''
        for key in ['mask', 'centerd']:
            if key not in data_dict:
                data_dict[key] = None
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Pad(object):
    def __init__(self, padding=0, pad_value=0, dims=3):
        self.padding = int(padding)
        self.pad_value = pad_value
        self.dims = dims

    def __call__(self, **data_dict):
        if self.padding > 0:
            bg = int(self.padding/2)
            pad_list = [[bg, self.padding-bg] for _ in range(self.dims)]
            pad_list.insert(0, [0,0])
            data_dict['data'] = np.pad(data_dict['data'], pad_list, mode='constant', constant_values=self.pad_value)
            if data_dict['mask'] is not None:
                data_dict['mask'] = np.pad(data_dict['mask'], pad_list, mode='constant', constant_values=0)
            if data_dict['centerd'] is not None:
                data_dict['centerd'] = data_dict['centerd'][:,:3]+bg
        return data_dict


class RandomCrop(object):
    """Crop the given Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (d,h,w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = [int(size), int(size), int(size)]
        else:
            self.size = [int(v) for v in size]

    def __call__(self, **data_dict):
        img = data_dict['data']
        for ii in range(len(img.shape)):
            diff = img.shape[ii]-self.size[ii]
            if diff>0:
                min_crop = random.randint(0, diff)
                max_crop = min_crop + self.size[ii]
                data_dict['data'] = np.take(data_dict['data'], indices=range(min_crop, max_crop), axis=ii)
                if data_dict['mask'] is not None:
                    data_dict['mask'] = np.take(data_dict['mask'], indices=range(min_crop, max_crop), axis=ii)
                if data_dict['centerd'] is not None:
                    data_dict['centerd'][:, ii] -= min_crop

        return data_dict


class CenterCrop(object):
    """Crops the given Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = [int(size), int(size), int(size)]
        else:
            self.size = [int(v) for v in size]

    def __call__(self, **data_dict):
        img = data_dict['data']
        for ii in range(len(img.shape)):
            diff = img.shape[ii]-self.size[ii]
            if diff>0:
                min_crop = diff//2
                max_crop = min_crop + self.size[ii]
                data_dict['data'] = np.take(data_dict['data'], indices=range(min_crop, max_crop), axis=ii)
                if data_dict['mask'] is not None:
                    data_dict['mask'] = np.take(data_dict['mask'], indices=range(min_crop, max_crop), axis=ii)
                if data_dict['centerd'] is not None:
                    data_dict['centerd'][:, ii] -= min_crop

        return data_dict


class RandomPadCrop(object):
    """Crop the given Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (d,h,w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """
    def __init__(self, size, pad_value=0, dims=3):
        if isinstance(size, numbers.Number):
            self.size = [int(size) for _ in range(dims)]
        else:
            self.size = [int(v) for v in size]
        self.pad_value = pad_value

    def __call__(self, **data_dict):
        # pad to crop size
        image_shape = data_dict['data'].shape[1:]
        if np.any([image_shape[dim] < ps for dim, ps in enumerate(self.size)]):
            new_shape = [max(image_shape[i], self.size[i]) for i in range(len(self.size))]
            difference = np.array(new_shape) - image_shape
            pad_below = difference // 2
            pad_above = difference-pad_below
            pad_list = [list(i) for i in zip(pad_below, pad_above)]
            pad_list.insert(0, [0,0])
            data_dict['data'] = np.pad(data_dict['data'], pad_list, mode='constant', constant_values=self.pad_value)
            if data_dict['mask'] is not None:
                data_dict['mask'] = np.pad(data_dict['mask'], pad_list, mode='constant', constant_values=0)
            if data_dict['centerd'] is not None:
                data_dict['centerd'] = data_dict['centerd'][:,:3] + np.array(pad_below)

        # random crop
        image_shape = data_dict['data'].shape[1:]
        for ii in range(len(image_shape)):
            diff = image_shape[ii]-self.size[ii]
            if diff>0:
                min_crop = random.randint(0, diff)
                max_crop = min_crop + self.size[ii]
                data_dict['data'] = np.take(data_dict['data'], indices=range(min_crop, max_crop), axis=ii+1)
                if data_dict['mask'] is not None:
                    data_dict['mask'] = np.take(data_dict['mask'], indices=range(min_crop, max_crop), axis=ii+1)
                if data_dict['centerd'] is not None:
                    data_dict['centerd'][:, ii] -= min_crop

        return data_dict


class CenterPadCrop(object):
    """Crops the given Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """
    def __init__(self, size, pad_value=0, dims=3):
        if isinstance(size, numbers.Number):
            self.size = [int(size) for _ in range(dims)]
        else:
            self.size = [int(v) for v in size]
        self.pad_value = pad_value

    def __call__(self, **data_dict):
        # pad to crop size
        image_shape = data_dict['data'].shape[1:]
        if np.any([image_shape[dim] < ps for dim, ps in enumerate(self.size)]):
            new_shape = [max(image_shape[i], self.size[i]) for i in range(len(self.size))]
            difference = np.array(new_shape) - image_shape
            pad_below = difference // 2
            pad_above = difference-pad_below
            pad_list = [list(i) for i in zip(pad_below, pad_above)]
            pad_list.insert(0, [0,0])
            data_dict['data'] = np.pad(data_dict['data'], pad_list, mode='constant', constant_values=self.pad_value)
            if data_dict['mask'] is not None:
                data_dict['mask'] = np.pad(data_dict['mask'], pad_list, mode='constant', constant_values=0)
            if data_dict['centerd'] is not None:
                data_dict['centerd'] = data_dict['centerd'][:,:3] + np.array(pad_below)

        # random crop
        image_shape = data_dict['data'].shape[1:]
        for ii in range(len(image_shape)):
            diff = image_shape[ii]-self.size[ii]
            if diff>0:
                min_crop = diff//2
                max_crop = min_crop + self.size[ii]
                data_dict['data'] = np.take(data_dict['data'], indices=range(min_crop, max_crop), axis=ii+1)
                if data_dict['mask'] is not None:
                    data_dict['mask'] = np.take(data_dict['mask'], indices=range(min_crop, max_crop), axis=ii+1)
                if data_dict['centerd'] is not None:
                    data_dict['centerd'][:, ii] -= min_crop

        return data_dict

        
class RandomResize(object):
    def __init__(self, scale=(0.7,1.3), order=1):
        self.scale = scale
        self.order = order

    def __call__(self, **data_dict):
        sp_scale = np.array([np.random.uniform(self.scale[0], self.scale[1]) for _ in range(len(self.scale))])
        data_dict['data'] = zoom(data_dict['data'], list(sp_scale), order=self.order)
        if data_dict['mask'] is not None:
            data_dict['mask'] = zoom(data_dict['mask'], list(sp_scale), order=0)
        if data_dict['centerd'] is not None:
            data_dict['centerd'][:, :3] *= sp_scale
            data_dict['centerd'][:, 3:6] *= sp_scale

        return data_dict


class RandomFlipAxis(object):
    def __init__(self, prob=0.5, axis=0):
        self.prob = prob
        self.axis = axis

    def __call__(self, **data_dict):
        """
        data/mask:[c,z,y,x]
        centerd:N*6
        """
        if random.random() < self.prob:
            ii = self.axis
            data_dict['data'] = np.flip(data_dict['data'], ii)
            if data_dict['mask'] is not None:
                data_dict['mask'] = np.flip(data_dict['mask'], ii)
            if data_dict['centerd'] is not None:
                data_dict['centerd'][:, ii-1] = data_dict['data'].shape[ii-1]-1 - data_dict['centerd'][:, ii-1]
        return data_dict


class RandomZFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, **data_dict):
        if random.random() < self.prob:
            ii = 1
            data_dict['data'] = np.flip(data_dict['data'], ii)
            if data_dict['mask'] is not None:
                data_dict['mask'] = np.flip(data_dict['mask'], ii)
            if data_dict['centerd'] is not None:
                data_dict['centerd'][:, ii-1] = data_dict['data'].shape[ii-1]-1 - data_dict['centerd'][:, ii-1]
        return data_dict


class RandomYFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, **data_dict):
        if random.random() < self.prob:
            ii = 2
            data_dict['data'] = np.flip(data_dict['data'], ii)
            if data_dict['mask'] is not None:
                data_dict['mask'] = np.flip(data_dict['mask'], ii)
            if data_dict['centerd'] is not None:
                data_dict['centerd'][:, ii-1] = data_dict['data'].shape[ii-1]-1 - data_dict['centerd'][:, ii-1]
        return data_dict


class RandomXFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, **data_dict):
        if random.random() < self.prob:
            ii = 3
            data_dict['data'] = np.flip(data_dict['data'], ii)
            if data_dict['mask'] is not None:
                data_dict['mask'] = np.flip(data_dict['mask'], ii)
            if data_dict['centerd'] is not None:
                data_dict['centerd'][:, ii-1] = data_dict['data'].shape[ii-1]-1 - data_dict['centerd'][:, ii-1]
        return data_dict


class RandomSwap(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, **data_dict):
        if random.random() < self.prob:
            axisorder = np.random.permutation(3)   #将维度顺序[0,1,2]打乱
            data_dict['data'] = np.transpose(data_dict['data'], axisorder)
            if data_dict['mask'] is not None:
                data_dict['mask'] = np.transpose(data_dict['mask'], axisorder)
            if data_dict['centerd'] is not None:
                data_dict['centerd'][:, :3] = data_dict['centerd'][:, :3][:, axisorder]
                data_dict['centerd'][:, 3:6] = data_dict['centerd'][:, 3:6][:, axisorder]

        return data_dict


class Standardize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean).reshape(-1,1,1,1)
        self.std = np.array(std).reshape(-1,1,1,1) + 1e-5

    def __call__(self, **data_dict):
        data_dict['data'] = (data_dict['data']-self.mean)/self.std
        return data_dict


class Normalize(object):
    def __init__(self, to255=False, to_zero_mean=False, vmin=-1000, vmax=600):
        '''
        to255 and to_zero_mean cann't be True at the same time.
        if to255 is True, data range [0,255], if to_zero_mean is True data range [-1,1], else [0,1].
        '''
        self.to255 = to255
        self.to_zero_mean = to_zero_mean
        self.vmin = vmin
        self.vmax = vmax
        if self.to255:
            self.to_zero_mean = False

    def __call__(self, **data_dict):
        data_dict['data'] = np.clip(data_dict['data'], self.vmin, self.vmax)
        data_dict['data'] = (data_dict['data']-self.vmin)/(self.vmax-self.vmin+1e-5)
        if self.to255:
            data_dict['data'] *= 255.0
        if self.to_zero_mean:
            data_dict['data'] = (data_dict['data']-0.5)*2

        return data_dict


class ZeroOut(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = [int(size), int(size), int(size)]
        else:
            self.size = [int(v) for v in size]

    def __call__(self, **data_dict):
        d,h,w = data_dict['data'].shape[1:] #size
        sz, sy, sx = min(self.size[0], d//2), min(self.size[1], h//2), min(self.size[2],w//2)
        x1 = random.randint(0, w-sx)
        y1 = random.randint(0, h-sy)
        z1 = random.randint(0, d-sz)
        data_dict['data'][:, z1:z1+sz, y1:y1+sy, x1:x1+sx] = np.array(np.zeros((sz, sy, sx)))
        return data_dict


class ZeroOut2D(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = [int(size), int(size)]
        else:
            self.size = [int(v) for v in size]

    def __call__(self, **data_dict):
        h,w = data_dict['data'].shape #size
        sy, sx = min(self.size[0], h//2), min(self.size[1],w//2)
        x1 = random.randint(0, w-sx)
        y1 = random.randint(0, h-sy)
        data_dict['data'][y1:y1+sy, x1:x1+sx] = np.array(np.zeros((sy, sx)))
        return data_dict


class ToTensor(object):
    def __call__(self, **data_dict):
        '''
        expand channel dim and convert to tensor.
        '''
        data_dict['data'] = torch.from_numpy(data_dict['data'].copy().astype(np.float32))
        if data_dict['mask'] is not None:
            data_dict['mask'] = data_dict['mask'].copy()

        return data_dict