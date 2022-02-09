import os
import random
import numpy as np
from PIL import Image


# Cal mean std
def compute_mn_std_channel(data_dir, num=2000, i=0):
    pixvlu, npix = 0, 0
    for fname in os.listdir(data_dir)[:num]:
        if fname.endswith('.jpeg'):
            img = Image.open(os.path.join(data_dir, fname)).convert("RGB")
            img = np.array(img)[:,:,i]/255.0
            pixvlu += np.sum(img)
            npix += np.prod(img.shape)
    pixmean = pixvlu / float(npix)
    print('Mean: ', pixmean)

    pixvlu = 0
    for fname in os.listdir(data_dir)[:num]:
        if fname.endswith('.jpeg'):
            img = Image.open(os.path.join(data_dir, fname)).convert("RGB")
            img = np.array(img)[:,:,i]/255.0
            img = img-pixmean
            pixvlu += np.sum(img * img)
    pixstd = np.sqrt(pixvlu / float(npix))
    print('Std: ', pixstd)


# Cal mean std
def compute_mn_std(data_dir, num=2000):
    pixvlu, npix = 0, 0
    fns = os.listdir(data_dir)
    random.shuffle(fns)
    for fname in fns[:num]:
        if fname.endswith('.npz'):
            img = np.load(os.path.join(data_dir, fname))['img']
            img = np.clip(img, -1000, 400)
            pixvlu += np.sum(img)
            npix += np.prod(img.shape)
    pixmean = pixvlu / float(npix)
    print('Mean: ', pixmean)

    pixvlu = 0
    for fname in fns[:num]:
        if fname.endswith('.npz'):
            img = np.load(os.path.join(data_dir, fname))['img']
            img = np.clip(img, -1000, 400)
            img = img-pixmean
            pixvlu += np.sum(img * img)
    pixstd = np.sqrt(pixvlu / float(npix))
    print('Std: ', pixstd)


if __name__ == '__main__':
    data_dir = "/data_local/data/train_data/nodule_growth/dataset/theta_1.5/"
    # mean: 179.29755471522114, std:49.76069488655138(32)
    # mean: 162.8, std:57.6(64)
    #-607.7044612776176, 400.2193869097047
    # for i in range(3):
    #     compute_mn_std_channel(data_dir, num=1000, i=i)
    
    compute_mn_std(data_dir, num=1000)