import numpy as np
from scipy import ndimage
import skimage
import matplotlib.pyplot as plt
import os, glob, pickle
from skimage import data
from skimage import filters
from skimage import exposure


MASK_DIR = "./train-all"
image = skimage.io.imread("zz-sample-image.png")/255
mask  = skimage.io.imread("zz-sample-label.png", dtype=int)/255
labels, numbers = ndimage.label(mask)

## sobel filter
Sobel = False
if Sobel:
    im = image
    mode = 'constant'
    cval = 0
    sx = ndimage.sobel(im, axis=0, mode=mode, cval=cval)
    sy = ndimage.sobel(im, axis=1, mode=mode, cval=cval)
    sob = np.hypot(sx, sy)

    plt.imsave("zz-sample-image-sobel.png", sob)

## Otsu filter
Otsu = False
if Otsu:
    camera = image[...,0]
    val = filters.threshold_otsu(camera)

    hist, bin_center = exposure.histogram(camera)

    plt.imsave("zz-sample-image-otsu.png", camera < val, cmap='gray')

# frequency size count
fq_size = False
if fq_size:
    masks = glob.glob(MASK_DIR + '/*_mask.png')

    data_dict = {}
    for _mask in masks:

        mask_name = os.path.basename(_mask)

        mask = skimage.io.imread(_mask, dtype=int)/255
        label_im, nb_labels = ndimage.label(mask)

        sizes_per_iceberg = ndimage.sum(mask, label_im, range(nb_labels + 1) )
        sizes_sum = ndimage.sum(mask, label_im)
        print(sizes_per_iceberg, sizes_per_iceberg.sum())
        data_dict[mask_name] = sizes_per_iceberg

    #    print("image {} has {} icebergs with sizes {}".format(_mask, nb_labels, sizes) )

    pickle.dump(data_dict, open('zz.p', 'wb') )

