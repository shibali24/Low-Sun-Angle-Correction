import os
import sys
import random
import math
import importlib
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import skimage
from util_unet import load_data_unit, reshape_image
from PIL import Image
import pickle

plt.rc('font', family='serif')
plt.rcParams['image.cmap'] = 'viridis'

ROOT_DIR = os.path.abspath("./tf_unet/")
sys.path.append(ROOT_DIR)

from tf_unet import unet, util, image_util

#MODEL_PATH = './out_path_20190603-adam_l4/'
MODEL_PATH = './out_paths/out_path_20190815-adam_l4-f64-dp75/'
#MODEL_PATH = './out_paths/out_path_20191008-adam_l4-f64-dp75_Sentinel/'
TRAIN_DATA_DIR = "./dense-train-all-original"

data_provider = image_util.ImageDataProviderWithWeights(TRAIN_DATA_DIR + "/*.png",
                                                        data_suffix=".png",
                                                        mask_suffix="_mask.png",
                                                        border_suffix="_border.png"
                                                        )
layers = 4
features_root = 64
##################################################
## check prediction on a random image in train-all
def test_one_random(data_provider, gt=True):

    #setup & training
    x_test, y_test, w_test = data_provider(1)

    net = unet.Unet(layers=layers, features_root=64, channels=3, n_class=2)

    prediction = net.predict(MODEL_PATH + 'model.ckpt', x_test)

    if gt:
        fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(12,5))
        ax[0].imshow(x_test[0,...], aspect="equal")
        ax[1].imshow(y_test[0,...,1], aspect="equal")
        mask = prediction[0,...,1] #> 0.05
        mask = reshape_image(mask, x_test.shape[1], x_test.shape[2], dtype=np.float64)
        print(mask.shape, x_test.shape)
        pos = ax[2].imshow(mask, aspect="equal", vmin=0, vmax=1)
        ax[0].set_title("Input")
        ax[1].set_title("Ground truth")
        ax[2].set_title("Prediction")
        fig.tight_layout()

    else:
        fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12,5))
        ax[0].imshow(x_test[0,...], aspect="equal")
        mask = prediction[0,...,1] #> 0.4
        mask = reshape_image(mask, x_test.shape[1], x_test.shape[2], dtype=np.float64)
        print(mask.shape, x_test.shape)
        pos = ax[1].imshow(mask, aspect="equal", vmin=0, vmax=1)
        ax[0].set_title("Input")
        ax[1].set_title("Prediction")
        fig.tight_layout()
        _ = fig.colorbar(pos, ax=ax[1])

    fig.savefig(MODEL_PATH + "test_orig-n.png", dpi=300)
    plt.clf()


#############################################
## check individual images - defined by IMAGE
def test_one_specified(IMAGE, save_name, ckpt_num):
    net = unet.Unet(layers=layers, features_root=features_root, channels=3, n_class=2)

    x_test = load_data_unit(IMAGE)

    prediction = net.predict(MODEL_PATH + 'model.ckpt-{}'.format(ckpt_num), x_test)

    print('prediction shape: {}'.format(prediction.shape) )
    print('input shape: {}'.format(x_test.shape) )

    fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12,5))
    ax[0].imshow(skimage.io.imread(IMAGE), aspect="equal")

    mask = prediction[0,...,1] #> 0.9
#    mask = reshape_image(mask, x_test.shape[1], x_test.shape[2], dtype=np.float64)
    #ax[1].imshow(unit_sample[0,...], aspect="equal")
    pos = ax[1].imshow(mask, aspect="equal", vmin=0, vmax=1)
    ax[0].set_title("Input")
    ax[1].set_title("Prediction")
    fig.tight_layout()
    _ = fig.colorbar(pos, ax=ax[1])
#    fig.savefig(MODEL_PATH + save_name, dpi=300)
    fig.savefig( save_name[:-4]+'_{}.png'.format(ckpt_num), dpi=300)
    print("prediction for {} saved".format(IMAGE) )

    plt.clf()

############################################
def record_all_loss(data_provider):
    size_all = data_provider.whole_batch_size
    startd_idx = data_provider.file_idx

    batch_x, batch_y, batch_w = data_provider(size_all)
    prediction, cost = net.predict_with_cost(MODEL_PATH + 'model.ckpt', batch_x, batch_y, batch_w, max_batch = 5)

    res = {}
    for i in range(len(data_provider.data_files)):
        res[data_provider.data_files[startd_idx+i]] = (cost[i], prediction[i, ...])
    pickle.dump(res, open(MODEL_PATH + 'costs.p', 'wb'))

############################################
#test_one_random(data_provider, gt=False)

NAME = "planet_tile_3600-10200.png"
IMAGE = '../imagery/Planet/kanger2_bad/tiles_planet_png_reshaped_hist_fix/' + NAME
test_one_specified(IMAGE, IMAGE[:-4] + "_prediction.png", ckpt_num=487)

#NAME = "zz123-reshaped.png"
#IMAGE = './fig-tests/' + NAME
#test_one_specified(IMAGE, IMAGE[:-4] + "_prediction.png", ckpt_num=487)

#NAME = "planet_tile_4200-10200.png"
#IMAGE = './fig-tests/' + NAME
#test_one_specified(IMAGE, IMAGE[:-4] + "_prediction.png", ckpt_num=487)

#IMAGE = './fig-tests/landsat_resample_tile_3900-18850.png'
#test_one_specified(IMAGE, './fig-tests/landsat_resample_tile_3900-18850_prediction.png', ckpt_num=49)
#IMAGE = './fig-tests/planet_tile_4800-9600.png'
#test_one_specified(IMAGE, './fig-tests/planet_tile_4800-9600_prediction.png', ckpt_num=487)
#IMAGE = './fig-tests/zzz123.png'
#test_one_specified(IMAGE, './fig-tests/zzz123_prediction.png', ckpt_num=487)
#IMAGE = './fig-tests/zzz-bad.png'
#test_one_specified(IMAGE, './fig-tests/zzz-bad_prediction.png', ckpt_num=487)
#IMAGE = './fig-tests/planet_tile_7200-1800-fixed.png'
#test_one_specified(IMAGE, './fig-tests/planet_tile_7200-1800-fixed_prediction.png', ckpt_num=487)
#IMAGE = './fig-tests/sentinel_resample_tile_2600-21450.png'
#test_one_specified(IMAGE, './fig-tests/sentinel_resample_tile_2600-21450_prediction-retrained.png', ckpt_num=1)
#IMAGE = './fig-tests/5dw5qe9ar25eagtrwk72_al_3_si_4_af_10.png'
#test_one_specified(IMAGE, './fig-tests/5dw5qe9ar25eagtrwk72_al_3_si_4_af_10_retrained-prediction.png', ckpt_num=49)
#IMAGE = './fig-tests/sentinel_resample_tile_9600-4800.png'
#test_one_specified(IMAGE, './fig-tests/sentinel_resample_tile_9600-4800_retrained-prediction.png', ckpt_num=49)
#IMAGE = './fig-tests/planet_tile_1800-25800.png'
#test_one_specified(IMAGE, './fig-tests/planet_tile_1800-25800_prediction-retrained.png', ckpt_num=0)
############################################

#net = unet.Unet(layers=layers, features_root=64, channels=3, n_class=2, cost="my_cost")
#record_all_loss(data_provider)

