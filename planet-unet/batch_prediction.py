import os
import sys
import random
import math
import importlib
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage
from PIL import Image
import pickle
from scipy import ndimage

ROOT_DIR = os.path.abspath("./tf_unet/")
sys.path.append(ROOT_DIR)

from tf_unet import unet, util, image_util

MODEL_PATH = './out_paths/out_path_20190815-adam_l4-f64-dp75/'

#PRED_DATA_DIR = "../imagery/all-imagery/tiles_planet_png_reshaped"
#PRED_DATA_DIR = "../imagery/all-imagery/tiles_planet_png_reshaped_prediction/fix-red-channel/red-channel-fixed/"
PRED_DATA_DIR = "./dense-train-test-split-all/test/"

layers = 4

data_provider = image_util.ImageDataProviderForPrediction(PRED_DATA_DIR + "*[!_border|!_mask].png", #"/*.png",
                                                          data_suffix=".png")


def record_predictions(data_provider):
    size_all = data_provider.whole_batch_size
    startd_idx = 0 # data_provider.file_idx

    batch_x = data_provider(size_all)
    prediction = net.batch_predict(MODEL_PATH + 'model.ckpt-487', batch_x, max_batch=6)

    res = {}
    for i in range(len(data_provider.data_files)):
        res[data_provider.data_files[startd_idx+i]] = prediction[i,...]
#    pickle.dump(res, open('prediction-Planet.p', 'wb'))
#    pickle.dump(res, open(PRED_DATA_DIR + 'prediction-Planet-fixed-red-channels.p', 'wb'))
    pickle.dump(res, open(PRED_DATA_DIR + 'XXX-SRB-validation.p', 'wb'))

# run the batch
net = unet.Unet(layers=layers, features_root=64, channels=3, n_class=2, cost="weighted_cross_entropy")
record_predictions(data_provider)
