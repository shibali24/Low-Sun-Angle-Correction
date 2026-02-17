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
from PIL import Image
import pickle
from scipy import ndimage

ROOT_DIR = os.path.abspath("./tf_unet/")
sys.path.append(ROOT_DIR)

from tf_unet import unet, util, image_util

MODEL_DIR = os.environ.get("MODEL_DIR", "./out_paths/out_path_20191008-adam_l4-f64-dp75_Sentinel")

#PRED_DATA_DIR = "../imagery/all-imagery/tiles_planet_png_reshaped"
#PRED_DATA_DIR = "../imagery/all-imagery/tiles_planet_png_reshaped_prediction/fix-red-channel/red-channel-fixed/"
PRED_DATA_DIR = os.environ.get("PRED_DATA_DIR", "./Sentinel-dense-train-test-split-all/test/")

layers = 4

data_provider = image_util.ImageDataProviderForPrediction(
    os.path.join(PRED_DATA_DIR, "*.png"),
    data_suffix=".png",
    shuffle_data=False,
)
data_provider.data_files = [
    f for f in data_provider.data_files
    if not f.endswith("_mask.png") and not f.endswith("_border.png")
]
data_provider.whole_batch_size = len(data_provider.data_files)
if data_provider.whole_batch_size == 0:
    raise RuntimeError("No prediction tiles found in {}".format(PRED_DATA_DIR))


def record_predictions(data_provider):
    size_all = data_provider.whole_batch_size
    startd_idx = 0 # data_provider.file_idx

    checkpoint = tf.train.latest_checkpoint(MODEL_DIR)
    if checkpoint is None:
        raise RuntimeError("No model checkpoint found in {}".format(MODEL_DIR))

    batch_x = data_provider(size_all)
    prediction = net.batch_predict(checkpoint, batch_x, max_batch=6)

    res = {}
    for i in range(len(data_provider.data_files)):
        res[data_provider.data_files[startd_idx+i]] = prediction[i,...]
#    pickle.dump(res, open('prediction-Planet.p', 'wb'))
#    pickle.dump(res, open(PRED_DATA_DIR + 'prediction-Planet-fixed-red-channels.p', 'wb'))
    pickle.dump(res, open(os.path.join(PRED_DATA_DIR, 'XXX-SRB-validation.p'), 'wb'))

# run the batch
net = unet.Unet(layers=layers, features_root=64, channels=3, n_class=2, cost="weighted_cross_entropy")
record_predictions(data_provider)
