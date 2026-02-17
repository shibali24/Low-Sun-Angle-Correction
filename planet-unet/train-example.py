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

ROOT_DIR = os.path.abspath("./tf_unet/")
sys.path.append(ROOT_DIR)

from tf_unet import unet, util, image_util

#MODEL_DIR = './out_paths/out_path_20190815-adam_l4-f64-dp75'
MODEL_DIR = os.environ.get("MODEL_DIR", "./out_paths/out_path_20191008-adam_l4-f64-dp75_Sentinel")
TRAIN_DATA_DIR = os.environ.get("TRAIN_DATA_DIR", "./Sentinel-dense-train-test-split-all/train")
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "./Sentinel-dense-train-test-split-all/test")

RESTORE = os.environ.get("RESTORE", "0") == "1"
RESTORE_PATH = os.path.abspath(os.environ.get("RESTORE_PATH", "./out_paths/out_path_20190815-adam_l4-f64-dp75"))
TRAINING_ITERS = int(os.environ.get("TRAINING_ITERS", "100"))
EPOCHS = int(os.environ.get("EPOCHS", "50"))

train_data_provider = image_util.ImageDataProviderWithWeights(TRAIN_DATA_DIR + "/*.png",
                                                              data_suffix=".png",
                                                              mask_suffix="_mask.png",
                                                              border_suffix="_border.png"
                                                             )

test_data_provider = image_util.ImageDataProviderWithWeights(TEST_DATA_DIR + "/*.png",
                                                              data_suffix=".png",
                                                              mask_suffix="_mask.png",
                                                              border_suffix="_border.png"
                                                             )

weights = None

#setup & training
net = unet.Unet(layers=4,
                features_root=64,
                channels=3,
                n_class=2,
                cost = "weighted_cross_entropy"
#                cost = "cross_entropy",
#                cost_kwargs=dict(class_weights=[.1,.2]),
               )

trainer = unet.Trainer(net, 
                       batch_size=2,
                       validation_batch_size=2,
                       optimizer="adam",
#                       opt_kwargs = dict(learning_rate=.0001,
#                                         decay_rate=0.95)
                       opt_kwargs = dict(learning_rate=.00005)
                      )

path = trainer.train(train_data_provider,
                     test_data_provider,
                     MODEL_DIR,
                     restore=RESTORE,
                     restore_path=RESTORE_PATH,
                     dropout=0.75,
                     training_iters=TRAINING_ITERS,
                     epochs=EPOCHS,
                     display_step=1
                     )
