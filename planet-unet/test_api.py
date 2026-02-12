#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.abspath("./tf_unet/")
sys.path.append(ROOT_DIR)

from tf_unet import unet

print("Creating U-Net model...")
try:
    net = unet.Unet(layers=4,
                    features_root=64,
                    channels=3,
                    n_class=2,
                    cost="weighted_cross_entropy")
    print("✓ U-Net model created successfully")
    print(f"  - Input shape: {net.x}")
    print(f"  - Output shape: {net.predictor}")
except Exception as e:
    print(f"✗ Error creating network: {e}")
    import traceback
    traceback.print_exc()
