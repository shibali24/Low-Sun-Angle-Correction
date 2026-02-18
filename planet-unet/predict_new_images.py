#!/usr/bin/env python
"""
predict_new_images.py

Implements the paper's tiling pipeline on arbitrary satellite images:

  1. Convert input image to 3-channel RGB
  2. Tile into 650x650 sub-images with 50px overlap
  3. Zero-pad each tile to 768x768 (original tile centered)
  4. Run U-Net on every 768x768 tile
  5. Extract the inner 650x650 from each prediction output
  6. Stitch tiles back using max-pixel value on overlapping regions
  7. Draw contour polygons on the original-resolution image → new_image_polygon/

Supported input formats: PNG, JPG/JPEG, TIF/TIFF
Output: one *_polygons.png per input image at original resolution.

Usage:
    python predict_new_images.py
    python predict_new_images.py --input my_dir/ --output my_out/ --threshold 0.75
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import find_contours
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "tf_unet"))
sys.path.append(ROOT_DIR)
from tf_unet import unet

# ── constants (match paper) ───────────────────────────────────────────────────
MODEL_DIR         = os.environ.get("MODEL_DIR", "./out_paths/out_path_20191008-adam_l4-f64-dp75_Sentinel")
INPUT_DIR         = "new_image"
OUTPUT_DIR        = "new_image_polygon"
TILE_SIZE         = 650    # paper: break mosaic into 650x650 sub-images
OVERLAP           = 50     # paper: 50px overlap in both directions
PAD_SIZE          = 768    # paper: zero-pad to 768x768 before U-Net
DEFAULT_THRESHOLD = 0.75   # paper: pixels >75% probability = iceberg
SUPPORTED_EXT     = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# U-Net valid convolutions shrink 768→676; the 650x650 content sits at offset
# (59,59) in the 768x768 canvas, so in prediction space it starts at 59-46=13
PAD_OFFSET   = (PAD_SIZE - TILE_SIZE) // 2   # = 59  (pixels of zero padding)
UNET_BORDER  = 46                             # = (768 - 676) // 2
PRED_OFFSET  = PAD_OFFSET - UNET_BORDER       # = 13  (where 650x650 starts in 676x676)
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Predict icebergs in arbitrary satellite images")
    p.add_argument("--input",     default=INPUT_DIR,        help="Directory of input images")
    p.add_argument("--output",    default=OUTPUT_DIR,       help="Directory for polygon overlay outputs")
    p.add_argument("--model",     default=MODEL_DIR,        help="Path to trained model checkpoint directory")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Confidence threshold (default: {})".format(DEFAULT_THRESHOLD))
    return p.parse_args()


# ── step 1: load & convert to RGB ────────────────────────────────────────────

def load_rgb(img_path):
    """Load any image and return a float32 RGB numpy array + original PIL image."""
    original = Image.open(img_path)
    rgb = original.convert("RGB")
    return np.array(rgb, dtype=np.float32), original


# ── step 2: tile into 650x650 with 50px overlap ───────────────────────────────

def make_tiles(img_arr):
    """Break (H, W, 3) image into 650x650 tiles with 50px overlap.

    Returns:
        tiles     : list of float32 arrays (650, 650, 3)
        positions : list of (row_start, col_start) in the original image
    """
    H, W = img_arr.shape[:2]
    step = TILE_SIZE - OVERLAP   # = 600

    tiles, positions = [], []
    r = 0
    while r < H:
        r_end = min(r + TILE_SIZE, H)
        r_start = r_end - TILE_SIZE   # keep full tile by shifting start if near edge
        if r_start < 0:
            r_start = 0

        c = 0
        while c < W:
            c_end = min(c + TILE_SIZE, W)
            c_start = c_end - TILE_SIZE
            if c_start < 0:
                c_start = 0

            tile = img_arr[r_start:r_start + TILE_SIZE, c_start:c_start + TILE_SIZE]

            # If the image is smaller than TILE_SIZE in either dim, zero-pad the tile itself
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            tiles.append(tile)
            positions.append((r_start, c_start))

            if c_end >= W:
                break
            c += step

        if r_end >= H:
            break
        r += step

    return tiles, positions


# ── step 3: zero-pad each 650x650 tile to 768x768 ────────────────────────────

def pad_tile(tile):
    """Center a 650x650 tile in a 768x768 zero canvas."""
    canvas = np.zeros((PAD_SIZE, PAD_SIZE, 3), dtype=np.float32)
    canvas[PAD_OFFSET:PAD_OFFSET + TILE_SIZE,
           PAD_OFFSET:PAD_OFFSET + TILE_SIZE] = tile
    return canvas


def normalize(arr):
    """Normalize float32 array to [0, 1] (matches BaseDataProvider._process_data)."""
    arr = np.clip(np.fabs(arr), 0, np.inf)
    arr -= arr.min()
    if arr.max() != 0:
        arr /= arr.max()
    return arr


# ── steps 4 & 5: U-Net predict, extract inner 650x650 ────────────────────────

def run_prediction(padded_tiles, model_dir):
    """Run U-Net on a batch of 768x768 padded tiles.

    Returns:
        List of (650, 650) float32 iceberg-probability arrays — one per tile,
        with the inner 650x650 extracted from the 676x676 U-Net output.
    """
    batch_x = np.stack([normalize(t) for t in padded_tiles], axis=0)

    net = unet.Unet(layers=4, features_root=64, channels=3, n_class=2,
                    cost="weighted_cross_entropy")

    checkpoint = tf.train.latest_checkpoint(model_dir)
    if checkpoint is None:
        raise RuntimeError("No model checkpoint found in: {}".format(model_dir))

    # predictions shape: (N, 676, 676, 2)
    predictions = net.batch_predict(checkpoint, batch_x, max_batch=6)

    # Extract the 650x650 region that corresponds to the original tile content.
    # In 676x676 prediction space, the tile content starts at PRED_OFFSET (=13).
    extracted = []
    for i in range(len(padded_tiles)):
        prob = predictions[i, ..., 1]   # iceberg probability channel
        inner = prob[PRED_OFFSET:PRED_OFFSET + TILE_SIZE,
                     PRED_OFFSET:PRED_OFFSET + TILE_SIZE]
        extracted.append(inner)

    return extracted


# ── step 6: stitch tiles back with max-pixel merge ───────────────────────────

def stitch_tiles(tile_probs, positions, img_shape):
    """Merge prediction tiles into a full-image probability map.

    Overlapping pixels take the maximum predicted probability (paper method).

    Args:
        tile_probs : list of (650, 650) float32 arrays
        positions  : list of (row_start, col_start) matching tile_probs
        img_shape  : (H, W) of the original image

    Returns:
        prob_map : float32 ndarray (H, W) — merged iceberg probabilities
    """
    H, W = img_shape
    prob_map = np.zeros((H, W), dtype=np.float32)

    for prob, (r0, c0) in zip(tile_probs, positions):
        r1 = min(r0 + TILE_SIZE, H)
        c1 = min(c0 + TILE_SIZE, W)
        h  = r1 - r0
        w  = c1 - c0
        prob_map[r0:r1, c0:c1] = np.maximum(prob_map[r0:r1, c0:c1], prob[:h, :w])

    return prob_map


# ── step 7: draw polygon contours on original image ───────────────────────────

def draw_polygons(original_pil, prob_map, threshold):
    """Threshold the probability map, trace contours, draw red outlines."""
    contours = find_contours(prob_map > threshold, fully_connected="high")

    out_img = original_pil.convert("RGB")
    draw    = ImageDraw.Draw(out_img)

    for contour in contours:
        # contour coords are already in original image pixel space
        pts = [(float(c), float(r)) for r, c in contour]
        if len(pts) < 2:
            continue
        draw.line(pts + [pts[0]], fill=(255, 0, 0), width=3)

    return out_img


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        print("Input directory '{}' not found. Create it and add satellite images.".format(args.input))
        sys.exit(1)

    image_files = sorted(
        f for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )

    if not image_files:
        print("No images found in '{}'. Supported: {}".format(
              args.input, ", ".join(sorted(SUPPORTED_EXT))))
        sys.exit(1)

    print("Found {} image(s) in '{}'".format(len(image_files), args.input))
    os.makedirs(args.output, exist_ok=True)

    for fname in image_files:
        fpath = os.path.join(args.input, fname)
        print("\n[{}]".format(fname))

        # Step 1
        img_arr, original_pil = load_rgb(fpath)
        H, W = img_arr.shape[:2]
        orig_mode = Image.open(fpath).mode
        print("  Size: {}x{}  Mode: {}".format(W, H, orig_mode))

        # Step 2
        tiles, positions = make_tiles(img_arr)
        print("  Tiled into {} tiles ({}x{}, {}px overlap)".format(
              len(tiles), TILE_SIZE, TILE_SIZE, OVERLAP))

        # Step 3
        padded_tiles = [pad_tile(t) for t in tiles]

        # Steps 4 & 5
        print("  Running U-Net on {} tiles...".format(len(padded_tiles)))
        tile_probs = run_prediction(padded_tiles, args.model)

        # Step 6
        prob_map = stitch_tiles(tile_probs, positions, (H, W))
        print("  Stitched probability map: {}x{}".format(prob_map.shape[1], prob_map.shape[0]))

        # Step 7
        out_img  = draw_polygons(original_pil, prob_map, args.threshold)
        n_poly   = len(find_contours(prob_map > args.threshold, fully_connected="high"))
        out_name = os.path.splitext(fname)[0] + "_polygons.png"
        out_path = os.path.join(args.output, out_name)
        out_img.save(out_path)
        print("  Saved: {}  ({} polygon(s) detected)".format(out_name, n_poly))

    print("\nDone. Outputs in '{}'".format(args.output))


if __name__ == "__main__":
    main()
