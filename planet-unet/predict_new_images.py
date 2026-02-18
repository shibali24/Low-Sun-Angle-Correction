#!/usr/bin/env python
"""
predict_new_images.py

Takes arbitrary satellite images from new_image/, converts each to 3-channel
RGB and resizes to 768x768 for the U-Net, runs iceberg prediction, then scales
the detected polygon contours back to the original image resolution and draws
them as red outlines saved to new_image_polygon/.

Supported input formats: PNG, JPG/JPEG, TIF/TIFF
Output: one *_polygons.png per input image, at the original resolution.

Usage:
    python predict_new_images.py
    python predict_new_images.py --input my_dir/ --output my_out/ --threshold 0.6
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

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL_DIR           = os.environ.get("MODEL_DIR", "./out_paths/out_path_20191008-adam_l4-f64-dp75_Sentinel")
INPUT_DIR           = "new_image"
OUTPUT_DIR          = "new_image_polygon"
TARGET_SIZE         = (768, 768)          # model input resolution (H, W)
DEFAULT_THRESHOLD   = 0.60
SUPPORTED_EXT       = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Predict icebergs in arbitrary satellite images")
    p.add_argument("--input",     default=INPUT_DIR,          help="Directory of input images")
    p.add_argument("--output",    default=OUTPUT_DIR,         help="Directory for polygon overlay outputs")
    p.add_argument("--model",     default=MODEL_DIR,          help="Path to trained model checkpoint directory")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Confidence threshold for iceberg detection (default: {})".format(DEFAULT_THRESHOLD))
    return p.parse_args()


# ── image preprocessing ───────────────────────────────────────────────────────

def load_and_prepare(img_path):
    """Load any image, convert to 3-channel RGB, resize to TARGET_SIZE.

    Returns:
        resized_array : float32 ndarray (768, 768, 3) — raw pixel values
        original      : PIL.Image at original resolution (any mode)
    """
    original = Image.open(img_path)

    # Convert to plain RGB (handles grayscale, RGBA, palette, CMYK, etc.)
    rgb = original.convert("RGB")

    resized = rgb.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.LANCZOS)
    return np.array(resized, dtype=np.float32), original


def normalize(arr):
    """Normalize a float32 image array to [0, 1] (same as BaseDataProvider)."""
    arr = np.clip(np.fabs(arr), 0, np.inf)
    arr -= arr.min()
    if arr.max() != 0:
        arr /= arr.max()
    return arr


# ── prediction ────────────────────────────────────────────────────────────────

def run_prediction(image_arrays, model_dir):
    """Run U-Net on a list of (768, 768, 3) float32 arrays.

    Returns:
        predictions : ndarray (N, out_h, out_w, 2)  — channel 1 = iceberg prob
    """
    batch_x = np.stack([normalize(a) for a in image_arrays], axis=0)

    net = unet.Unet(layers=4, features_root=64, channels=3, n_class=2,
                    cost="weighted_cross_entropy")

    checkpoint = tf.train.latest_checkpoint(model_dir)
    if checkpoint is None:
        raise RuntimeError("No model checkpoint found in: {}".format(model_dir))

    predictions = net.batch_predict(checkpoint, batch_x, max_batch=6)
    return predictions


# ── polygon drawing ───────────────────────────────────────────────────────────

def draw_polygons(original_pil, pred_map, threshold):
    """Find contours in pred_map, scale to original resolution, draw as red
    outlines on the original image.

    The U-Net shrinks input from 768×768 → pred_map.shape (e.g. 676×676) via
    valid convolutions.  Each prediction pixel therefore aligns with the center
    region of the 768×768 input, offset by `border` pixels on every side.
    We undo that offset, then scale from 768×768 to the original resolution.

    Args:
        original_pil : PIL.Image at original resolution
        pred_map     : float32 ndarray (out_h, out_w) — iceberg probabilities
        threshold    : float — minimum probability to classify as iceberg

    Returns:
        PIL.Image with red polygon outlines drawn at original resolution
    """
    orig_w, orig_h = original_pil.size                  # PIL: (width, height)
    out_h, out_w   = pred_map.shape

    # Border removed by valid convolutions on each side
    border_y = (TARGET_SIZE[0] - out_h) // 2           # e.g. (768 - 676) // 2 = 46
    border_x = (TARGET_SIZE[1] - out_w) // 2

    # Scale from 768×768 → original
    scale_x = orig_w / TARGET_SIZE[1]
    scale_y = orig_h / TARGET_SIZE[0]

    contours = find_contours(pred_map > threshold, fully_connected="high")

    out_img = original_pil.convert("RGB")
    draw    = ImageDraw.Draw(out_img)

    for contour in contours:
        # contour: array of (row, col) in prediction space
        # step 1 — shift into 768×768 space
        # step 2 — scale to original resolution
        pts = [
            ((c + border_x) * scale_x, (r + border_y) * scale_y)
            for r, c in contour
        ]
        if len(pts) < 2:
            continue
        draw.line(pts + [pts[0]], fill=(255, 0, 0), width=3)

    return out_img


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        print("Input directory '{}' not found.\n"
              "Create it and add satellite images.".format(args.input))
        sys.exit(1)

    image_files = sorted(
        f for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )

    if not image_files:
        print("No images found in '{}'. Supported formats: {}".format(
              args.input, ", ".join(sorted(SUPPORTED_EXT))))
        sys.exit(1)

    print("Found {} image(s) in '{}'".format(len(image_files), args.input))
    os.makedirs(args.output, exist_ok=True)

    # ── preprocess ────────────────────────────────────────────────────────────
    arrays, originals, names = [], [], []
    for fname in image_files:
        fpath = os.path.join(args.input, fname)
        arr, orig = load_and_prepare(fpath)
        mode_note = "" if Image.open(fpath).mode == "RGB" else \
                    " (converted from {})".format(Image.open(fpath).mode)
        print("  Loaded {:30s}  {}x{}{}  →  {}x{} RGB".format(
              fname,
              orig.size[0], orig.size[1], mode_note,
              TARGET_SIZE[1], TARGET_SIZE[0]))
        arrays.append(arr)
        originals.append(orig)
        names.append(fname)

    # ── predict ───────────────────────────────────────────────────────────────
    print("\nRunning U-Net prediction on {} image(s)...".format(len(arrays)))
    predictions = run_prediction(arrays, args.model)
    print("Prediction done. Output shape: {}".format(predictions.shape))

    # ── draw & save ───────────────────────────────────────────────────────────
    print("\nDrawing polygons and saving to '{}'...".format(args.output))
    for i, (fname, orig) in enumerate(zip(names, originals)):
        pred_map = predictions[i, ..., 1]           # iceberg probability map
        out_img  = draw_polygons(orig, pred_map, args.threshold)

        out_name = os.path.splitext(fname)[0] + "_polygons.png"
        out_path = os.path.join(args.output, out_name)
        out_img.save(out_path)

        n_poly = len(find_contours(pred_map > args.threshold, fully_connected="high"))
        print("  {} → {}  ({} polygon(s) detected)".format(fname, out_name, n_poly))

    print("\nDone.")


if __name__ == "__main__":
    main()
