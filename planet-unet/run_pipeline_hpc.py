#!/usr/bin/env python
"""
HPC-friendly end-to-end pipeline:
1) train
2) batch prediction
3) polygons GeoJSON
4) overlay tiles
"""

import json
import os
import pickle
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import find_contours

MODEL_DIR = os.environ.get("MODEL_DIR", "./out_paths/out_path_20191008-adam_l4-f64-dp75_Sentinel")
TRAIN_DATA_DIR = os.environ.get("TRAIN_DATA_DIR", "./Sentinel-dense-train-test-split-all/train")
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "./Sentinel-dense-train-test-split-all/test")
PRED_PKL = os.path.join(TEST_DATA_DIR, "XXX-SRB-validation.p")
POLY_GEOJSON = os.environ.get("POLY_GEOJSON", "iceberg_polygons.geojson")
OVERLAY_DIR = os.environ.get("OVERLAY_DIR", "overlay_tiles")

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))
TRAIN_TIMEOUT_SEC = int(os.environ.get("TRAIN_TIMEOUT_SEC", "43200"))
PRED_TIMEOUT_SEC = int(os.environ.get("PRED_TIMEOUT_SEC", "7200"))
FORCE_TRAIN = os.environ.get("FORCE_TRAIN", "0") == "1"
SKIP_TRAIN = os.environ.get("SKIP_TRAIN", "0") == "1"
SKIP_PRED = os.environ.get("SKIP_PRED", "0") == "1"


def run_script(script_name, timeout_sec, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        return False
    return True


def train_model():
    if SKIP_TRAIN:
        print("Skipping training (SKIP_TRAIN=1)")
        return True
    has_model = os.path.exists(MODEL_DIR) and any(x.endswith(".meta") for x in os.listdir(MODEL_DIR))
    if has_model and not FORCE_TRAIN:
        print("Model exists, skipping train. Set FORCE_TRAIN=1 to retrain.")
        return True
    print("Training model...")
    ok = run_script("train-example.py", TRAIN_TIMEOUT_SEC)
    print("Training {}".format("done" if ok else "failed"))
    return ok


def predict():
    if SKIP_PRED:
        print("Skipping prediction (SKIP_PRED=1)")
        return True
    if os.path.exists(PRED_PKL):
        os.remove(PRED_PKL)
    print("Running predictions...")
    ok = run_script("batch_prediction.py", PRED_TIMEOUT_SEC, extra_env={"PRED_DATA_DIR": TEST_DATA_DIR})
    print("Prediction {}".format("done" if ok else "failed"))
    return ok and os.path.exists(PRED_PKL)


def build_geojson():
    if not os.path.exists(PRED_PKL):
        print("Missing prediction file:", PRED_PKL)
        return False

    preds = pickle.load(open(PRED_PKL, "rb"))
    features = []
    for filename, pred_array in preds.items():
        iceberg_prob = pred_array[..., 1]

        # U-Net valid convolutions produce output smaller than input.
        # Compute offset so contour coords align with the original image pixels.
        row_off = col_off = 0
        img_path = os.path.join(TEST_DATA_DIR, os.path.basename(filename))
        if os.path.exists(img_path):
            img_w, img_h = Image.open(img_path).size
            row_off = (img_h - iceberg_prob.shape[0]) // 2
            col_off = (img_w - iceberg_prob.shape[1]) // 2

        contours = find_contours(iceberg_prob > CONFIDENCE_THRESHOLD, fully_connected="high")
        for contour_idx, contour in enumerate(contours):
            coords = [[float(c) + col_off, float(r) + row_off] for r, c in contour]
            if not coords:
                continue
            coords.append(coords[0])
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {
                        "image": os.path.basename(filename),
                        "iceberg_id": contour_idx,
                        "confidence": float(np.max(iceberg_prob)),
                    },
                }
            )

    geo = {"type": "FeatureCollection", "features": features}
    with open(POLY_GEOJSON, "w") as f:
        json.dump(geo, f)
    print("GeoJSON polygons:", len(features))
    return True


def draw_overlays():
    if not os.path.exists(POLY_GEOJSON):
        print("Missing GeoJSON:", POLY_GEOJSON)
        return False

    os.makedirs(OVERLAY_DIR, exist_ok=True)
    geo = json.load(open(POLY_GEOJSON, "r"))
    grouped = {}
    for feat in geo.get("features", []):
        fname = feat["properties"]["image"]
        grouped.setdefault(fname, []).append(feat["geometry"]["coordinates"][0])

    count = 0
    for fname, polygons in grouped.items():
        img_path = os.path.join(TEST_DATA_DIR, fname)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for poly in polygons:
            draw.line(poly + [poly[0]], fill=(255, 0, 0), width=2)
        img.save(os.path.join(OVERLAY_DIR, fname))
        count += 1
    print("Overlay tiles:", count)
    return True


def main():
    print("=== HPC Pipeline ===")
    print("TRAIN_DATA_DIR:", TRAIN_DATA_DIR)
    print("TEST_DATA_DIR:", TEST_DATA_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("CONFIDENCE_THRESHOLD:", CONFIDENCE_THRESHOLD)

    if not train_model():
        return 1
    if not predict():
        return 1
    if not build_geojson():
        return 1
    if not draw_overlays():
        return 1
    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
