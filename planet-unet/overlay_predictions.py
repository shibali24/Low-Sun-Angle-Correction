import os
import pickle
import json
import numpy as np
from PIL import Image, ImageDraw

# -------- CONFIG --------
TILES_DIR = "dense-train-test-split-all/test"
PRED_PKL = "dense-train-test-split-all/test/XXX-SRB-validation.p"
POLY_GEOJSON = "iceberg_polygons.geojson"
OUT_DIR = "overlay_tiles"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- LOAD DATA --------
preds = pickle.load(open(PRED_PKL, "rb"))
geo = json.load(open(POLY_GEOJSON))

# Group polygons by image filename
polys_by_image = {}
for feat in geo["features"]:
    fname = os.path.basename(feat["properties"]["image"])
    polys_by_image.setdefault(fname, []).append(feat["geometry"]["coordinates"][0])

print(f"Loaded polygons for {len(polys_by_image)} tiles")

# -------- DRAW POLYGONS --------
for fname, polys in polys_by_image.items():
    img_path = os.path.join(TILES_DIR, fname)
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for poly in polys:
        # poly = [(x,y), (x,y), ...]
        draw.line(poly + [poly[0]], fill=(255, 0, 0), width=2)

    img.save(os.path.join(OUT_DIR, fname))

print(f"Saved overlay tiles to {OUT_DIR}/")
