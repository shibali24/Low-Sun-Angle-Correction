import os
import re
import numpy as np
from PIL import Image
from collections import defaultdict

# -------- CONFIG --------
TILES_DIR = "dense-train-test-split-all/test"
OUT_DIR = "reassembled"
os.makedirs(OUT_DIR, exist_ok=True)

# Only use real images (not masks or borders)
def is_image_tile(fname):
    return (
        fname.endswith(".png")
        and "_mask" not in fname
        and "_border" not in fname
    )

# -------- LOAD FILES --------
files = [f for f in os.listdir(TILES_DIR) if is_image_tile(f)]

# Group by scene prefix (before `_al_`)
scenes = defaultdict(list)
for f in files:
    prefix = f.split("_al_")[0]
    scenes[prefix].append(f)

print(f"Found {len(scenes)} scenes")

# -------- REASSEMBLE EACH SCENE --------
for scene_id, scene_files in scenes.items():
    print(f"Reassembling scene: {scene_id} ({len(scene_files)} tiles)")

    # Load images
    imgs = []
    for f in scene_files:
        img = Image.open(os.path.join(TILES_DIR, f))
        imgs.append((f, img))

    # Assume all tiles same size
    tile_w, tile_h = imgs[0][1].size

    # Sort deterministically by filename
    imgs.sort(key=lambda x: x[0])

    # Infer grid size (best guess: square-ish)
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    canvas = Image.new(
        imgs[0][1].mode,
        (cols * tile_w, rows * tile_h)
    )

    for idx, (_, img) in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        canvas.paste(img, (c * tile_w, r * tile_h))

    out_path = os.path.join(OUT_DIR, f"{scene_id}_reassembled.png")
    canvas.save(out_path)
    print(f"Saved {out_path}")
