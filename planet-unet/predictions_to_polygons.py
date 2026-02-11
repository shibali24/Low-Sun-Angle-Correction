import os
import re
from PIL import Image

TILES_DIR = "dense-train-test-split-all/test"
OUT_DIR = "reassembled"
os.makedirs(OUT_DIR, exist_ok=True)

# Only real image tiles (not masks/borders)
def is_image_tile(f):
    return f.endswith(".png") and "_mask" not in f and "_border" not in f

# Parse: <prefix>_al_<r>_si_<c>_af_<af>.png
PAT = re.compile(r"^(?P<prefix>.+)_al_(?P<al>\d+)_si_(?P<si>\d+)_af_(?P<af>\d+)\.png$")

files = [f for f in os.listdir(TILES_DIR) if is_image_tile(f)]
parsed = []
for f in files:
    m = PAT.match(f)
    if m:
        parsed.append((m.group("prefix"), int(m.group("al")), int(m.group("si")), int(m.group("af")), f))

if not parsed:
    raise RuntimeError("No files matched the expected naming pattern.")

# Pick ONE scene prefix to mosaic (the first one)
scene_prefix = parsed[0][0]

# Pick ONE af value (the most common af for that scene)
afs = [af for (p, al, si, af, f) in parsed if p == scene_prefix]
af_choice = max(set(afs), key=afs.count)

scene_tiles = [(al, si, f) for (p, al, si, af, f) in parsed if p == scene_prefix and af == af_choice]

print(f"Scene: {scene_prefix}")
print(f"Using af={af_choice}")
print(f"Tiles: {len(scene_tiles)}")

# Determine grid extents
als = [al for al, si, f in scene_tiles]
sis = [si for al, si, f in scene_tiles]
min_al, max_al = min(als), max(als)
min_si, max_si = min(sis), max(sis)

rows = (max_al - min_al + 1)
cols = (max_si - min_si + 1)

# Load one image to get tile size
sample_img = Image.open(os.path.join(TILES_DIR, scene_tiles[0][2])).convert("RGB")
tile_w, tile_h = sample_img.size

# Create canvas
canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (0, 0, 0))

# Paste each tile into its grid position
for al, si, f in scene_tiles:
    img = Image.open(os.path.join(TILES_DIR, f)).convert("RGB")
    r = al - min_al
    c = si - min_si
    canvas.paste(img, (c * tile_w, r * tile_h))

out_path = os.path.join(OUT_DIR, f"{scene_prefix}_af{af_choice}_mosaic.png")
canvas.save(out_path)
print(f"Saved: {out_path}")
