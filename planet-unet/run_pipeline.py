#!/usr/bin/env python
"""
End-to-end pipeline to detect icebergs and create overlay tiles.

Steps:
1. Train U-Net model (if not already trained)
2. Run batch predictions on test data
3. Generate iceberg_polygons.geojson from predictions
4. Create overlay tiles with polygon outlines
"""

import os
import sys
import json
import pickle
import subprocess
import glob
from pathlib import Path

import numpy as np
import skimage.io
from skimage.measure import find_contours
from PIL import Image, ImageDraw

# Configuration
ROOT_DIR = os.path.abspath("./tf_unet/")
sys.path.append(ROOT_DIR)

MODEL_DIR = './out_paths/out_path_20191008-adam_l4-f64-dp75_Sentinel'
RESTORE_PATH = os.path.abspath("./out_paths/out_path_20190815-adam_l4-f64-dp75")

TRAIN_DATA_DIR = "./Sentinel-dense-train-test-split-all/train"
TEST_DATA_DIR = "./Sentinel-dense-train-test-split-all/test"
TRAINING_ITERS = int(os.environ.get("TRAINING_ITERS", "60"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
FORCE_TRAIN = os.environ.get("FORCE_TRAIN", "0") == "1"
TRAIN_TIMEOUT_SEC = int(os.environ.get("TRAIN_TIMEOUT_SEC", "43200"))

PRED_PKL = os.path.join(TEST_DATA_DIR, "XXX-SRB-validation.p")
POLY_GEOJSON = "iceberg_polygons.geojson"
OVERLAY_DIR = "overlay_tiles"
CONFIDENCE_THRESHOLD = 0.75

print("="*80)
print("ICEBERG DETECTION PIPELINE - END TO END")
print("="*80)

# ============================================================================
# STEP 1: Train Model (if needed)
# ============================================================================
def train_model():
    """Train the U-Net model on training data."""
    print("\n" + "="*80)
    print("STEP 1: Training U-Net Model")
    print("="*80)
    
    if not FORCE_TRAIN and os.path.exists(MODEL_DIR) and len(glob.glob(f"{MODEL_DIR}/*.meta")) > 0:
        print(f"✓ Model already trained at {MODEL_DIR}")
        print("  Skipping training...")
        return True
    
    print("\n⏳ Training U-Net model...")
    print(f"   Training directory: {TRAIN_DATA_DIR}")
    print(f"   Test directory: {TEST_DATA_DIR}")
    print("   Model parameters: layers=4, features_root=64, dropout=0.75")
    print(f"   Training iterations: {TRAINING_ITERS}, Epochs: {EPOCHS}")
    if FORCE_TRAIN:
        print("   Force train: enabled")
    
    try:
        result = subprocess.run(
            [sys.executable, "train-example.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=dict(os.environ, TRAINING_ITERS=str(TRAINING_ITERS), EPOCHS=str(EPOCHS)),
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT_SEC
        )
        
        if result.returncode != 0:
            print(f"✗ Training failed with return code {result.returncode}")
            print("\nSTDOUT:\n", result.stdout[-1000:])
            print("\nSTDERR:\n", result.stderr[-1000:])
            return False
        
        print("✓ Model training completed successfully!")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"✗ Training timed out after {TRAIN_TIMEOUT_SEC} seconds")
        return False
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        return False


# ============================================================================
# STEP 2: Generate Predictions
# ============================================================================
def generate_predictions():
    """Run batch predictions on test data."""
    print("\n" + "="*80)
    print("STEP 2: Running Batch Predictions")
    print("="*80)
    
    if os.path.exists(PRED_PKL):
        print(f"✓ Predictions already exist at {PRED_PKL}")
        return True
    
    print("\n⏳ Running predictions on test data...")
    print(f"   Test data: {TEST_DATA_DIR}")
    print(f"   Output: {PRED_PKL}")
    
    try:
        result = subprocess.run(
            [sys.executable, "batch_prediction.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode != 0:
            print(f"✗ Predictions failed with return code {result.returncode}")
            print("\nSTDOUT:\n", result.stdout[-500:])
            print("\nSTDERR:\n", result.stderr[-500:])
            return False
        
        if os.path.exists(PRED_PKL):
            print("✓ Predictions generated successfully!")
            return True
        else:
            print("✗ Predictions file not found after execution")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Predictions timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"✗ Predictions failed with error: {e}")
        return False


# ============================================================================
# STEP 3: Generate GeoJSON from Predictions
# ============================================================================
def generate_geojson():
    """Convert predictions to iceberg_polygons.geojson."""
    print("\n" + "="*80)
    print("STEP 3: Generating iceberg_polygons.geojson")
    print("="*80)
    
    if os.path.exists(POLY_GEOJSON):
        print(f"✓ GeoJSON already exists: {POLY_GEOJSON}")
        return True
    
    if not os.path.exists(PRED_PKL):
        print(f"✗ Predictions file not found: {PRED_PKL}")
        return False
    
    print(f"\n⏳ Loading predictions from {PRED_PKL}...")
    try:
        preds = pickle.load(open(PRED_PKL, "rb"))
        print(f"   Loaded {len(preds)} predictions")
    except Exception as e:
        print(f"✗ Failed to load predictions: {e}")
        return False
    
    print(f"\n⏳ Extracting iceberg polygons...")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    features = []
    polygon_count = 0
    
    for filename, pred_array in preds.items():
        # Get probability map for iceberg class (class 1)
        iceberg_prob = pred_array[..., 1]
        
        # Find contours where confidence > threshold
        contours = find_contours(iceberg_prob > CONFIDENCE_THRESHOLD, fully_connected='high')
        
        for contour_idx, contour in enumerate(contours):
            # Convert (row, col) to (x, y) coordinates
            # Reverse order and close the polygon
            coordinates = [[float(c), float(r)] for r, c in contour]
            coordinates.append(coordinates[0])  # Close polygon
            
            max_confidence = float(np.max(iceberg_prob[
                int(contour[:, 0].min()):int(contour[:, 0].max())+1,
                int(contour[:, 1].min()):int(contour[:, 1].max())+1
            ]))
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                },
                "properties": {
                    "image": os.path.basename(filename),
                    "iceberg_id": contour_idx,
                    "confidence": max_confidence
                }
            })
            polygon_count += 1
    
    # Write GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    try:
        with open(POLY_GEOJSON, "w") as f:
            json.dump(geojson, f, indent=2)
        
        print(f"✓ Generated {polygon_count} iceberg polygons")
        print(f"✓ Saved to {POLY_GEOJSON}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to write GeoJSON: {e}")
        return False


# ============================================================================
# STEP 4: Create Overlay Tiles
# ============================================================================
def create_overlay_tiles():
    """Draw polygons on images to create overlay tiles."""
    print("\n" + "="*80)
    print("STEP 4: Creating Overlay Tiles with Polygon Outlines")
    print("="*80)
    
    if not os.path.exists(POLY_GEOJSON):
        print(f"✗ GeoJSON file not found: {POLY_GEOJSON}")
        return False
    
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    
    print(f"\n⏳ Loading polygons from {POLY_GEOJSON}...")
    try:
        with open(POLY_GEOJSON, "r") as f:
            geo = json.load(f)
        
        # Group polygons by image filename
        polys_by_image = {}
        for feat in geo["features"]:
            fname = feat["properties"]["image"]
            polys_by_image.setdefault(fname, []).append({
                "coords": feat["geometry"]["coordinates"][0],
                "confidence": feat["properties"].get("confidence", 0)
            })
        
        print(f"   Loaded polygons for {len(polys_by_image)} images")
    except Exception as e:
        print(f"✗ Failed to load GeoJSON: {e}")
        return False
    
    print(f"\n⏳ Drawing polygon overlays on test images...")
    
    overlay_count = 0
    for fname, polys in polys_by_image.items():
        img_path = os.path.join(TEST_DATA_DIR, fname)
        
        if not os.path.exists(img_path):
            print(f"   ⚠ Image not found: {fname}")
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            for poly_data in polys:
                poly = poly_data["coords"]
                confidence = poly_data["confidence"]
                
                # Draw polygon outline (red with confidence-based thickness)
                thickness = max(1, int(confidence * 3))  # 1-3 pixels based on confidence
                
                # Draw line segments
                for i in range(len(poly) - 1):
                    x1, y1 = poly[i]
                    x2, y2 = poly[i+1]
                    draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=thickness)
            
            # Save overlay image
            out_path = os.path.join(OVERLAY_DIR, fname)
            img.save(out_path)
            overlay_count += 1
            print(f"   ✓ {fname} ({len(polys)} icebergs)")
            
        except Exception as e:
            print(f"   ✗ Failed to process {fname}: {e}")
    
    print(f"\n✓ Created {overlay_count} overlay tiles")
    print(f"✓ Saved to {OVERLAY_DIR}/")
    return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Run the complete pipeline."""
    
    steps = [
        ("Train Model", train_model),
        ("Generate Predictions", generate_predictions),
        ("Generate GeoJSON", generate_geojson),
        ("Create Overlays", create_overlay_tiles),
    ]
    
    results = {}
    for step_name, step_func in steps:
        try:
            success = step_func()
            results[step_name] = "✓ SUCCESS" if success else "✗ FAILED"
            
            if not success:
                print(f"\n❌ Pipeline stopped at: {step_name}")
                break
                
        except Exception as e:
            print(f"\n✗ Unexpected error in {step_name}: {e}")
            results[step_name] = "✗ ERROR"
            break
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    for step_name, result in results.items():
        print(f"{step_name:.<40} {result}")
    
    print("\n" + "="*80)
    all_success = all("SUCCESS" in r for r in results.values())
    
    if all_success:
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"\n📍 Output files:")
        print(f"   - Trained model: {MODEL_DIR}/")
        print(f"   - Predictions: {PRED_PKL}")
        print(f"   - Polygons: {POLY_GEOJSON}")
        print(f"   - Overlay tiles: {OVERLAY_DIR}/")
        return 0
    else:
        print("✗ PIPELINE FAILED - See details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
