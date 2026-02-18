# Running planet-unet on Bowdoin HPC (moosehead)

Tested February 2026. Cluster: moosehead.bowdoin.edu, Rocky Linux 9, CUDA 12.x, RTX 3080.

> **Important:** `moosehead.bowdoin.edu` is a login node only — it has no compute resources.
> Do **not** run programs directly on it. All heavy work (cloning, pip install, training) must
> run on a compute node via `srun` (interactive) or `sbatch` (batch job).
> For interactive compute work you can also use `dover.bowdoin.edu` or `foxcroft.bowdoin.edu`.

---

## What this pipeline does

Trains a U-Net on Sentinel satellite imagery to detect icebergs, then produces:

1. **Trained model** — saved to `out_paths/out_path_20191008-adam_l4-f64-dp75_Sentinel/`
2. **Prediction pickle** — `Sentinel-dense-train-test-split-all/test/XXX-SRB-validation.p`
3. **GeoJSON** — `iceberg_polygons.geojson` (iceberg outlines as polygons)
4. **Overlay tiles** — `overlay_tiles/*.png` (test images with red polygon outlines drawn on)

---

## Prerequisites

- Bowdoin HPC account with access to `moosehead.bowdoin.edu`
- SSH key set up for moosehead (same keypair as hopper if you use that)
- A GitHub personal access token for pushing results back (scope: `repo`)
  - Generate at: https://github.com/settings/tokens

---

## One-time setup

### Step 1 — SSH into moosehead

```bash
ssh <your-username>@moosehead.bowdoin.edu
```

### Step 2 — Get a compute node shell (required for heavy operations)

The headnode has memory limits that prevent git clone and pip install.
**All setup must run on a compute node:**

```bash
srun --mem=8G --pty /bin/bash
```

You will land on a node like `moose19`. Your home directory is shared, so files
created here are visible everywhere.

### Step 3 — Clone the repository

```bash
git clone --depth=1 https://github.com/shibali24/Low-Sun-Angle-Correction.git ~/Low-Sun-Angle-Correction
cd ~/Low-Sun-Angle-Correction
git fetch --depth=1 origin new_HPC_run
git checkout -b new_HPC_run FETCH_HEAD
```

> Use `--depth=1` — the full history is too large for the headnode to map into memory.

### Step 4 — Accept conda Terms of Service (one-time, can do on headnode)

```bash
module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Step 5 — Create the conda environment (stay on compute node)

```bash
module load miniconda3
conda init bash
source ~/.bashrc

conda create -n planet-unet python=3.10 -y
conda activate planet-unet

cd ~/Low-Sun-Angle-Correction/planet-unet
pip install -r requirements-modern.txt
```

This will take several minutes — TensorFlow 2.20 with bundled CUDA is ~2 GB.

```bash
exit    # back to headnode
```

---

## Running the pipeline

From the **headnode** (submitting a job is lightweight):

```bash
ssh <your-username>@moosehead.bowdoin.edu

cd ~/Low-Sun-Angle-Correction/planet-unet
mkdir -p logs
sbatch slurm_pipeline.sh
```

Note the job ID that prints (e.g. `Submitted batch job 35499`).

### Monitoring

```bash
squeue -u <your-username>          # check if job is running (R) or pending (PD)
tail -f logs/planet-unet-35499.out  # watch live output (replace 35499)
cat logs/planet-unet-35499.err      # check for errors if job fails
```

The log will be **mostly silent during training** — this is normal. The pipeline
captures subprocess output and only prints stage-level messages:

```
=== HPC Pipeline ===
TRAIN_DATA_DIR: ./Sentinel-dense-train-test-split-all/train
...
Training model...
Training done
Running predictions...
Prediction done
GeoJSON polygons: 503
Overlay tiles: 24
Pipeline complete.
End: Tue Feb 17 08:06:11 PM EST 2026
```

Training + prediction takes roughly **30–60 minutes** on an RTX 3080.
You can close your laptop — the job keeps running on the cluster.

### Pipeline settings (in slurm_pipeline.sh)

| Variable | Default | Meaning |
|---|---|---|
| `FORCE_TRAIN` | `1` | Always retrain even if model exists |
| `SKIP_TRAIN` | `0` | Set to `1` to skip training entirely |
| `SKIP_PRED` | `0` | Set to `1` to skip prediction (reuse existing .p file) |
| `EPOCHS` | `10` | Training epochs |
| `TRAINING_ITERS` | `60` | Iterations per epoch |
| `CONFIDENCE_THRESHOLD` | `0.60` | Minimum probability to count as iceberg |

To re-run only GeoJSON + overlays (fastest, reuses existing prediction):
```bash
# Edit slurm_pipeline.sh on moosehead:
sed -i 's/FORCE_TRAIN=1/FORCE_TRAIN=0/' slurm_pipeline.sh
# Add SKIP_PRED=1 line:
sed -i '/FORCE_TRAIN/a export SKIP_PRED=1' slurm_pipeline.sh
sbatch slurm_pipeline.sh
```

---

## Pulling updates from GitHub

All git operations that touch the packfile must run on a compute node:

```bash
srun --mem=8G --pty /bin/bash
cd ~/Low-Sun-Angle-Correction
git restore .          # discard any local changes first
git fetch --depth=1 origin new_HPC_run
git reset --hard origin/new_HPC_run
exit
```

> Do **not** run `git pull` on the headnode — it will fail with "Cannot allocate memory".

---

## Pushing results back to GitHub

After the job completes, push from a **compute node**:

```bash
srun --mem=8G --pty /bin/bash
cd ~/Low-Sun-Angle-Correction

git add planet-unet/overlay_tiles/
git add planet-unet/iceberg_polygons.geojson
git add planet-unet/out_paths/
git add planet-unet/Sentinel-dense-train-test-split-all/test/XXX-SRB-validation.p
git commit -m "Add HPC results: overlay tiles, model, GeoJSON"
git push --set-upstream origin new_HPC_run --force
exit
```

You will be prompted for your GitHub username and password.
**Use your personal access token as the password** (not your GitHub password).

---

## Key technical notes

- **TF 1.x → TF 2.x migration**: The codebase originally used TF 1.x APIs (`tf.Session`,
  `tf.placeholder`). Moosehead only has CUDA 12.x which is incompatible with TF 1.15.
  All Python files now use `import tensorflow.compat.v1 as tf` + `tf.disable_v2_behavior()`.

- **Bundled CUDA**: `tensorflow[and-cuda]==2.20.0` ships its own CUDA libraries.
  Do **not** `module load cuda` — it will conflict.

- **GPU partition CPU limit**: Max 4 CPUs per user (`QOSMaxCpuPerUserL`).
  The script uses `--cpus-per-task=4`. If you see a job stuck with reason
  `QOSMaxCpuPerUserL`, check that value.

- **conda activate in SLURM**: `conda activate` requires shell functions that only
  exist in interactive sessions. The SLURM script bypasses this with:
  `export PATH="$HOME/.conda/envs/planet-unet/bin:$PATH"`

- **Headnode memory limits**: The headnode cannot map large git packfiles (~780 MB).
  Always run `git clone`, `git pull`, `git push`, and `pip install` from a compute
  node via `srun --mem=8G --pty /bin/bash`.

- **U-Net coordinate offset**: The U-Net valid convolutions shrink the output from
  768×768 → 676×676 pixels (46px border removed on each side). The pipeline
  automatically corrects for this when drawing polygon overlays on the original images.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Job immediately ends, empty log | conda PATH issue | Check that `slurm_pipeline.sh` has `export PATH="$HOME/.conda/envs/planet-unet/bin:$PATH"` |
| Job stuck in PD with `QOSMaxCpuPerUserL` | Too many CPUs requested | Ensure `--cpus-per-task=4` in slurm_pipeline.sh |
| `git pull` fails with "Cannot allocate memory" | Running on headnode | Use `srun --mem=8G --pty /bin/bash` first |
| `git push` rejected (non-fast-forward) | Remote has newer commits | Use `--force` flag (from compute node) |
| `CondaError: Run 'conda init' before 'conda activate'` | conda activate in batch script | Use PATH export instead (already fixed in slurm_pipeline.sh) |
| Polygon outlines visually offset from icebergs | Missing U-Net border offset | Check `run_pipeline_hpc.py` computes `row_off`/`col_off` from image vs pred size |
