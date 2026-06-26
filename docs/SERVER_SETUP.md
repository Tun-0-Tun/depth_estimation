# Server setup: data, training, evaluation (rel2metric)

End-to-end recipe to train the amortized **rel→metric** projection on a GPU server
and evaluate it on ZJU-L5 against the existing baselines.

## 1. Clone + environment

```bash
git clone https://github.com/Tun-0-Tun/depth_estimation.git
cd depth_estimation
git checkout feature/rel2metric-cross-domain      # branch with the new model

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .            # installs deps from pyproject (torch, transformers, h5py, pandas, pyarrow, ...)
```

On a CUDA box install the matching PyTorch build first if `pip install -e .` pulls a
CPU-only wheel, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

The Depth-Anything-V2 backbone is fetched automatically from Hugging Face on first
run (set `HF_HOME` to a large disk if needed).

## 2. Download the datasets

All datasets live under `data/` (git-ignored). NYU and KITTI download automatically:

```bash
python scripts/download_data.py --all          # NYU (~2.8 GB) + KITTI raw-depth (~0.5 GB)
python scripts/download_data.py --zju-info      # prints ZJU-L5 instructions
```

| Dataset | How | Size | Used for |
|---|---|---|---|
| **NYU labeled** | auto (public URL) | 2.8 GB | indoor train/val |
| **KITTI raw-depth** | auto (HF `WyettZ/kitti-raw-depth`) | 0.5 GB | outdoor train/val (cross-domain) |
| **ZJU-L5** | **manual** (DELTAR, gated) | 2.1 GB | L5 dToF train + final eval |

ZJU-L5 is gated by the DELTAR authors — fetch it from
<https://github.com/zju3dv/DELTAR> and extract so the repo sees
`data/ZJUL5/data.json` plus `data/ZJUL5/<scene>/<ts>.h5` (483 train / 527 test).

Expected final layout:

```
data/
  nyu_depth_v2/nyu_depth_v2_labeled.mat
  kitti_raw_depth/data/*.parquet
  ZJUL5/data.json + <scene>/*.h5
```

Sanity check:

```bash
python -c "from depth_estimation.data.unified import build_samples as B; \
print({s:len(B({'nyu_mat':'data/nyu_depth_v2/nyu_depth_v2_labeled.mat',\
'kitti_root':'data/kitti_raw_depth','zju_l5_root':'data/ZJUL5'}, s)) for s in ['train','val']})"
```

## 3. Train (amortized pretraining on the mix)

```bash
python scripts/train_rel2metric.py --config configs/train_rel2metric_mixed.json
```

- Trains `RelToMetricCNN` on NYU + KITTI + ZJU-L5 with **randomized sparse priors**
  (so it generalizes across sensor patterns instead of memorizing the L5 8×8 layout).
- Caches per-frame Depth-Anything output under `outputs/rel2metric/drel_cache/`
  (first epoch is slow, later epochs reuse the cache).
- Saves the best checkpoint to `outputs/rel2metric/rel2metric_mixed.pt`.

Key config knobs (`configs/train_rel2metric_mixed.json`):

| key | meaning |
|---|---|
| `epochs`, `lr`, `hidden`, `num_layers` | optimizer / capacity |
| `max_per_dataset` | cap per dataset (`null` = use all) for quick runs |
| `anchor_lambda`, `smooth_lambda` | sparse-anchor vs field-smoothness weights |
| `prior_noise` | simulated dToF noise (rel/abs std, outliers) |
| `device` | `null` = auto (CUDA→MPS→CPU), or `"cuda:0"` |

## 4. Evaluate on ZJU-L5 (vs baselines)

```bash
python scripts/run_experiment.py --config configs/exp_zju_l5_rel2metric_eval.json
```

Runs `rel2metric_tta` (with per-image test-time adaptation), `rel2metric_no_tta`,
`direct_depth_cnn`, and `local_bilateral` on the ZJU-L5 test split with the real L5
zone prior. Metrics + comparison figures land in
`outputs/experiments/zju_l5_rel2metric_eval/`.

The `rel2metric_tta` method loads `outputs/rel2metric/rel2metric_mixed.pt` and
fine-tunes a few steps on each frame's sparse anchors before predicting — this is
the "learn the rel→metric projection per image" stage.

## 5. Cross-domain check (optional, for the paper)

Train on NYU+KITTI only (drop `zju_l5_root` from the train config) and evaluate
zero-shot on ZJU-L5: if metrics stay close to the in-domain run, the learned
projection generalizes rather than memorizing one dataset.
