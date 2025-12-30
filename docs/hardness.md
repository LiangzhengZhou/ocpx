# Crystal hardness training & prediction (all model examples)

This guide shows how to use the existing OCP training stack (GemNet + `EnergyTrainer`) to
train and predict crystal hardness from CIF structures. It focuses on the end-to-end flow:
data formatting, preprocessing, configuration, training, and inference outputs.

## 1) Prepare a CSV file

Create a CSV with at least the following columns:

```csv
cif_path,hardness
structures/Fe2O3.cif,12.3
structures/TiO2.cif,14.8
```

Optional columns:
- `sample_id` (or any column specified via `--id-column`) is stored as `sample_id` in the LMDB.

Recommended conventions:
- Use consistent hardness units (e.g., GPa). The model treats the target as a scalar regression
  label, so mixing units will harm training.
- Use relative paths in `cif_path` when possible, then provide the common root via `--cif-root`.
- Make sure CIFs are fully reduced and include all required lattice information; failures in
  parsing show up during preprocessing.

## 2) Build LMDB datasets

Use the preprocessing script to convert CIF files into LMDBs. Run it separately
for train/val/test splits:

```bash
python scripts/preprocess_hardness.py \
  --csv data/hardness/train.csv \
  --out-path data/hardness/train \
  --cif-root data/hardness/cifs \
  --get-edges

python scripts/preprocess_hardness.py \
  --csv data/hardness/val.csv \
  --out-path data/hardness/val \
  --cif-root data/hardness/cifs \
  --get-edges
```

This command writes:
- `data.lmdb` with graph data
- `metadata.npz` with atom/neighbor counts (for load balancing)
- `target_stats.json` with `target_mean` and `target_std`

Copy the `target_mean` and `target_std` values into your config file (see below).

Common preprocessing flags:
- `--get-edges` builds neighbor edges during preprocessing. Use this when training GNNs
  that expect graph edges in the LMDB (e.g., GemNet, DimeNet++). If you omit it, the
  training pipeline will compute neighbors on the fly.
- `--id-column` selects the CSV column to store as `sample_id`.
- `--max-neighbors`, `--radius` can be set to match your model config if you want
  preprocessing and training to share the same neighbor settings.

### Auto-split from a single CSV

If you only have one CSV, you can ask the script to split and build all three
datasets in one step:

```bash
python scripts/preprocess_hardness.py \
  --csv data/hardness/all.csv \
  --out-root data/hardness \
  --cif-root data/hardness/cifs \
  --get-edges \
  --split 0.8 0.1 0.1
```

This generates:
- `data/hardness/train.csv`, `val.csv`, `test.csv`
- `data/hardness/train/`, `val/`, `test/` (each containing `data.lmdb`, `metadata.npz`, `target_stats.json`)

You can customize split names with `--split-names train val test` and seed with
`--split-seed`.

## 3) Configure model for hardness regression

This repo ships example configs for all included models under `configs/hardness/`:

- `cgcnn/cgcnn.yml`
- `dimenet/dimenet.yml`
- `dimenet_plus_plus/dpp.yml`
- `forcenet/forcenet.yml`
- `gemnet/gemnet.yml`
- `schnet/schnet.yml`
- `spinconv/spinconv.yml`

Pick one config and update the dataset statistics:

```yaml
dataset:
  - src: data/hardness/train/data.lmdb
    normalize_labels: True
    target_mean: <from target_stats.json>
    target_std: <from target_stats.json>
  - src: data/hardness/val/data.lmdb
  - src: data/hardness/test/data.lmdb
```

Notes:
- Hardness is an intensive property, so `extensive: False` is set in the GemNet config.
- `scale_file` in the GemNet config reuses the existing GemNet scaling factors.
- If you train on a different neighbor radius or max neighbors, ensure the model config
  (`cutoff`, `max_neighbors`, etc.) matches the preprocessing settings to avoid graph
  mismatches.
- The hardness configs already set `target_property: hardness`; keep it consistent if you
  copy configs to new locations.

## 4) Train and predict

### What the training command does

The training command starts a full training run using the YAML config:

```bash
python main.py --mode train --config-yml configs/hardness/gemnet/gemnet.yml
```

Key behavior:
- `--mode train` uses the `TrainTask`, which calls the `EnergyTrainer` training loop.
- `--config-yml` points to the model-specific config, which includes the shared
  `configs/hardness/base.yml`. Together they define:
  - `dataset`: train/val/test LMDB paths and label normalization stats
  - `task`: regression settings and label name (`hardness`)
  - `model`: GemNet architecture and neighbor settings
  - `optim`: batch size, learning rate, scheduler, and epochs
- The run creates time-stamped output directories under the current working directory:
  - `checkpoints/<timestamp>/` for `checkpoint.pt` and `best_checkpoint.pt`
  - `results/<timestamp>/` for prediction files
  - `logs/<logger>/<timestamp>/` for training logs (e.g., TensorBoard)

Typical files produced during/after training:
- `checkpoints/<timestamp>/checkpoint.pt`: latest training checkpoint.
- `checkpoints/<timestamp>/best_checkpoint.pt`: best checkpoint based on validation MAE.
- `results/<timestamp>/is2re_predictions.npz`: predictions saved during training whenever
  validation improves and a test split exists.
- `results/<timestamp>/hardness_predictions/{train,val,test}.csv`: exported CSV predictions
  if `task.export_predictions: True` (enabled in the hardness base config).

### Predict (inference-only)

Use predict mode to run inference on the test split with a chosen checkpoint:

```bash
python main.py --mode predict --config-yml configs/hardness/gemnet/gemnet.yml \
  --checkpoint checkpoints/<run-id>/checkpoint.pt
```

Notes:
- The config still provides the test dataset (`dataset[2]` in the hardness configs).
- `--checkpoint` loads model weights from a training run. You can use `best_checkpoint.pt`
  or `checkpoint.pt`.
- `run-id` refers to the timestamp directory created during training (e.g., `2024-01-01-12-00-00`).

### Prediction outputs

The prediction file contains:
- `ids`: the `sample_id` values (or row indices if no `sample_id` provided).
- `predictions`: model outputs in the same units as your training labels.
- `targets`: ground-truth hardness values when labels are present.

Where to find them:
- Predict mode writes `results/<timestamp>/is2re_predictions.npz`.
- During training, the same filename is used when predictions are written after a
  validation improvement (with a test split available).

For inference-only runs (no labels), `targets` will be absent; use `ids` to map predictions
back to the input CSV/CIFs. If you exported CSV predictions, the same mapping is already
flattened into `results/<timestamp>/hardness_predictions/*.csv`.

### Heteroscedastic loss (optional)

You can enable heteroscedastic loss to have the model output both a prediction mean and
sample-level uncertainty. This augments the usual regression output with a variance or
standard deviation and typically uses a Gaussian negative log-likelihood formulation.

Recommended config pattern:
- `loss.type`: `homoscedastic` or `heteroscedastic` (default: `homoscedastic`).
- `loss.heteroscedastic.enabled`: `true` to activate the heteroscedastic path.
- `loss.heteroscedastic.output`: `variance` or `std` to control the uncertainty field.

When enabled:
- The model emits both a prediction mean and an uncertainty value per sample.
- Exported CSVs at `results/<timestamp>/hardness_predictions/{train,val,test}.csv` include
  the additional uncertainty column (`variance` or `std`) alongside `sample_id`, `target`,
  and `prediction`.

When disabled (default), the training and export flow is unchanged and CSVs keep the
original three columns.
