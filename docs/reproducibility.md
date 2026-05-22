# Reproducibility

This release supports two levels of reproducibility:

1. Reproduce inference exactly from the released checkpoints.
2. Reproduce training when compatible private training tables are supplied.

The original curated training data are intentionally not included in this
public repository.

## 1. Reproduce Released Inference

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the example:

```bash
python models/best_models/predict.py \
  --column IC \
  --smiles_file examples/example_smiles.csv \
  --out_file predictions.csv
```

The script loads:

```text
models/best_models/IC/scaler.pkl
models/best_models/IC/checkpoints/*.ckpt
```

and writes `p_separated` plus `pred_label`.

## 2. Reproduce Best-Model Training

Prepare the private training tables described in `docs/data_format.md`:

```text
data/private/training_tables/IA_stage1.csv
data/private/training_tables/IA_stage2_rs.csv
data/private/training_tables/IB_stage1.csv
data/private/training_tables/IB_stage2_rs.csv
data/private/training_tables/IC_stage1.csv
data/private/training_tables/IC_stage2_rs.csv
data/private/training_tables/IG_stage1.csv
data/private/training_tables/IG_stage2_rs.csv
```

Then run the released best-model settings for each column:

```bash
python scripts/train_stage1_denoise.py \
  --approach approach3 \
  --column IA \
  --joint_train \
  --rs_threshold 0.5 \
  --data_dir data/private/training_tables \
  --out_dir outputs/reproduced/IA \
  --split scaffold \
  --seed 2028 \
  --n_seeds 5 \
  --mpnn_hidden 600 \
  --mpnn_depth 4 \
  --ffn_hidden 600 \
  --dropout 0.1 \
  --patience 30 \
  --epochs 200 \
  --accelerator gpu
```

Change only `--column`, `--rs_threshold`, and `--out_dir` for the other
released columns:

| Column | Rs threshold | Output directory |
| --- | ---: | --- |
| IA | 0.5 | `outputs/reproduced/IA` |
| IB | 0.75 | `outputs/reproduced/IB` |
| IC | 1.5 | `outputs/reproduced/IC` |
| IG | 0.75 | `outputs/reproduced/IG` |

Or run all four released configurations:

```bash
bash scripts/reproduce_best_models.sh data/private/training_tables
```

The script writes:

```text
scaler.pkl
metrics.json
test_predictions.csv
experiment_config.json
seed_*/checkpoints/*.ckpt
```

## 3. Five-Fold Evaluation

To repeat a scaffold cross-validation fold with private data:

```bash
python scripts/train_stage1_denoise.py \
  --approach approach3 \
  --column IC \
  --joint_train \
  --rs_threshold 1.5 \
  --data_dir data/private/training_tables \
  --out_dir outputs/cv5/IC/fold_0 \
  --split scaffold_cv \
  --fold_idx 0 \
  --n_folds 5 \
  --seed 2028 \
  --n_seeds 5 \
  --accelerator gpu
```

Repeat `--fold_idx 0` through `4` and each released column/Rs-threshold pair.
