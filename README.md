# ChiralRTlib

ChiralRTlib provides released Chemprop-based ensemble models for predicting
whether a molecule is likely to separate on four chiral chromatographic
columns: `IA`, `IB`, `IC`, and `IG`.

This public release includes:

- trained best checkpoints and scalers under `models/best_models/`;
- a prediction script for new SMILES;
- the released prediction input table under `data/processed/`;
- derived training tables under `training_tables/`;
- model metadata, validation metrics, and reproducibility instructions.

The released derived training tables can be used to reproduce the best-model
training workflow. The upstream raw chromatographic table is also provided
under `data/raw/` for traceability.

## Installation

Create a fresh Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Prediction

Run the released ensemble on one column:

```bash
python models/best_models/predict.py \
  --column IC \
  --smiles_file data/processed/filtered_molecules.csv \
  --out_file predictions.csv
```

The input CSV must contain a `smiles` column by default. Use `--smiles_col` for
a different column name.

Output columns:

- `p_separated`: ensemble probability of successful chiral separation.
- `pred_label`: binary prediction using the released column-specific threshold.

Released thresholds:

| Column | Threshold |
| --- | ---: |
| IA | 0.43894472361809045 |
| IB | 0.37110552763819094 |
| IC | 0.3530150753768844 |
| IG | 0.6741206030150754 |

Override the decision threshold with `--threshold`.

## Reproducibility

See [docs/reproducibility.md](docs/reproducibility.md) for:

- exact inference reproduction from the released checkpoints;
- training-data schema expected by the training script;
- commands for retraining the best-model configuration;
- optional scaffold cross-validation commands.

## Repository Layout

```text
data/raw/                   Upstream chromatographic input table
data/processed/             Released SMILES table for inference
training_tables/            Derived stage-1 and Rs tables for retraining
models/best_models/        Released ensembles, scalers, metrics, configs
scripts/                   Training script used for model reproduction
docs/                      Model card, data format, and reproducibility notes
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
