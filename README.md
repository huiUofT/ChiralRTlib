# ChiralRTlib

ChiralRTlib provides released Chemprop-based ensemble models for predicting
whether a molecule is likely to separate on four chiral chromatographic
columns: `IA`, `IB`, `IC`, and `IG`.

This public release includes:

- trained best checkpoints and scalers under `models/best_models/`;
- a prediction script for new SMILES;
- a small example input file;
- model metadata, validation metrics, and reproducibility instructions.

The original curated training data are not included in this repository. Training
can be reproduced by providing compatible training tables with the documented
schema.

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
  --smiles_file examples/example_smiles.csv \
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
models/best_models/        Released ensembles, scalers, metrics, configs
scripts/                   Training script used for model reproduction
examples/                  Minimal public SMILES input
docs/                      Model card, data format, and reproducibility notes
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
