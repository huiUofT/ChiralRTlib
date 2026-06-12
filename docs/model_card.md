# ChiralRTlib Model Card

## Overview

ChiralRTlib provides Chemprop message-passing neural network ensembles for
predicting chiral chromatographic separation on `IA`, `IB`, `IC`, and `IG`
columns.

Each released column model is an ensemble of five independently trained
checkpoints with seeds `2028`, `2128`, `2228`, `2328`, and `2428`. Prediction is
performed by averaging seed-level probabilities and applying a
column-specific validation-optimized threshold.

## Architecture

- Backbone: Chemprop MPNN
- Message passing: `BondMessagePassing`
- Aggregation: `MeanAggregation`
- Prediction head: `BinaryClassificationFFN`
- Message-passing hidden dimension: 600
- Message-passing depth: 4
- FFN hidden dimension: 600
- Dropout: 0.1
- Auxiliary feature dimension: 26

The auxiliary feature vector contains 22 chirality and physicochemical
descriptors plus a 4-dimensional one-hot encoding of column identity.

## Training Summary

The released models use joint training over the four columns with column
identity included as an auxiliary feature. Low-Rs positive examples are handled
by soft sample weighting:

```text
weight = clip(Rs / rs_threshold, 0, 1)
```

Column-specific Rs thresholds:

| Column | Rs threshold |
| --- | ---: |
| IA | 0.5 |
| IB | 0.75 |
| IC | 1.5 |
| IG | 0.75 |

## Released Metrics

Column-specific metrics and training configs are stored at:

```text
models/best_models/{IA,IB,IC,IG}/metrics.json
models/best_models/{IA,IB,IC,IG}/experiment_config.json
```

Optional scaffold cross-validation can be reproduced with the commands in
`docs/reproducibility.md` using the released derived training tables.

## Intended Use

These models are intended for research prioritization of candidate molecules
for chiral chromatographic separation experiments. They should be interpreted
as screening signals rather than definitive experimental outcomes.

## Limitations

Performance depends on similarity between new molecules and the training data
distribution. Retraining uses the released derived tables in `training_tables/`;
new external datasets should follow the documented schema.
