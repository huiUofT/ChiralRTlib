#!/usr/bin/env python3
"""Predict chiral chromatographic separation from SMILES.

Example:
    python models/best_models/predict.py \
        --column IC \
        --smiles_file data/processed/filtered_molecules.csv \
        --out_file predictions.csv

The output CSV contains the separation probability (`p_separated`) and a
binary prediction (`pred_label`). Invalid SMILES are kept in the output with
missing prediction values.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from chemprop import data as cpdata, nn as cpnn, models as cpmodels

COLUMNS = ["IA", "IB", "IC", "IG"]
COLUMN_IDX = {c: i for i, c in enumerate(COLUMNS)}

BEST_THR = {
    "IA": 0.43894472361809045,
    "IB": 0.37110552763819094,
    "IC": 0.3530150753768844,
    "IG": 0.6741206030150754,
}

BEST_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Features: keep these identical to the training pipeline.
# ---------------------------------------------------------------------------

def compute_chiral_features(mol: Chem.Mol, hist_bins: int = 8) -> np.ndarray:
    n_atoms = max(mol.GetNumAtoms(), 1)
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, includeCIP=True, useLegacyImplementation=False,
    )
    center_indices = [idx for idx, _ in centers]
    cip_tags       = [tag for _, tag in centers]
    n_R   = float(cip_tags.count("R"))
    n_S   = float(cip_tags.count("S"))
    n_unk = float(len(cip_tags) - int(n_R) - int(n_S))
    n_tot = float(len(center_indices))

    if n_tot == 0:
        pos_feats = np.zeros(6, dtype=np.float32)
        hist      = np.zeros(hist_bins, dtype=np.float32)
    else:
        ranks = np.asarray(
            list(Chem.CanonicalRankAtoms(mol, breakTies=True)), dtype=np.float32
        )
        pos = ranks[center_indices] / max(float(n_atoms - 1), 1.0)
        pos = np.clip(pos, 0.0, 1.0)
        pos_feats = np.array(
            [n_tot, n_tot / n_atoms, float(pos.min()),
             float(pos.mean()), float(pos.max()), float(pos.std())],
            dtype=np.float32,
        )
        hist, _ = np.histogram(pos, bins=hist_bins, range=(0.0, 1.0))
        hist = (hist / n_tot).astype(np.float32)

    try:
        mw   = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        hbd  = float(rdMolDescriptors.CalcNumHBD(mol))
        hba  = float(rdMolDescriptors.CalcNumHBA(mol))
        fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    except Exception:
        mw = logp = tpsa = hbd = hba = fsp3 = 0.0

    physico = np.array([n_R, n_S, n_unk, mw, logp, tpsa, hbd + hba, fsp3], dtype=np.float32)
    return np.concatenate([pos_feats, hist, physico])


def column_onehot(column: str) -> np.ndarray:
    v = np.zeros(4, dtype=np.float32)
    v[COLUMN_IDX[column]] = 1.0
    return v


def build_cls_model(extra_dim: int) -> cpmodels.MPNN:
    mp  = cpnn.BondMessagePassing(d_h=600, depth=4, dropout=0.1)
    agg = cpnn.MeanAggregation()
    ffn = cpnn.BinaryClassificationFFN(
        input_dim=mp.output_dim + extra_dim,
        hidden_dim=600, n_layers=2, dropout=0.1,
    )
    return cpmodels.MPNN(message_passing=mp, agg=agg, predictor=ffn)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(
    column: str,
    smiles_list: list[str],
    threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    col_dir = BEST_DIR / column
    with open(col_dir / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    ckpt_dir = col_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    # Compute features
    rows, keep = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        feat = np.concatenate([compute_chiral_features(mol), column_onehot(column)])
        rows.append(feat)
        keep.append(i)

    selected_threshold = BEST_THR[column] if threshold is None else threshold
    if not rows:
        return np.full(len(smiles_list), np.nan), selected_threshold

    xd_raw = np.vstack(rows).astype(np.float32)
    xd_scaled = scaler.transform(xd_raw)
    extra_dim = xd_scaled.shape[1]

    valid_smiles = [smiles_list[i] for i in keep]
    dset = cpdata.MoleculeDataset([
        cpdata.MoleculeDatapoint.from_smi(smi, y=np.array([0.0]), x_d=xd_scaled[j])
        for j, smi in enumerate(valid_smiles)
    ])
    loader = cpdata.build_dataloader(dset, batch_size=64, shuffle=False)

    # Ensemble over all seeds
    all_proba = []
    model = build_cls_model(extra_dim)
    import lightning as L
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )
    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    for ckpt in ckpts:
        m = type(model).load_from_checkpoint(str(ckpt), map_location=map_location)
        preds = trainer.predict(m, loader)
        proba = torch.cat(preds, dim=0).squeeze(-1).cpu().numpy()
        all_proba.append(proba)

    ens_proba = np.mean(all_proba, axis=0)

    # Map back to all inputs (invalid SMILES get NaN)
    out = np.full(len(smiles_list), np.nan)
    for j, i in enumerate(keep):
        out[i] = ens_proba[j]

    return out, selected_threshold


def main():
    ap = argparse.ArgumentParser(
        description="Predict chiral separation on IA, IB, IC, or IG columns."
    )
    ap.add_argument("--column", required=True, choices=COLUMNS)
    ap.add_argument("--smiles_file", required=True, help="Input CSV file.")
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--out_file", required=True)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold. Defaults to the validation-optimized threshold.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.smiles_file)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' was not found in {args.smiles_file}")
    smiles = df[args.smiles_col].tolist()

    proba, thr = predict(args.column, smiles, args.threshold)

    df["p_separated"] = proba
    df["pred_label"] = pd.Series(
        np.where(np.isnan(proba), pd.NA, proba >= thr),
        dtype="boolean",
    ).astype("Int64")
    df.to_csv(args.out_file, index=False)
    print(f"Done. threshold={thr:.4f} saved to {args.out_file}")
    print(f"Separated (1): {(df['pred_label'] == 1).sum()}")
    print(f"Not separated (0): {(df['pred_label'] == 0).sum()}")
    print(f"Invalid/missing predictions: {df['pred_label'].isna().sum()}")


if __name__ == "__main__":
    main()
