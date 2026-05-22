#!/usr/bin/env python3
"""Train ChiralRTlib stage-1 separation classifiers.

The released best models use approach3 with joint training over IA, IB, IC,
and IG. The original curated training data are not distributed in this public
repository; provide compatible derived training tables with the schema in
docs/data_format.md.

Noise-handling options:
  approach1: remove positive training examples with Rs < rs_threshold.
  approach2: relabel positive training examples with Rs < rs_threshold as 0.
  approach3: soft-weight positives by clip(Rs / rs_threshold, 0, 1).

Example:
    python scripts/train_stage1_denoise.py \
        --approach approach3 \
        --column IC \
        --joint_train \
        --rs_threshold 1.5 \
        --data_dir training_tables \
        --out_dir outputs/reproduced/IC \
        --accelerator gpu
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import lightning as L
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

import chemprop
from chemprop import data as cpdata, nn as cpnn, models as cpmodels

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("medium")

COLUMNS = ["IA", "IB", "IC", "IG"]
COLUMN_IDX = {c: i for i, c in enumerate(COLUMNS)}


# ---------------------------------------------------------------------------
# Features used by the released models.
# ---------------------------------------------------------------------------

def compute_chiral_features(mol: Chem.Mol, hist_bins: int = 8) -> np.ndarray:
    n_atoms = max(mol.GetNumAtoms(), 1)
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, includeCIP=True, useLegacyImplementation=False)
    center_indices = [idx for idx, _ in centers]
    cip_tags       = [tag for _, tag in centers]
    n_R, n_S = float(cip_tags.count("R")), float(cip_tags.count("S"))
    n_unk = float(len(cip_tags) - int(n_R) - int(n_S))
    n_tot = float(len(center_indices))
    if n_tot == 0:
        pos_feats = np.zeros(6, dtype=np.float32)
        hist      = np.zeros(hist_bins, dtype=np.float32)
    else:
        ranks = np.asarray(list(Chem.CanonicalRankAtoms(mol, breakTies=True)), dtype=np.float32)
        pos   = np.clip(ranks[center_indices] / max(float(n_atoms - 1), 1.0), 0.0, 1.0)
        pos_feats = np.array([n_tot, n_tot/n_atoms, pos.min(), pos.mean(), pos.max(), pos.std()], dtype=np.float32)
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
    physico = np.array([n_R, n_S, n_unk, mw, logp, tpsa, hbd+hba, fsp3], dtype=np.float32)
    return np.concatenate([pos_feats, hist, physico])


def compute_feature_matrix(smiles_list, columns_list=None):
    """Compute chiral features. columns_list is ignored (single-column mode)."""
    rows, keep = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        rows.append(compute_chiral_features(mol))
        keep.append(i)
    return np.vstack(rows).astype(np.float32), keep


# ---------------------------------------------------------------------------
# Data loading + Rs join
# ---------------------------------------------------------------------------

def load_column_data(data_dir: Path, column: str) -> pd.DataFrame:
    """
    Load Stage1 data for a single column, joined with Rs values.
    Returns full_df with extra column 'rs' (NaN for class 0).
    """
    s1 = pd.read_csv(data_dir / f"{column}_stage1.csv")
    s2 = pd.read_csv(data_dir / f"{column}_stage2_rs.csv")[["mol_id", "rs"]]
    s1["column"] = column
    s1 = s1.merge(s2, on="mol_id", how="left")  # rs = NaN for class 0
    return s1


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def mol_scaffold_split(df: pd.DataFrame, test_size: float, seed: int):
    mol_df = df.drop_duplicates("mol_id")[["mol_id", "scaffold"]].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_m, te_m = next(splitter.split(mol_df, groups=mol_df["scaffold"]))
    tr_mols = set(mol_df.iloc[tr_m]["mol_id"])
    te_mols = set(mol_df.iloc[te_m]["mol_id"])
    tr = np.where(df["mol_id"].isin(tr_mols))[0]
    te = np.where(df["mol_id"].isin(te_mols))[0]
    return tr, te


def mol_scaffold_kfold_split(df: pd.DataFrame, fold_idx: int, n_folds: int):
    mol_df = df.drop_duplicates("mol_id")[["mol_id", "scaffold"]].reset_index(drop=True)
    splitter = GroupKFold(n_splits=n_folds)
    splits = list(splitter.split(mol_df, groups=mol_df["scaffold"]))
    if not 0 <= fold_idx < len(splits):
        raise ValueError(f"fold_idx must be in [0, {len(splits) - 1}], got {fold_idx}")
    tr_m, te_m = splits[fold_idx]
    tr_mols = set(mol_df.iloc[tr_m]["mol_id"])
    te_mols = set(mol_df.iloc[te_m]["mol_id"])
    tr = np.where(df["mol_id"].isin(tr_mols))[0]
    te = np.where(df["mol_id"].isin(te_mols))[0]
    return tr, te


def scaffold_kfold_split(df: pd.DataFrame, fold_idx: int, n_folds: int):
    splitter = GroupKFold(n_splits=n_folds)
    splits = list(splitter.split(df, groups=df["scaffold"]))
    if not 0 <= fold_idx < len(splits):
        raise ValueError(f"fold_idx must be in [0, {len(splits) - 1}], got {fold_idx}")
    return splits[fold_idx]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def balanced_sample_weights(labels: np.ndarray, extra_weights: np.ndarray | None = None) -> np.ndarray:
    """Balanced weights, optionally multiplied by extra_weights (for soft labels)."""
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    w = np.zeros(len(labels), dtype=np.float32)
    for cls in (0, 1):
        if counts[cls] > 0:
            w[labels == cls] = len(labels) / (2.0 * counts[cls])
    if extra_weights is not None:
        w *= extra_weights.astype(np.float32)
        # re-normalize so mean weight stays ~1
        w = w / (w.mean() + 1e-8)
    return w


def optimize_threshold(y_true: np.ndarray, proba: np.ndarray, n: int = 200) -> float:
    best_thr, best_ba = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, n):
        ba = balanced_accuracy_score(y_true, (proba >= thr).astype(int))
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)
    return best_thr


# ---------------------------------------------------------------------------
# Chemprop helpers
# ---------------------------------------------------------------------------

def build_cls_dataset(smiles, labels, weights, x_d):
    points = [
        cpdata.MoleculeDatapoint.from_smi(
            smi,
            y=np.array([lbl], dtype=np.float32),
            weight=float(w),
            x_d=x_d[i] if x_d is not None else None,
        )
        for i, (smi, lbl, w) in enumerate(zip(smiles, labels, weights))
    ]
    return cpdata.MoleculeDataset(points)


def build_cls_model(extra_dim, mpnn_hidden, mpnn_depth, ffn_hidden, dropout):
    mp  = cpnn.BondMessagePassing(d_h=mpnn_hidden, depth=mpnn_depth, dropout=dropout)
    agg = cpnn.MeanAggregation()
    ffn = cpnn.BinaryClassificationFFN(
        input_dim=mp.output_dim + extra_dim,
        hidden_dim=ffn_hidden, n_layers=2, dropout=dropout,
    )
    return cpmodels.MPNN(message_passing=mp, agg=agg, predictor=ffn)


def train_one_seed(train_dset, val_dset, test_dset, args, ckpt_dir, seed, extra_dim):
    rng   = np.random.default_rng(seed)
    n_val = max(1, int(len(train_dset) * 0.1))
    idx   = np.arange(len(train_dset))
    rng.shuffle(idx)
    tr_sub  = torch.utils.data.Subset(train_dset, idx[n_val:].tolist())
    val_sub = torch.utils.data.Subset(train_dset, idx[:n_val].tolist())

    tr_loader   = cpdata.build_dataloader(tr_sub,  batch_size=64, shuffle=True)
    val_loader  = cpdata.build_dataloader(val_sub, batch_size=64, shuffle=False)
    val_full_l  = cpdata.build_dataloader(val_dset,  batch_size=64, shuffle=False)
    test_loader = cpdata.build_dataloader(test_dset, batch_size=64, shuffle=False)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = build_cls_model(extra_dim, args.mpnn_hidden, args.mpnn_depth, args.ffn_hidden, args.dropout)
    L.seed_everything(seed)
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
        gradient_clip_val=1.0, enable_progress_bar=True,
        enable_model_summary=False, logger=False,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_dir), filename=f"seed{seed}_best",
                monitor="val_loss", mode="min", save_top_k=1,
            ),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, mode="min", verbose=False,
            ),
        ],
    )
    trainer.fit(model, tr_loader, val_loader)
    best = trainer.checkpoint_callback.best_model_path
    if best and Path(best).exists():
        model = type(model).load_from_checkpoint(best)

    val_proba  = torch.cat(trainer.predict(model, val_full_l),  dim=0).squeeze(-1).cpu().numpy()
    test_proba = torch.cat(trainer.predict(model, test_loader), dim=0).squeeze(-1).cpu().numpy()
    return val_proba, test_proba


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    rs_thr   = args.rs_threshold

    (out_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2))

    # ── Load data ─────────────────────────────────────────────────────────
    column    = args.column
    is_joint  = args.joint_train
    mode_str  = "joint(4-col)" if is_joint else column

    print(f"Loading {mode_str} data (approach={args.approach}, rs_threshold={rs_thr}) …", flush=True)

    if is_joint:
        parts = []
        for col in COLUMNS:
            parts.append(load_column_data(data_dir, col))
        full_df = pd.concat(parts, ignore_index=True)
    else:
        full_df = load_column_data(data_dir, column)

    print(f"  Total rows: {len(full_df)}", flush=True)

    # ── Feature computation ───────────────────────────────────────────────
    # Joint: chiral(22) + col-onehot(4)  |  single: chiral(22) only
    print("  Computing features …", flush=True)
    rows, keep = [], []
    for i, row in full_df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        feats = [compute_chiral_features(mol)]
        if is_joint:
            ohe = np.zeros(4, dtype=np.float32)
            ohe[COLUMN_IDX[row["column"]]] = 1.0
            feats.append(ohe)
        rows.append(np.concatenate(feats))
        keep.append(i - full_df.index[0])

    xd_raw    = np.vstack(rows).astype(np.float32)
    full_df   = full_df.iloc[keep].reset_index(drop=True)
    extra_dim = xd_raw.shape[1]
    print(f"  Feature dim: {extra_dim}  Valid mols: {len(full_df)}", flush=True)

    # ── Train/test split ──────────────────────────────────────────────────
    if args.split == "scaffold_cv":
        if is_joint:
            tr_idx_arr, te_idx_arr = mol_scaffold_kfold_split(full_df, args.fold_idx, args.n_folds)
        else:
            tr_idx_arr, te_idx_arr = scaffold_kfold_split(full_df, args.fold_idx, args.n_folds)
    elif is_joint:
        tr_idx_arr, te_idx_arr = mol_scaffold_split(full_df, args.test_size, args.seed)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx_arr, te_idx_arr = next(splitter.split(full_df, groups=full_df["scaffold"]))

    train_df_orig = full_df.iloc[tr_idx_arr].reset_index(drop=True)
    test_df       = full_df.iloc[te_idx_arr].reset_index(drop=True)
    xd_tr_all = xd_raw[tr_idx_arr]
    xd_te_all = xd_raw[te_idx_arr]

    # Column-specific test slice.
    if is_joint:
        ic_te_mask   = test_df["column"] == column
        ic_test_df   = test_df[ic_te_mask].reset_index(drop=True)
        xd_te_ic     = xd_te_all[ic_te_mask.values]
    else:
        ic_test_df   = test_df
        xd_te_ic     = xd_te_all
    y_te_ic_orig = ic_test_df["stage1_label"].to_numpy()

    # ── Apply approach to training data ───────────────────────────────────
    train_df = train_df_orig.copy()

    if args.approach == "approach1":
        # 灰区移除：去掉训练集中 class 1 且 Rs < threshold 的样本
        is_noisy = (train_df["stage1_label"] == 1) & (train_df["rs"] < rs_thr)
        n_removed = is_noisy.sum()
        train_df  = train_df[~is_noisy].reset_index(drop=True)
        xd_tr_all = xd_tr_all[~is_noisy.values]
        print(f"  [Approach 1] Removed {n_removed} noisy class-1 samples (Rs < {rs_thr})", flush=True)
        y_tr   = train_df["stage1_label"].to_numpy()
        w_soft = None

    elif args.approach == "approach2":
        # 标签重定义：Rs < threshold 的 class 1 → 重标为 class 0
        is_noisy = (train_df["stage1_label"] == 1) & (train_df["rs"] < rs_thr)
        n_relabeled = is_noisy.sum()
        train_df.loc[is_noisy, "stage1_label"] = 0
        print(f"  [Approach 2] Relabeled {n_relabeled} samples: class1(Rs<{rs_thr}) → class0", flush=True)
        y_tr   = train_df["stage1_label"].to_numpy()
        w_soft = None

    elif args.approach == "approach3":
        # 软标签加权：class 1 权重 = clip(Rs/threshold, 0, 1)
        y_tr   = train_df["stage1_label"].to_numpy()
        rs_arr = train_df["rs"].fillna(0.0).to_numpy(dtype=np.float64)
        w_soft = np.ones(len(train_df), dtype=np.float32)
        cls1_mask = (y_tr == 1)
        w_soft[cls1_mask] = np.clip(rs_arr[cls1_mask] / rs_thr, 0.0, 1.0).astype(np.float32)
        n_low = (w_soft[cls1_mask] < 1.0).sum()
        print(f"  [Approach 3] Soft-weighted {n_low} class-1 samples with Rs < {rs_thr}", flush=True)
    else:
        raise ValueError(f"Unknown approach: {args.approach}")

    # ── Balanced weights (with optional soft scaling) ─────────────────────
    w_tr = balanced_sample_weights(y_tr, extra_weights=w_soft)

    print(
        f"  Train: {len(train_df)}  (cls0={int((y_tr==0).sum())}  cls1={int((y_tr==1).sum())})"
        f"  Test(IC): {len(ic_test_df)}", flush=True
    )

    # ── Build datasets ────────────────────────────────────────────────────
    # Feature scaling
    scaler  = RobustScaler()
    xd_tr   = scaler.fit_transform(xd_tr_all)
    xd_te   = scaler.transform(xd_te_all)
    pickle.dump(scaler, open(out_dir / "scaler.pkl", "wb"))

    train_dset = build_cls_dataset(train_df["smiles"].tolist(), y_tr, w_tr, xd_tr)

    # Test dataset
    y_te_all = test_df["stage1_label"].to_numpy()
    test_dset_all = build_cls_dataset(
        test_df["smiles"].tolist(), y_te_all,
        np.ones(len(test_df), dtype=np.float32), xd_te,
    )

    # Validation rows from the target column are used for threshold optimization.
    if is_joint:
        ic_tr_mask = train_df["column"] == column
        ic_tr_df   = train_df[ic_tr_mask].reset_index(drop=True)
        xd_tr_ic   = xd_tr[ic_tr_mask.values]
    else:
        ic_tr_df   = train_df
        xd_tr_ic   = xd_tr
    y_tr_ic    = ic_tr_df["stage1_label"].to_numpy()
    rng_val    = np.random.default_rng(args.seed + 9999)
    n_val_ic   = max(50, int(len(ic_tr_df) * 0.15))
    val_ic_idx = rng_val.choice(len(ic_tr_df), size=n_val_ic, replace=False)
    val_dset_ic = build_cls_dataset(
        ic_tr_df.iloc[val_ic_idx]["smiles"].tolist(),
        y_tr_ic[val_ic_idx],
        np.ones(n_val_ic, dtype=np.float32),
        xd_tr_ic[val_ic_idx],
    )
    print(f"  {column} val set (threshold opt): {n_val_ic}", flush=True)

    # ── Multi-seed ensemble ───────────────────────────────────────────────
    seeds = [args.seed + i * 100 for i in range(args.n_seeds)]
    all_val_proba_ic  = []
    all_test_proba_ic = []
    all_test_proba_all = []

    print(f"\n  Running {args.n_seeds}-seed ensemble …\n", flush=True)
    for si, seed in enumerate(seeds):
        print(f"  ── Seed {si+1}/{args.n_seeds} (seed={seed}) ──", flush=True)
        ckpt_dir = out_dir / f"seed_{seed}" / "checkpoints"

        val_proba, test_proba_all = train_one_seed(
            train_dset, val_dset_ic, test_dset_all,
            args, ckpt_dir, seed, extra_dim,
        )
        # Slice IC-specific test probabilities for joint mode
        if is_joint:
            test_proba_ic = test_proba_all[ic_te_mask.values]
        else:
            test_proba_ic = test_proba_all

        all_val_proba_ic.append(val_proba)
        all_test_proba_ic.append(test_proba_ic)
        all_test_proba_all.append(test_proba_all)

        pred_ic = (test_proba_ic >= 0.5).astype(int)
        ba_ic   = balanced_accuracy_score(y_te_ic_orig, pred_ic)
        f1_ic   = f1_score(y_te_ic_orig, pred_ic, zero_division=0)
        print(f"    Seed {seed}: {column} BA={ba_ic:.4f}  F1={f1_ic:.4f}", flush=True)

    # ── Ensemble + threshold optimization ─────────────────────────────────
    ens_val_ic  = np.mean(all_val_proba_ic,  axis=0)
    ens_test_ic = np.mean(all_test_proba_ic, axis=0)
    ens_test_all = np.mean(all_test_proba_all, axis=0)

    y_val_ic_labels = ic_tr_df.iloc[val_ic_idx]["stage1_label"].to_numpy()
    best_thr = optimize_threshold(y_val_ic_labels, ens_val_ic)
    print(f"\n  Optimal threshold ({column} val): {best_thr:.3f}", flush=True)

    # ── Final metrics ──────────────────────────────────────────────────────
    print()
    for thr_name, thr in [("thr=0.50", 0.5), (f"thr={best_thr:.3f}(opt)", best_thr)]:
        pred = (ens_test_ic >= thr).astype(int)
        ba   = balanced_accuracy_score(y_te_ic_orig, pred)
        f1   = f1_score(y_te_ic_orig, pred, zero_division=0)
        print(f"  [{thr_name}]  {column}: BA={ba:.4f}  F1={f1:.4f}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────
    pred_df = ic_test_df.copy()
    pred_df["p_separated"] = ens_test_ic
    pred_df["pred_05"]     = (ens_test_ic >= 0.5).astype(int)
    pred_df["pred_opt"]    = (ens_test_ic >= best_thr).astype(int)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    metrics = {
        "approach":          args.approach,
        "split":             args.split,
        "fold_idx":          args.fold_idx,
        "n_folds":           args.n_folds,
        "rs_threshold":      rs_thr,
        "n_train":           len(train_df),
        "n_test":            len(ic_test_df),
        "optimal_threshold": best_thr,
        "ba_thr05":          float(balanced_accuracy_score(y_te_ic_orig, (ens_test_ic >= 0.5).astype(int))),
        "f1_thr05":          float(f1_score(y_te_ic_orig, (ens_test_ic >= 0.5).astype(int), zero_division=0)),
        "ba_opt":            float(balanced_accuracy_score(y_te_ic_orig, (ens_test_ic >= best_thr).astype(int))),
        "f1_opt":            float(f1_score(y_te_ic_orig, (ens_test_ic >= best_thr).astype(int), zero_division=0)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDone → {out_dir}")
    print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", choices=["approach1","approach2","approach3"], required=True)
    ap.add_argument("--column",       default="IC", choices=COLUMNS)
    ap.add_argument("--joint_train",  action="store_true",
                    help="Train on all 4 columns jointly; evaluate on IC test slice")
    ap.add_argument("--rs_threshold", type=float, default=1.0)
    ap.add_argument("--data_dir",  default="training_tables")
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--split",     choices=["scaffold", "scaffold_cv"], default="scaffold")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed",      type=int,   default=2028)
    ap.add_argument("--fold_idx",  type=int,   default=0)
    ap.add_argument("--n_folds",   type=int,   default=5)
    ap.add_argument("--n_seeds",   type=int,   default=5)
    ap.add_argument("--mpnn_hidden", type=int, default=600)
    ap.add_argument("--mpnn_depth",  type=int, default=4)
    ap.add_argument("--ffn_hidden",  type=int, default=600)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--epochs",      type=int,   default=200)
    ap.add_argument("--patience",    type=int,   default=30)
    ap.add_argument("--accelerator", default="gpu")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
