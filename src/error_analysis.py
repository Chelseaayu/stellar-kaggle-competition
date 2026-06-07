"""Analyse where the current blend mis-classifies, using cached OOF.

Tells us which class confusions dominate and in what feature regions,
so feature engineering can target the actual failure modes.

Usage
-----
    cd stellar-kaggle
    python src/error_analysis.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

import features as F
import metrics as MET

# ── Resolve paths ───────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_PROJECT_ROOT, "oof_predictions.npz")

d = np.load(_CACHE)
y = d["y"]
oof = 0.7263 * d["oof_lgb"] + 0.1728 * d["oof_xgb"] + 0.1009 * d["oof_cat"]

# Tuned class weights from the training run
class_w = d["class_w"] if "class_w" in d else np.array([0.3313, 1.1885, 1.4802])
pred = MET.weighted_argmax(oof, class_w)

# ── Overall accuracy ────────────────────────────────────────────────
print("Balanced acc:", MET.fast_balanced_acc(y, pred))

# ── Confusion matrix ────────────────────────────────────────────────
print("\nConfusion matrix (rows=true, cols=pred) [GALAXY, QSO, STAR]:")
cm = confusion_matrix(y, pred)
print(cm)
print("\nPer-class recall:")
print((cm.diagonal() / cm.sum(1)).round(4))
print("\nClassification report:")
print(classification_report(y, pred, target_names=F.CLASSES, digits=4))

# ── Feature-level error analysis ────────────────────────────────────
train, _, feat_cols = F.load_data()
train = train.reset_index(drop=True)
train["y"] = y
train["pred"] = pred
train["wrong"] = (y != pred)

print("=" * 60)
print("Error rate:", train["wrong"].mean().round(4))

print("\nMost common confusions (true -> pred):")
conf = train[train["wrong"]].groupby(["y", "pred"]).size().sort_values(ascending=False)
inv = F.INT_TO_CLASS
for (ty, py), n in conf.items():
    print(f"  {inv[ty]:>6} -> {inv[py]:<6}: {n}")

print("\nRedshift stats for the biggest confusion groups:")
for (ty, py), n in conf.head(4).items():
    mask = (train["y"] == ty) & (train["pred"] == py)
    print(f"  {inv[ty]}->{inv[py]} (n={n}): redshift "
          f"mean={train.loc[mask, 'redshift'].mean():.3f} "
          f"median={train.loc[mask, 'redshift'].median():.3f} "
          f"min={train.loc[mask, 'redshift'].min():.3f} "
          f"max={train.loc[mask, 'redshift'].max():.3f}")

print("\nConfidence (max prob) on wrong vs right:")
maxp = oof.max(1)
print("  wrong mean conf:", maxp[train["wrong"].values].mean().round(4))
print("  right mean conf:", maxp[~train["wrong"].values].mean().round(4))
