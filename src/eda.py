"""Quick EDA for the Stellar Classification Playground (S6E6).

Prints class distribution, dtypes, missingness, basic stats and
category cardinalities.  Uses only pandas/numpy so it runs even before
the heavy ML libraries finish installing.

Usage
-----
    cd stellar-kaggle
    python src/eda.py
"""
import os
import pandas as pd
import numpy as np

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

# ── Resolve data paths ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find(name: str) -> str:
    for p in [os.path.join(_PROJECT_ROOT, "data", name),
              os.path.join(_PROJECT_ROOT, name)]:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Cannot find '{name}'")


train = pd.read_csv(_find("train.csv"))
test = pd.read_csv(_find("test.csv"))

# ── Shapes ──────────────────────────────────────────────────────────
print("=" * 70)
print("SHAPES")
print("train:", train.shape, "test:", test.shape)

# ── Dtypes ──────────────────────────────────────────────────────────
print("=" * 70)
print("DTYPES")
print(train.dtypes)

# ── Class distribution ──────────────────────────────────────────────
print("=" * 70)
print("CLASS DISTRIBUTION (train)")
print(train["class"].value_counts())
print(train["class"].value_counts(normalize=True).round(4))

# ── Missing values ──────────────────────────────────────────────────
print("=" * 70)
print("MISSING VALUES (train)")
print(train.isna().sum()[train.isna().sum() > 0])
print("MISSING VALUES (test)")
print(test.isna().sum()[test.isna().sum() > 0])

# ── Categorical cardinalities ───────────────────────────────────────
print("=" * 70)
print("CATEGORICAL CARDINALITIES")
for c in train.columns:
    if train[c].dtype == object and c != "class":
        print(f"--- {c} ---")
        print("train uniques:", sorted(train[c].dropna().unique().tolist()))
        print("test uniques :", sorted(test[c].dropna().unique().tolist()))

# ── Numeric statistics ──────────────────────────────────────────────
print("=" * 70)
print("NUMERIC DESCRIBE (train)")
num_cols = [c for c in train.columns if train[c].dtype != object and c != "id"]
print(train[num_cols].describe().T)

# ── Categorical vs. class ──────────────────────────────────────────
print("=" * 70)
print("CATEGORICAL vs CLASS (spectral_type)")
print(pd.crosstab(train["spectral_type"], train["class"], normalize="index").round(3))
print("CATEGORICAL vs CLASS (galaxy_population)")
print(pd.crosstab(train["galaxy_population"], train["class"], normalize="index").round(3))

# ── Redshift by class ──────────────────────────────────────────────
print("=" * 70)
print("REDSHIFT by class")
print(train.groupby("class")["redshift"].describe().round(4))

# ── Photometric band ranges (check for sentinels) ──────────────────
print("=" * 70)
print("MIN/MAX of photometric bands (look for -9999 sentinels)")
for c in ["u", "g", "r", "i", "z"]:
    print(c, "min:", train[c].min(), "max:", train[c].max())
