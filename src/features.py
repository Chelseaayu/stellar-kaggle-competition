"""Feature engineering for the Stellar Classification Playground (S6E6).

Domain notes
------------
* ``u, g, r, i, z`` are SDSS photometric magnitudes.  In astronomy the
  *colours* (differences between bands) carry most of the discriminative
  signal, so we build all the adjacent and a few cross-band colours.
* ``redshift`` is by far the strongest single feature (STAR ≈ 0, GALAXY
  moderate, QSO large).  We add monotonic transforms and interactions.
* ``spectral_type`` and ``galaxy_population`` are low-cardinality
  categoricals.  We expose them both as pandas ``category`` dtype (for
  LightGBM / CatBoost native handling) and as integer codes (for XGBoost).
* **Stellar locus residuals** measure the distance of each object from
  the colour-colour locus defined by STAR rows — a classic separator
  between stars and extragalactic objects.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

TARGET = "class"
ID = "id"
CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

BANDS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]

# Project root: two levels up from this file (src/ → project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_data_path(filename: str) -> str:
    """Find a data file in either ``data/`` subfolder or project root."""
    for candidate in [
        os.path.join(_PROJECT_ROOT, "data", filename),
        os.path.join(_PROJECT_ROOT, filename),
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Cannot find '{filename}'. Place it in data/ or the project root."
    )


# ═══════════════════════════════════════════════════════════════════════
# Feature construction helpers
# ═══════════════════════════════════════════════════════════════════════

def _add_colour_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SDSS colour indices, curvatures, and band statistics.

    Colour indices (magnitude differences between bands) are the most
    physically meaningful features for stellar classification.  We also
    compute second-order colour curvatures that capture the shape of the
    spectral energy distribution (SED).
    """
    # Adjacent-band colours (classic SDSS colour indices)
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]

    # Wider baseline colours
    df["u_r"] = df["u"] - df["r"]
    df["g_i"] = df["g"] - df["i"]
    df["r_z"] = df["r"] - df["z"]
    df["u_z"] = df["u"] - df["z"]

    # Full optical slope
    df["u_minus_z"] = df["u"] - df["z"]

    # Colour curvature (2nd differences) — captures SED shape
    df["c1"] = df["u_g"] - df["g_r"]   # UV excess curvature
    df["c2"] = df["g_r"] - df["r_i"]   # optical slope change
    df["c3"] = df["r_i"] - df["i_z"]   # red slope change

    # Summary statistics over the five magnitudes
    band_vals = df[BANDS].values
    df["mag_mean"]  = band_vals.mean(axis=1)
    df["mag_std"]   = band_vals.std(axis=1)
    df["mag_min"]   = band_vals.min(axis=1)
    df["mag_max"]   = band_vals.max(axis=1)
    df["mag_range"] = df["mag_max"] - df["mag_min"]

    return df


def _add_redshift_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive redshift transforms and interactions.

    Redshift is the strongest single predictor — STARs cluster near 0,
    GALAXYs around 0.5, and QSOs above 1.9.  We add monotonic transforms
    to help tree-based models capture nonlinear breaks.
    """
    z = df["redshift"].clip(lower=-0.05)
    df["redshift_log1p"] = np.log1p(z - z.min())
    df["redshift_sq"]    = df["redshift"] ** 2

    # Redshift × brightness/colour interactions
    df["redshift_x_r"]  = df["redshift"] * df["r"]
    df["redshift_x_gr"] = df["redshift"] * df["g_r"]
    df["redshift_x_iz"] = df["redshift"] * df["i_z"]

    # Low-z indicator (STAR vs low-z GALAXY is the top confusion)
    df["is_low_z"]      = (df["redshift"] < 0.2).astype("int8")
    df["redshift_inv"]  = 1.0 / (df["redshift"].abs() + 0.05)

    return df


# ── Stellar locus (star/extragalactic separator) ─────────────────────
#
# Stars occupy a tight, nearly one-dimensional track in SDSS colour
# space; galaxies and QSOs scatter off it.  We fit each target colour
# onto a predictor colour using the STAR rows of the training set, then
# measure the residual (distance from the locus) for every row.
# This directly targets the GALAXY ↔ STAR confusion.

_LOCUS_PAIRS = [
    ("r_i", "g_r"),
    ("i_z", "r_i"),
    ("g_r", "u_g"),
    ("u_g", "g_r"),
]


def fit_stellar_locus(train: pd.DataFrame) -> dict:
    """OLS-fit each locus colour pair on the STAR rows of the training set."""
    stars = train[train[TARGET] == "STAR"]
    locus = {}
    for tgt, src in _LOCUS_PAIRS:
        x = stars[src].values
        yv = stars[tgt].values
        a, b = np.polyfit(x, yv, 1)
        locus[(tgt, src)] = (float(a), float(b))
    return locus


def _add_stellar_locus_features(df: pd.DataFrame, locus: dict) -> pd.DataFrame:
    """Add residuals from the stellar locus and an L2 distance feature."""
    resid_cols = []
    for (tgt, src), (a, b) in locus.items():
        col = f"locus_resid_{tgt}_{src}"
        df[col] = df[tgt] - (a * df[src] + b)
        resid_cols.append(col)
    df["locus_dist"] = np.sqrt((df[resid_cols] ** 2).sum(axis=1))
    return df


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, locus: dict | None = None) -> pd.DataFrame:
    """Apply all feature engineering steps to a DataFrame."""
    df = df.copy()
    df = _add_colour_features(df)
    df = _add_redshift_features(df)
    if locus is not None:
        df = _add_stellar_locus_features(df, locus)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (everything except id/target)."""
    drop = {ID, TARGET}
    return [c for c in df.columns if c not in drop]


def load_data(
    train_path: str | None = None,
    test_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load CSVs, engineer features, and return (train, test, feature_cols).

    Automatically resolves file paths — looks in ``data/`` first, then
    the project root.
    """
    if train_path is None:
        train_path = _resolve_data_path("train.csv")
    if test_path is None:
        test_path = _resolve_data_path("test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Colours first (needed to fit the stellar locus)
    train = _add_colour_features(train.copy())
    test = _add_colour_features(test.copy())

    # Fit the stellar locus on training STARs, apply to both sets
    locus = fit_stellar_locus(train)
    train = _add_redshift_features(train)
    test = _add_redshift_features(test)
    train = _add_stellar_locus_features(train, locus)
    test = _add_stellar_locus_features(test, locus)

    # Consistent categorical dtype across train/test
    for c in CAT_COLS:
        cats = sorted(set(train[c].unique()) | set(test[c].unique()))
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        train[c] = train[c].astype(dtype)
        test[c] = test[c].astype(dtype)

    feat_cols = feature_columns(train)
    return train, test, feat_cols
