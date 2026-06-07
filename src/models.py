"""Model factories for the three gradient-boosting learners.

Each factory returns an *untrained* estimator configured for multiclass
classification with predicted probabilities.  Hyper-parameters are tuned
for a large (~577k row) tabular problem and favour generalisation
(moderate depth, strong subsampling, L2 regularisation).
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
N_CLASSES = 3


def make_lgbm(seed: int = RANDOM_STATE) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_bin=255,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )


def make_xgb(seed: int = RANDOM_STATE, gpu: bool = False) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        tree_method="hist",
        device="cuda" if gpu else "cpu",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=seed,
        enable_categorical=True,
    )


def make_cat(seed: int = RANDOM_STATE, gpu: bool = False) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=4000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=4.0,
        random_strength=1.0,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        border_count=254,
        random_seed=seed,
        task_type="GPU" if gpu else "CPU",
        devices="0" if gpu else None,
        allow_writing_files=False,
        verbose=False,
    )
