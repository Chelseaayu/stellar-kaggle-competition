"""End-to-end training pipeline for the Stellar Classification competition.

Pipeline
--------
1. Load data and engineer features (``features.py``).
2. Train 3 gradient-boosting models (LightGBM, XGBoost, CatBoost) with
   stratified K-fold CV and early stopping.  Collect out-of-fold (OOF)
   probabilities and averaged test probabilities for each model.
3. Optimise a non-negative blend of the model OOF probabilities to
   maximise balanced accuracy.
4. Optimise per-class posterior multipliers for balanced accuracy
   (``metrics.py``).
5. Write two final submissions:
     * submission_1.csv  – tuned blend + tuned class weights (aggressive)
     * submission_2.csv  – equal blend + inverse-prior weights (robust)

Run
---
    python src/train.py                 # full strength (5 folds)
    python src/train.py --folds 3       # faster
    python src/train.py --quick         # tiny smoke test on a subsample
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score  # noqa: F401 (kept for per-fold reporting)
from lightgbm import early_stopping, log_evaluation

import features as F
import models as M
import metrics as MET

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Project root: one level up from src/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project_path(filename: str) -> str:
    """Resolve a filename relative to the project root."""
    return os.path.join(_PROJECT_ROOT, filename)


# --------------------------------------------------------------------------- #
# Per-model cross-validated training
# --------------------------------------------------------------------------- #
def run_lgbm(X, y, X_test, cat_cols, folds, seed):
    oof = np.zeros((len(X), M.N_CLASSES))
    test_pred = np.zeros((len(X_test), M.N_CLASSES))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = M.make_lgbm(seed)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="multi_logloss",
            categorical_feature=cat_cols,
            callbacks=[early_stopping(150), log_evaluation(0)],
        )
        oof[va] = model.predict_proba(X.iloc[va])
        test_pred += model.predict_proba(X_test) / folds
        sc = balanced_accuracy_score(y[va], oof[va].argmax(1))
        print(f"  [LGBM] fold {fold}  bal_acc={sc:.5f}  best_iter={model.best_iteration_}")
    return oof, test_pred


def run_xgb(X, y, X_test, folds, seed, gpu=False):
    oof = np.zeros((len(X), M.N_CLASSES))
    test_pred = np.zeros((len(X_test), M.N_CLASSES))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = M.make_xgb(seed, gpu=gpu)
        model.set_params(early_stopping_rounds=150)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            verbose=False,
        )
        oof[va] = model.predict_proba(X.iloc[va])
        test_pred += model.predict_proba(X_test) / folds
        sc = balanced_accuracy_score(y[va], oof[va].argmax(1))
        print(f"  [XGB ] fold {fold}  bal_acc={sc:.5f}  best_iter={model.best_iteration}")
    return oof, test_pred


def run_cat(X, y, X_test, cat_idx, folds, seed, gpu=False):
    oof = np.zeros((len(X), M.N_CLASSES))
    test_pred = np.zeros((len(X_test), M.N_CLASSES))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = M.make_cat(seed, gpu=gpu)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=(X.iloc[va], y[va]),
            cat_features=cat_idx,
            early_stopping_rounds=200,
            verbose=False,
        )
        oof[va] = model.predict_proba(X.iloc[va])
        test_pred += model.predict_proba(X_test) / folds
        sc = balanced_accuracy_score(y[va], oof[va].argmax(1))
        print(f"  [CAT ] fold {fold}  bal_acc={sc:.5f}  best_iter={model.get_best_iteration()}")
    return oof, test_pred


# --------------------------------------------------------------------------- #
# Blend weight optimisation (Dirichlet random search, then class weights once)
# --------------------------------------------------------------------------- #
def optimise_blend(y, oof_list, n_iter=300, seed=0):
    """Search a non-negative simplex blend of model OOF probabilities.

    To keep this fast on 577k rows we score blends with a *fixed*
    inverse-prior class weighting during the search, then refine the
    per-class multipliers once on the winning blend.
    """
    rng = np.random.default_rng(seed)
    n = len(oof_list)

    priors = np.bincount(y, minlength=MET.N_CLASSES) / len(y)
    fixed_w = 1.0 / np.clip(priors, 1e-6, None)
    fixed_w = fixed_w / fixed_w.sum() * MET.N_CLASSES

    def quick_score(w):
        blend = sum(wi * o for wi, o in zip(w, oof_list))
        return MET.balanced_acc(y, blend, fixed_w)

    best_w = np.ones(n) / n
    best_s = quick_score(best_w)
    # always consider each single model on its own
    for i in range(n):
        w = np.zeros(n); w[i] = 1.0
        s = quick_score(w)
        if s > best_s:
            best_s, best_w = s, w
    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(n) * 1.0)
        s = quick_score(w)
        if s > best_s:
            best_s, best_w = s, w

    # refine per-class multipliers once on the winning blend
    blend = sum(wi * o for wi, o in zip(best_w, oof_list))
    class_w, final_s = MET.optimise_weights(y, blend, n_rounds=4, grid=41)
    return best_w, class_w, final_s


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="subsample for a fast smoke test")
    ap.add_argument("--blend-iter", type=int, default=400)
    ap.add_argument("--gpu", action="store_true", help="use GPU for XGBoost and CatBoost")
    ap.add_argument("--from-cache", action="store_true",
                    help="skip training; reuse oof_predictions.npz to redo blending/submissions")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading data and building features ...")
    train, test, feat_cols = F.load_data()

    if args.quick:
        train = train.sample(n=40000, random_state=RANDOM_STATE).reset_index(drop=True)
        args.folds = 3
        args.blend_iter = 500
        print(f"QUICK mode: subsampled train to {len(train)} rows")

    y = train[F.TARGET].map(F.CLASS_TO_INT).values
    X = train[feat_cols].copy()
    X_test = test[feat_cols].copy()

    cat_cols = F.CAT_COLS
    cat_idx = [feat_cols.index(c) for c in cat_cols]
    print(f"n_features={len(feat_cols)}  train={X.shape}  test={X_test.shape}")
    print(f"features: {feat_cols}")

    # CatBoost prefers string categoricals
    Xc = X.copy()
    Xtc = X_test.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str)
        Xtc[c] = Xtc[c].astype(str)

    cache = _project_path("oof_predictions.npz")
    if args.from_cache:
        print(f"Loading cached OOF/test predictions from {cache} ...")
        d = np.load(cache)
        oof_lgb, oof_xgb, oof_cat = d["oof_lgb"], d["oof_xgb"], d["oof_cat"]
        te_lgb, te_xgb, te_cat = d["te_lgb"], d["te_xgb"], d["te_cat"]
    else:
        print("\n=== LightGBM ===")
        oof_lgb, te_lgb = run_lgbm(X, y, X_test, cat_cols, args.folds, RANDOM_STATE)
        print("\n=== XGBoost ===")
        oof_xgb, te_xgb = run_xgb(X, y, X_test, args.folds, RANDOM_STATE, gpu=args.gpu)
        print("\n=== CatBoost ===")
        oof_cat, te_cat = run_cat(Xc, y, Xtc, cat_idx, args.folds, RANDOM_STATE, gpu=args.gpu)
        # persist immediately so the (cheap) blend step never risks the
        # expensive model training again.
        np.savez_compressed(
            cache, y=y, oof_lgb=oof_lgb, oof_xgb=oof_xgb, oof_cat=oof_cat,
            te_lgb=te_lgb, te_xgb=te_xgb, te_cat=te_cat,
        )
        print(f"\nSaved model OOF/test predictions to {cache}")

    # ---- single-model OOF balanced accuracy (argmax + tuned weights) ----
    print("\n=== Single-model OOF balanced accuracy ===")
    for name, oof in [("LGBM", oof_lgb), ("XGB", oof_xgb), ("CAT", oof_cat)]:
        raw = MET.fast_balanced_acc(y, oof.argmax(1))
        w, s = MET.optimise_weights(y, oof)
        print(f"  {name}: raw={raw:.5f}  weighted={s:.5f}")

    # ---- optimise blend ----
    print("\n=== Optimising blend weights ===")
    oof_list = [oof_lgb, oof_xgb, oof_cat]
    te_list = [te_lgb, te_xgb, te_cat]
    blend_w, class_w, blend_s = optimise_blend(y, oof_list, n_iter=args.blend_iter)
    print(f"  blend weights (LGBM,XGB,CAT) = {np.round(blend_w,4)}")
    print(f"  class  weights              = {np.round(class_w,4)}")
    print(f"  blended OOF balanced acc     = {blend_s:.5f}")

    # ---- build test predictions ----
    blend_te = sum(w * o for w, o in zip(blend_w, te_list))

    # Submission 1: tuned blend + tuned class weights (aggressive)
    pred1 = MET.weighted_argmax(blend_te, class_w)

    # Submission 2: equal blend + inverse-prior weights (robust hedge)
    eq_te = sum(o for o in te_list) / len(te_list)
    priors = np.bincount(y, minlength=M.N_CLASSES) / len(y)
    inv_w = 1.0 / priors
    inv_w = inv_w / inv_w.sum() * M.N_CLASSES
    pred2 = MET.weighted_argmax(eq_te, inv_w)

    # report what submission 2 would have scored OOF too
    eq_oof = sum(o for o in oof_list) / len(oof_list)
    s2 = MET.fast_balanced_acc(y, MET.weighted_argmax(eq_oof, inv_w))
    print(f"  robust (eq blend + inv-prior) OOF balanced acc = {s2:.5f}")

    # ---- write submissions ----
    # Look for sample_submission.csv in data/ or project root
    for _p in [_project_path("data/sample_submission.csv"),
               _project_path("sample_submission.csv")]:
        if os.path.isfile(_p):
            sub = pd.read_csv(_p)
            break
    else:
        raise FileNotFoundError("Cannot find sample_submission.csv")
    sub1 = sub.copy()
    sub1["class"] = [F.INT_TO_CLASS[i] for i in pred1]
    sub1.to_csv(_project_path("submission_1.csv"), index=False)

    sub2 = sub.copy()
    sub2["class"] = [F.INT_TO_CLASS[i] for i in pred2]
    sub2.to_csv(_project_path("submission_2.csv"), index=False)

    print(f"\nWrote {_project_path('submission_1.csv')} (tuned blend)")
    print(f"  and {_project_path('submission_2.csv')} (robust).")
    print("submission_1 class distribution:")
    print(pd.Series(sub1["class"]).value_counts(normalize=True).round(4).to_dict())
    print("submission_2 class distribution:")
    print(pd.Series(sub2["class"]).value_counts(normalize=True).round(4).to_dict())

    # update cache with the chosen blend / class weights for reproducibility
    np.savez_compressed(
        cache,
        y=y, oof_lgb=oof_lgb, oof_xgb=oof_xgb, oof_cat=oof_cat,
        te_lgb=te_lgb, te_xgb=te_xgb, te_cat=te_cat,
        blend_w=blend_w, class_w=class_w,
    )
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
