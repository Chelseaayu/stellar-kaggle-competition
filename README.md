<p align="center">
  <h1 align="center">🔭 Stellar Classification</h1>
  <p align="center">
    <strong>Kaggle Playground Series — Season 6, Episode 6</strong>
    <br />
    Classifying celestial objects (Galaxy, Star, QSO) with gradient boosting ensembles
    <br /><br />
    <a href="https://www.kaggle.com/competitions/playground-series-s6e6">
      <img src="https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle"/>
    </a>
    <img src="https://img.shields.io/badge/Balanced_Accuracy-96.68%25-brightgreen" alt="Score"/>
    <img src="https://img.shields.io/badge/Rank-%23293-blue" alt="Rank"/>
    <img src="https://img.shields.io/badge/Python-3.10+-yellow?logo=python&logoColor=white" alt="Python"/>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"/>
    </a>
  </p>
</p>

---

## 📋 Overview

This repository contains my solution for the [Kaggle Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6) competition. The goal is to predict whether a celestial object is a **Galaxy**, **Star**, or **QSO** (quasi-stellar object) using photometric and spectral data inspired by the [Sloan Digital Sky Survey (SDSS)](https://www.sdss.org/).

**Competition metric**: Balanced Accuracy (mean per-class recall)

### Individual Model Performance (OOF)

| Model | Raw Balanced Acc | Weighted Balanced Acc | Blend Weight |
|-------|------------------|-----------------------|--------------|
| LightGBM | 0.95662 | 0.96559 | 72.6% |
| XGBoost | 0.95574 | 0.96516 | 17.3% |
| CatBoost | 0.95446 | 0.96437 | 10.1% |
| **Ensemble** | — | **0.96572** | — |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA                                 │
│              577K rows × SDSS photometric bands                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                           │
│  • SDSS colour indices (u-g, g-r, r-i, i-z, wide-baseline)    │
│  • Colour curvatures (SED shape)                                │
│  • Redshift transforms & interactions                           │
│  • Stellar locus residuals (star/extragalactic separator)       │
│  • Band statistics (mean, std, min, max, range)                 │
│                         60+ features                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌────────────┐┌────────────┐┌────────────┐
       │  LightGBM  ││  XGBoost   ││  CatBoost  │
       │  5-fold CV ││  5-fold CV ││  5-fold CV │
       │  w = 0.726 ││  w = 0.173 ││  w = 0.101 │
       └─────┬──────┘└─────┬──────┘└─────┬──────┘
             │             │             │
             └─────────────┼─────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               WEIGHTED PROBABILITY BLEND                        │
│         Dirichlet random search on OOF predictions              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         PER-CLASS DECISION WEIGHT OPTIMISATION                  │
│    Coordinate-ascent search for Bayes-optimal class weights     │
│            (maximises balanced accuracy directly)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
           ┌─────────────┐    ┌─────────────┐
           │submission_1  │    │submission_2  │
           │(tuned blend) │    │(robust hedge)│
           └─────────────┘    └─────────────┘
```

## 🧪 Feature Engineering

### Colour Indices
The most physically meaningful features for stellar classification. Colour indices are differences between SDSS photometric bands (`u`, `g`, `r`, `i`, `z`):

| Feature Group | Examples | Why It Works |
|---------------|----------|--------------|
| Adjacent colours | `u-g`, `g-r`, `r-i`, `i-z` | Classic SDSS colour indices that separate stellar types |
| Wide-baseline | `u-r`, `u-z`, `g-i` | Capture broader spectral slopes |
| Curvatures | `(u-g)-(g-r)`, `(g-r)-(r-i)` | SED shape — UV excess for QSOs |

### Redshift Transforms
Redshift is the single strongest predictor:
- **STAR** ≈ 0 (no cosmological redshift)
- **GALAXY** ≈ 0.5
- **QSO** ≈ 1.9

We add `log1p`, `squared`, `inverse`, and interaction terms with colours and magnitudes.

### Stellar Locus Residuals
Stars occupy a tight, nearly 1D track in colour-colour space. We fit OLS lines through the STAR training data in multiple colour-colour planes, then measure each object's distance from this locus. This directly targets the **Galaxy ↔ Star** confusion (the hardest boundary).

## 🔧 Two Submission Strategy

Kaggle allows selecting **2 submissions** for final judging. To hedge against leaderboard shake-up:

| File | Strategy | Class Weights |
|------|----------|---------------|
| `submission_1.csv` | Tuned blend + tuned per-class weights | Optimised |
| `submission_2.csv` | Equal-weight blend + inverse-prior weights | Bayesian default |

## 📁 Project Structure

```
stellar-kaggle/
├── README.md              ← You are here
├── requirements.txt       ← Python dependencies
├── LICENSE                ← MIT License
├── .gitignore
│
├── src/
│   ├── features.py        ← Feature engineering + data loading
│   ├── models.py          ← LightGBM / XGBoost / CatBoost factories
│   ├── metrics.py         ← Balanced accuracy + class weight optimisation
│   ├── train.py           ← End-to-end training pipeline + submissions
│   ├── eda.py             ← Exploratory data analysis
│   └── error_analysis.py  ← Post-training confusion analysis
│
└── data/                  ← Download from Kaggle (not committed)
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## 🚀 How to Reproduce

### 1. Clone & Setup

```bash
git clone https://github.com/Chelseaayu/stellar-kaggle-competition.git
cd stellar-kaggle-competition
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):
 
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
```

### 2. Download Data

Download the competition data from [Kaggle](https://www.kaggle.com/competitions/playground-series-s6e6/data) and place the CSV files in the `data/` directory (or the project root).

```bash
# Using the Kaggle CLI:
kaggle competitions download -c playground-series-s6e6
unzip playground-series-s6e6.zip -d data/
```

### 3. Run EDA (Optional)

```bash
python src/eda.py
```

### 4. Train & Generate Submissions

```bash
# Full run (5 folds, ~17 min on GPU)
python src/train.py --folds 5

# Quick smoke test (3 folds, 40K subsample, ~2 min)
python src/train.py --quick

# Enable GPU for XGBoost + CatBoost
python src/train.py --folds 5 --gpu

# Reuse cached OOF predictions (skip training, redo blending)
python src/train.py --from-cache
```

### 5. Error Analysis (Optional)

```bash
python src/error_analysis.py
```

## 💡 Key Takeaways

1. **Domain knowledge matters** — even in tabular ML. Understanding that SDSS colour indices are astrophysically meaningful led to better features than blind feature generation.

2. **Post-prediction calibration is underrated.** Optimising per-class decision weights added **+0.9%** balanced accuracy beyond raw model predictions — more than any individual hyperparameter change.

3. **Ensembles still work.** The weighted blend consistently outperformed every individual model, with LightGBM contributing the most (73% weight).

4. **The stellar locus is a powerful separator.** Measuring distance from the star colour-colour locus directly targets the Galaxy ↔ Star confusion, which is the hardest classification boundary.

## 📚 References

- [Kaggle Playground S6E6](https://www.kaggle.com/competitions/playground-series-s6e6)
- [SDSS Photometric System](https://www.sdss.org/dr18/imaging/photometry/)
- [Stellar Classification Dataset (original)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
