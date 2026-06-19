# DSC-Football-Match-Prediction

End-to-end machine learning pipeline that predicts the **exact final score**
(home goals + away goals) of international football matches, optimized
against a custom asymmetric metric rather than plain regression error.

Built as a portfolio project to demonstrate time-series-safe feature
engineering, dual gradient-boosting model design, and metric-driven ensemble
tuning — not just "fit a model and submit."

---

## Why this isn't a standard regression problem

Most football-prediction writeups stop at "predict goals with XGBoost." Two
things make this pipeline different:

1. **The target is two correlated counts, not one.** `team_goals` and
   `opp_goals` are modeled as two separate regressors per algorithm (4 models
   total), because attack-side features (form, attack strength) and
   defense-side features (defense pressure, opponent's recent scoring) don't
   carry the same weight for each target.
2. **The scoring metric isn't MAE.** Competitions/evaluators in this space
   care about getting the *match outcome* (win/draw/loss) right at least as
   much as the *raw goal count* — a prediction that's off by one goal on each
   side but flips a win into a loss is worse than one that nails the outcome
   with the same raw error. See `notebooks/04_ensemble_evaluation.ipynb` for
   the full **AW-MAE** metric definition and why the rounding step explicitly
   forces draws under a tuned threshold.

## Pipeline overview

| # | Notebook | What happens |
|---|----------|---------------|
| 01 | [`data_preprocessing`](notebooks/01_data_preprocessing.ipynb) | Domain-aware missing value handling, confederation-grouped imputation for macro stats, outlier correction, categorical encoding — with EDA on missingness, target distribution, and the Elo↔outcome relationship |
| 02 | [`feature_engineering`](notebooks/02_feature_engineering.ipynb) | 34 leak-proof, time-aware features: rolling form (`shift(1)` before every rolling window), Elo ratios, attack/defense proxies, rest-days advantage, exponentially time-decayed head-to-head history, tournament importance weighting |
| 03 | [`model_training`](notebooks/03_model_training.ipynb) | Chronological 80:20 split, LightGBM + CatBoost trained independently per target, with learning curves, feature importance, and a model-agreement check that justifies ensembling at all |
| 04 | [`ensemble_evaluation`](notebooks/04_ensemble_evaluation.ipynb) | The AW-MAE metric implementation, blend-weight grid search optimized directly against it, residual analysis, and an outcome confusion matrix |
| 05 | [`dynamic_forecasting`](notebooks/05_dynamic_forecasting.ipynb) | Sequential, match-by-match feature reconstruction for genuinely future fixtures — state (form, H2H, rest days) updates after every prediction, the way a real scheduled inference job would run |

## Architecture at a glance

```
Raw match feed
      │
      ▼
01 Preprocessing  ──►  confederation-aware imputation, outlier fixes, encoding
      │
      ▼
02 Feature Eng.   ──►  34 leak-proof features (rolling form, Elo, H2H decay, rest days)
      │
      ▼
03 Model Training ──►  LightGBM(team) + LightGBM(opp) + CatBoost(team) + CatBoost(opp)
      │
      ▼
04 Ensemble Eval. ──►  AW-MAE-optimized blend weight + draw-forcing rounding threshold
      │
      ▼
05 Forecasting    ──►  sequential state-updating inference loop → submission.csv
```

## A note on the data

The original model was trained on a historical international-football dataset
(2005–2024, 30k+ matches) that isn't published in this repo for size/licensing
reasons. `generate_dummy_data.py` produces a **synthetic dataset with the
identical schema** (same columns, similar missingness patterns, Elo-driven
goal simulation) so every notebook here runs end-to-end out of the box and
every chart you see is a real, freshly rendered output — not a placeholder.

To run on your own data, just point `generate_dummy_data.py`'s output paths
at your real `train.csv` / `test.csv` with matching columns, or swap the data
loading cell in `01_data_preprocessing.ipynb`.

## Getting started

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt

# generate the synthetic dataset (or drop in your own train.csv / test.csv)
python generate_dummy_data.py

# run the notebooks in order
jupyter notebook notebooks/
```

Run `01` → `02` → `03` → `04` → `05` in sequence — each notebook reads the
output of the one before it.

## Project structure

```
.
├── generate_dummy_data.py        # synthetic data generator (schema-matched)
├── requirements.txt
├── data/                         # train.csv, test.csv + intermediate cleaned tables
├── models/                       # serialized LightGBM / CatBoost models
├── outputs/                      # final submission.csv
└── notebooks/
    ├── plot_style.py             # shared chart theme
    ├── 01_data_preprocessing.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_model_training.ipynb
    ├── 04_ensemble_evaluation.ipynb
    └── 05_dynamic_forecasting.ipynb
```

## Possible next steps

- Replace fixed LightGBM/CatBoost hyperparameters with a tracked Optuna study
  (the current values reflect an earlier search, not re-run in these
  notebooks for reproducibility/runtime reasons)
- Add a proper time-series cross-validation loop (multiple chronological
  folds) instead of a single 80:20 split, to get a confidence interval on
  AW-MAE rather than a point estimate
- Persist `state_2011.pkl`-style snapshots automatically at the end of each
  training run instead of rebuilding state from the cleaned CSV in notebook 05


## Pipeline Architecture

This project is divided into 6 modular stages to maintain code cleanliness and ensure experiment reproducibility. You can run each stage directly in your browser using Google Colab:
1. **Data Preprocessing & Cleaning** 
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ravitirta/DSC-Football-Match-Prediction/blob/main/notebooks/01_data_preprocessing.ipynb)
2. **Feature Engineering**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ravitirta/DSC-Football-Match-Prediction/blob/main/notebooks/02_feature_engineering.ipynb)
3. **Model Training**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ravitirta/DSC-Football-Match-Prediction/blob/main/notebooks/03_model_training.ipynb)
4. **Ensemble Evaluation**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ravitirta/DSC-Football-Match-Prediction/blob/main/notebooks/04_ensemble_evaluation.ipynb)
5. **Dynamic Forecasting** 
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ravitirta/DSC-Football-Match-Prediction/blob/main/notebooks/05_dynamic_forecasting.ipynb)
