# 🚀 Bitcoin Price Prediction using Machine Learning

## 📌 Project Overview

This project provides an end-to-end pipeline for predicting Bitcoin prices using tree-based ML models and time series methods. The analysis covers 14+ years of historical data (2010–2024) and includes EDA, feature engineering, model training, hyperparameter tuning, anomaly detection, and volatility modeling.

> ⚠️ **Note on this README**: The performance numbers below were corrected on 2 September 2026 to match the actual output of `03_model_training_evaluation.ipynb`. An earlier version of this README significantly overstated the results (claiming R² > 0.98) — see [Corrected vs. Original](#-corrected-vs-original) at the bottom.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `01_eda.ipynb` | Exploratory Data Analysis |
| `02_feature_engineering.ipynb` | Lag features, resampling, train-test splits |
| `03_model_training_evaluation.ipynb` | XGBoost, LightGBM, CatBoost (+ tuning), ARIMA, anomaly detection, GARCH |
| `Bitcoin_Price_Dataset_2014_2023.csv` | Historical price dataset |

---

## 📊 Dataset

- **Source:** Bitcoin historical prices, 2010–2024
- **Observations:** ~5,266 daily records (4,212 train / 1,054 test for regression models; 4,236 train / 1,060 test for ARIMA)
- **Target Variable:** Daily `High` price

---

## 📈 Models & Performance

*(actual output of `03_model_training_evaluation.ipynb`, run 2 September 2026)*

### Base Models

| Model | MSE | RMSE | R² Score |
|-------|-----|------|----------|
| **CatBoost** | 1.068 × 10⁸ | $10,336.65 | **0.8088** |
| XGBoost | 1.073 × 10⁸ | $10,356.25 | 0.8081 |
| LightGBM | 1.077 × 10⁸ | $10,376.49 | 0.8073 |

### Tuned Models

| Model | R² Score | Change vs. Base |
|-------|----------|------------------|
| CatBoost (Tuned) | 0.8081 | -0.09% |
| XGBoost (Tuned) | 0.8032 | -0.60% |
| LightGBM (Tuned) | 0.8015 | -0.71% |

> Tuning via RandomizedSearchCV did **not** improve performance on this dataset — every tuned model performed marginally worse than its base version. This is a real and useful finding, not a failure to report: it suggests the base hyperparameters were already close to optimal, or the search space/budget needs revisiting.

### Time Series Baseline

| Model | R² Score |
|-------|----------|
| ARIMA(5,1,0) | -0.03 |

ARIMA underperformed a naive forecast on this data — tree-based models with engineered features substantially outperformed the pure time-series baseline.

**Best model: CatBoost (Base) — R² = 0.81, RMSE ≈ $10,337.**

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Languages | Python 3 |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, LightGBM, CatBoost, Scikit-learn |
| Time Series | ARIMA, Auto ARIMA (pmdarima), Statsmodels |
| Anomaly Detection | Isolation Forest, Hidden Markov Model (HMM) |
| Volatility | GARCH (Arch library) |

---

## 🏆 Key Findings

1. Tree-based models (CatBoost, XGBoost, LightGBM) explain **~81% of price variance** (R² ≈ 0.81) — solid, but not near-perfect.
2. Hyperparameter tuning did not improve results on this dataset — worth investigating rather than assuming tuning always helps.
3. ARIMA alone is a weak baseline here; engineered features matter more than the time-series structure alone.
4. Anomaly detection (Isolation Forest, HMM) and GARCH volatility modeling are included in the pipeline as complementary risk-management tools, separate from the price-prediction accuracy above.

---

## 🚀 How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install statsmodels pmdarima arch hmmlearn
jupyter notebook 01_eda.ipynb
jupyter notebook 02_feature_engineering.ipynb
jupyter notebook 03_model_training_evaluation.ipynb
```

---

## ✅ Corrected vs. Original

| Metric | Original README | Actual Notebook Output |
|--------|-----------------|--------------------------|
| Best model R² | "> 0.98" | **0.81** (CatBoost) |
| Tuning impact | Implied improvement | **Tuning slightly hurt performance** on all 3 models |
| ARIMA | Listed as a baseline, no result given | **R² = -0.03** — underperformed a naive forecast |

The original README's headline claim (R² > 0.98) did not match the notebook's actual printed output. This version reflects what the code actually produced, including results that don't flatter the project — that's the more useful and defensible version for anyone (including a recruiter) who reads the code.

---

## 👤 Author

**Richi Upadhyay** — Data Science & Machine Learning
