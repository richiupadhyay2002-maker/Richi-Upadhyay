# 🚀 Bitcoin Price Prediction using Machine Learning

## 📌 Project Overview

This project provides a **comprehensive end-to-end solution** for predicting Bitcoin prices using multiple machine learning, time series, and statistical models. The analysis covers **14+ years of historical data (2010–2024)** and includes EDA, feature engineering, model training, hyperparameter tuning, anomaly detection, and volatility modeling.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| **`01_eda.ipynb`** | Exploratory Data Analysis — data cleaning, statistics, visualizations, correlation analysis, time series decomposition, stationarity testing |
| **`02_feature_engineering.ipynb`** | Feature Engineering — lag feature creation, data resampling, train-test splits for regression and ARIMA models |
| **`03_model_training_evaluation.ipynb`** | Model Training & Evaluation — XGBoost, LightGBM, CatBoost (+ tuning), ARIMA, Anomaly Detection (Isolation Forest, HMM), GARCH Volatility Model |
| **`BTC-USD-Price-History-2010-2024.csv`** | Bitcoin historical price dataset (USD) |

---

## 📊 Dataset

- **Source:** Bitcoin historical prices from 2010 to 2024
- **Features:** Date, Open, High, Low, Close, Volume, Adj Close
- **Target Variable:** `High` price (daily high)
- **Observations:** ~5,000+ daily records after cleaning

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Languages** | Python 3 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | XGBoost, LightGBM, CatBoost, Scikit-learn |
| **Time Series** | ARIMA, Auto ARIMA (pmdarima), Statsmodels |
| **Anomaly Detection** | Isolation Forest, Hidden Markov Model (HMM) |
| **Volatility** | GARCH (Arch library) |
| **Environment** | Jupyter Notebooks |

---

## 📈 Models & Performance

### Models Implemented

| Category | Models | Tuning |
|----------|--------|--------|
| 📊 **Tree-Based** | XGBoost, LightGBM, CatBoost | ✅ RandomizedSearchCV |
| 📈 **Time Series** | ARIMA(5,1,0), Auto ARIMA | ✅ Auto-optimized |
| 🔍 **Anomaly Detection** | Isolation Forest, HMM | ✅ Threshold optimization |
| 📉 **Volatility** | GARCH(1,1) | ❌ Fixed order |

### Performance Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error — lower is better |
| **RMSE** | Root Mean Squared Error — lower is better |
| **R² Score** | Coefficient of Determination — higher is better (max 1.0) |

> ℹ️ **R² Interpretation:** An R² score of 0.98 means the model explains **98% of the variance** in Bitcoin prices.

---

## 🏆 Key Findings

1. **Tree-based models** (XGBoost, LightGBM, CatBoost) achieve the **highest R² scores (>0.98)** for price prediction
2. **Hyperparameter tuning** consistently improves model performance
3. **ARIMA** provides a solid baseline for time series forecasting
4. **Anomaly detection models** successfully identify price anomalies
5. **GARCH** captures volatility clustering patterns in Bitcoin returns

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- Jupyter Notebook / JupyterLab
- Required libraries (see install instructions below)

### Installation

```bash
# Navigate to the project directory
cd bitcoin_pred

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install statsmodels pmdarima arch
pip install hmmlearn
pip install jupyter nbformat
```

### Execution

Run the notebooks in order:

```bash
jupyter notebook 01_eda.ipynb
jupyter notebook 02_feature_engineering.ipynb
jupyter notebook 03_model_training_evaluation.ipynb
```

Or use JupyterLab:

```bash
jupyter lab
```

---

## 📋 Business Impact

| Application | Benefit |
|-------------|---------|
| **Price Trend Prediction** | High accuracy models (R² > 0.98) reliably predict trends |
| **Risk Management** | GARCH volatility modeling aids in risk assessment |
| **Anomaly Detection** | Identifies unusual market events and outliers |
| **Investment Strategy** | Model insights support data-driven trading decisions |

---

## 👨‍💻 Author

This project demonstrates expertise in:
- **Data Science & Machine Learning**
- **Time Series Analysis & Forecasting**
- **Statistical Modeling**
- **Anomaly Detection**
- **Data Visualization & Communication**

---

## 📄 License

This project is for educational and demonstration purposes.

---

> **⭐ If you found this project helpful, consider giving it a star!**