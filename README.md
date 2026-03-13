# time-series-sales-forecast


Time series forecasting of weekly sales across 45 Walmart stores using statistical and machine learning approaches.

---

## Project Overview

Retail demand forecasting is critical for inventory planning and business decision-making. This project builds an end-to-end forecasting pipeline on Walmart's historical weekly sales data (2010–2012), covering 45 stores and 6,435 records. Multiple forecasting models are implemented, evaluated, and compared.

---

## Dataset

- **Source:** Walmart Store Sales dataset
- **Records:** 6,435 rows × 8 columns
- **Features:** Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
- **Time Range:** February 2010 – October 2012
- **Stores:** 45 unique stores, 143 weeks each

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Checked for null values and duplicates (none found)
- Correlation heatmap across all numeric features
- Weekly sales distribution by store, month, week, and year
- Identified top-performing stores (Store 20: ~$301M) and worst-performing (Store 33: ~$37M)

### 2. Stationarity Testing
- **ADF Test** — confirmed stationarity (p < 0.05)
- **KPSS Test** — flagged non-stationarity, indicating trend/seasonality
- Applied rolling mean detrending (window = 52 weeks) and seasonal decomposition to prepare series for ARIMA modelling

### 3. Models Implemented

| Model | Scope | Key Result |
|-------|-------|------------|
| **Prophet** | All 45 stores | MASE ≈ 0.048, R² ≈ 0.53 (Store 1) |
| **ARIMA (auto_arima)** | Store 1 | Best order: ARIMA(1,0,1), AIC = 2421 |
| **SARIMA(1,0,1)(1,0,1,52)** | All 45 stores | MASE ≈ 0.29, R² ≈ 0.55 (Store 1) |

### 4. Forecasting
- Generated 12–52 week forecasts per store using both Prophet and SARIMA
- Automated pipeline loops across all 45 stores
- Visualised actual vs forecast for every store

---

## Results

**Prophet (Store 1 — hold-out evaluation):**
- MAE: 75,542
- RMSE: 106,468
- MASE: 0.048
- R²: 0.53

**SARIMA (Store 1 — test set):**
- MAE: 60,987
- RMSE: 82,811
- MASE: 0.29
- R²: 0.55

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, matplotlib, seaborn, statsmodels, prophet, pmdarima, scikit-learn

---

## Project Structure

```
walmart-sales-forecasting/
│
├── walmart_prophet.ipynb        # Prophet forecasting pipeline
├── walmart_arima_sarima.ipynb   # ARIMA/SARIMA modelling and EDA
├── Walmart.csv                  # Dataset
└── README.md
```

---

## Key Findings

- Sales exhibit strong **yearly seasonality**, with consistent spikes around the holiday season (weeks 47–51)
- **Holiday weeks** show measurably higher sales across most stores
- Store size and type significantly affect baseline sales levels — Store 20 outperforms Store 33 by nearly 8x
- **SARIMA slightly outperforms Prophet** on RMSE for Store 1, but Prophet generalises better across stores with fewer convergence issues

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet pmdarima scikit-learn
```

Open either notebook and ensure `Walmart.csv` is in the same directory. Run all cells sequentially.
