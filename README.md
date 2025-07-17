<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Work+Sans:wght@400;600&display=swap');

:root {
  --font-heading: 'Cinzel', serif;
  --font-body: 'Work Sans', sans-serif;
  --font-code: 'Fira Code', monospace;
}

/* Body text */
body, p, li {
  font-family: var(--font-body);
}

/* Headings */
h1, h2, h3, h4 {
  font-family: var(--font-heading);
}

/* Code blocks and inline code */
code, pre {
  font-family: var(--font-code);
}

/* Tables */
table {
  font-family: var(--font-body);
  border-collapse: collapse;
}
table th, table td {
  border: 1px solid #ccc;
  padding: 0.5em;
}
</style>

# Holiday‑Weekend High‑Value Transaction Prediction

## 1. Purpose & Overview

This project demonstrates how to predict “high‑value” customer transactions in the two weeks prior to Black Friday, enabling targeted holiday incentives:

- **Free gift** for the top 25% of predicted spenders
- **10% off coupon** for the remaining customers

By accurately forecasting the highest spenders using only demographics and time‑of‑purchase features, we estimate incremental holiday‑weekend revenue uplift, inform incentive strategy, and showcase an end‑to‑end data science pipeline.

---

## 2. Goals & Objectives

1. **Define high‑value transactions**

   - Calculate the 75th percentile of `Total_Amount` in the holiday window (2023‑11‑10 to 2023‑11‑23)
   - Create binary label `is_high`: 1 for top 25% spenders, 0 otherwise

2. **Build predictive models**

   - Baseline: Logistic Regression
   - Advanced: Random Forest, Gradient Boosting Machine
   - Evaluate with ROC AUC, precision/recall, lift curves

3. **Simulate revenue uplift**

   - Assign incentives based on predicted probabilities
   - Compare incremental revenue under multiple `(α₁, α₂, p*)` scenarios

4. **Deliverables**

   - Reproducible scripts/notebooks for each pipeline stage
   - A `run_all.sh` orchestrator for end‑to‑end execution
   - Slide deck summarizing methods, results, and business impact

---

## 3. Data Overview

- **Source**: `data/raw/retail_sales_dataset.csv` (see [data dictionary](#data-dictionary))
- **Timeframe**: 2023‑01‑01 to 2024‑01‑01 (all transactions)
- **Key columns**:

| Column            | Type     | Description                        |
| ----------------- | -------- | ---------------------------------- |
| `TransactionID`   | string   | Unique transaction identifier      |
| `Date`            | datetime | Purchase timestamp                 |
| `CustomerID`      | string   | Customer identifier (all unique)   |
| `Age`             | integer  | Customer age in years              |
| `Gender`          | category | M / F                              |
| `ProductCategory` | category | E.g. Electronics, Beauty, Clothing |
| `Quantity`        | integer  | Units purchased                    |
| `PricePerUnit`    | float    | Price per item                     |
| `Total_Amount`    | float    | Quantity × PricePerUnit            |

### 3.1 Holiday Window Definition

- **Black Friday 2023**: 2023‑11‑24
- **Holiday window**: All purchases with `Date` ≥ 2023‑11‑10 and < 2023‑11‑24

---

## 4. Feasibility Assessment

| Aspect                                     | Feasible? | Notes / Caveats                                                                      |
| ------------------------------------------ | --------- | ------------------------------------------------------------------------------------ |
| **Label definition** (top 25% spenders)    | ✔️        | Compute 75th percentile of `Total_Amount` in holiday window                          |
| **Features** (Age, Gender, Category, time) | ✔️        | All derivable—no missing values                                                      |
| **Modeling** (binary classification)       | ✔️        | Logistic Regression, Random Forest, GBM                                              |
| **Holiday window**                         | ✔️        | Filter by date                                                                       |
| **Control group / uplift measurement**     | ◼️        | No true A/B data for gift vs. coupon; uplift must be simulated using assumed factors |

---

## 5. Pipeline & Prototype Steps

### 5.1 Holiday Window Filtering & Labeling

```python
import pandas as pd

df = pd.read_csv('data/raw/retail_sales_dataset.csv', parse_dates=['Date'])
BF_DATE = pd.Timestamp('2023-11-24')

# Filter
mask = (df.Date >= '2023-11-10') & (df.Date < '2023-11-24')
df_holiday = df.loc[mask].copy()

# Threshold & Label
threshold = df_holiday.Total_Amount.quantile(0.75)
df_holiday['is_high'] = (df_holiday.Total_Amount >= threshold).astype(int)
```

### 5.2 Feature Engineering

```python
# Demographics
df_holiday['Age_bin'] = pd.cut(df_holiday.Age, bins=[18,25,35,45,55,100], labels=False)

# One-hot categories
df_holiday = pd.get_dummies(df_holiday, columns=['ProductCategory'], prefix='cat')

# Temporal features
df_holiday['day_of_week'] = df_holiday.Date.dt.dayofweek
df_holiday['hour_of_day'] = df_holiday.Date.dt.hour
df_holiday['days_to_bf'] = (BF_DATE - df_holiday.Date).dt.days

# Final feature set
feature_cols = ['Age_bin','Gender_M','Gender_F','day_of_week','hour_of_day','days_to_bf'] + \
               [c for c in df_holiday if c.startswith('cat_')]
X = df_holiday[feature_cols]
y = df_holiday['is_high']
```

### 5.3 Train/Test Split & Modeling

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Baseline
logit = LogisticRegression(max_iter=1000)
logit.fit(X_train, y_train)

# Evaluate
y_pred_proba = logit.predict_proba(X_test)[:,1]
print('ROC AUC:', roc_auc_score(y_test, y_pred_proba))
```

### 5.4 Revenue Simulation

```python
import numpy as np

def simulate_revenue(df, model, α1, α2, p_thresh):
    probs = model.predict_proba(df[feature_cols])[:,1]
    df = df.copy()
    df['incentive'] = np.where(probs >= p_thresh, 'gift', 'coupon')
    df['rev_sim'] = np.where(
        df['is_high']==1,
        (1+α1) * df['Total_Amount'],
        (1+α2) * df['Total_Amount'] * 0.90
    )
    return df['rev_sim'].sum()

# Example
revenue = simulate_revenue(df_holiday, logit, α1=0.10, α2=0.05, p_thresh=0.55)
print('Simulated Revenue:', revenue)
```

### 5.5 Evaluation & Reporting

- **Metrics**: ROC AUC, Precision\@25%, Recall\@25%
- **Charts**: ROC curve, feature importances, uplift comparison
- **Sensitivity Analysis**: Vary α₁, α₂, p\* to show impact on incremental revenue

---

## 6. Limitations & Assumptions

1. **No true A/B test data**: uplift factors α₁, α₂ are **assumed** or drawn from literature.
2. **Small sample** (\~1 000 holiday transactions): high model variance—report 95% CI.
3. **Unique customers**: cannot track repeat behavior; model predicts one‑off spending only.
4. **External validity**: results may not generalize outside the 2023 holiday window.

---

## 7. How to Reproduce

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Feature engineering & labeling
python src/features.py \
  --input data/raw/retail_sales_dataset.csv \
  --output data/processed/holiday.pkl

# 3. Train & evaluate models
python src/train.py \
  --data data/processed/holiday.pkl \
  --model-dir models/

# 4. Simulate uplift
python src/simulate.py \
  --model models/rf.pkl \
  --alpha-gift 0.10 \
  --alpha-coupon 0.05 \
  --p-thresh 0.55

# 5. Generate report
python src/report.py \
  --results reports/metrics.json \
  --output reports/summary.pdf

# Or simply run all steps:
./run_all.sh
```

---

## 8. Contributors & Credits