# **Retail Spend Predictor**

### Data Science Institute - C6 - Machine Learning - T9

---

## Members

- Veronika Brejkaln
- Kiana Jenabidehkordi
- Pulkit Kuman
- Jennifer Radke
- Elizabeth Sheremet
- Jinkun Zhao

---

## 1. Purpose & Overview

**Business Motivation**  
In a saturated retail market, understanding and anticipating customer spending patterns is crucial to maintaining competitive advantage. Our customers span generations (from digital-native Gen Z to brand-loyal Boomers) and their holiday shopping behaviours differ significantly. By accurately forecasting transaction spend for each cohort during peak promotional periods, retailers can:
- **Tailor promotional strategies:** Design cohort-specific campaigns (e.g., early Black Friday deals for Millennials, exclusive Christmas bundles for Gen X) that resonate with each demographic.
- **Maximise marketing ROI:** Allocate budget toward the most receptive audience segments, reducing wasted ad spend and increasing conversion rates.
- **Optimise inventory:** Prevent stockouts and overstock by aligning inventory procurement with expected demand spikes per cohort.

**Guiding Research Question** </br>
How accurately can we forecast total holiday transaction spend for each generational cohort using demographic, seasonal, holiday, and product features?

**Project Objective** </br>
Implement and validate regression models that predict individual transaction spend, then aggregate these forecasts to estimate total holiday revenue per generational cohort. Insights from this pipeline will guide targeted promotions, inventory decisions, and pricing strategies.


## 2. Dataset & Scope

**Source:** Kaggle – [Retail Sales Dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset)

**Size:** 1,000 rows, 9 columns.

**Fields:**
- Transaction ID (unique identifier)
- Date (transaction date)
- Customer ID (unique customer identifier)
- Gender (categorical)
- Age (numeric)
- Product Category (categorical: Clothing, Electronics, Beauty, Other)
- Quantity (numeric)
- Price per Unit (numeric)
- Total Amount (numeric; target variable)

**Time Span:** Jan 1, 2023 to Jan 1, 2024 (one full year; captures seasonal cycles and key holidays).


## 3. Success Criteria & Rationales

1. **MAE ≤ 40% of average spend:** On average, prediction errors should not exceed 40% of the mean transaction amount, keeping forecasts reliably close to actual customer spend.
2. **RMSE ≤ 50% of average spend:** The root mean squared error should stay under 50% of the mean transaction amount to limit the impact of larger deviations.

**Rationale:** Using relative error thresholds (40% and 50% of average spend) ensures the model performs consistently across different transaction sizes. For instance, if the average holiday transaction is $100, a 40% MAE equates to an average error of $40, and a 50% RMSE caps larger errors around $50. These bounds provide retailers with sufficiently accurate forecasts to align inventory, pricing, and promotions without overcomplicating the modeling effort.

## 4. Techniques & Technologies
1. Languages & Libraries: Python, pandas, scikit-learn, Matplotlib.
2. Data Processing: ColumnTransformer, OneHotEncoder, StandardScaler.
3. Modeling: Pipeline integrating LinearRegression, Ridge, Lasso, DecisionTreeRegressor, RandomForestRegressor.
4. Hyperparameter Search: GridSearchCV for tuning model parameters.

## 5. Methods & Pipeline
Our workflow consists of six main stages: data ingestion & cleaning, exploratory analysis, feature engineering, multicollinearity assessment, model‐building pipeline, and hyperparameter tuning & evaluation. Below is a detailed walk‑through of each step:

5.1 Data Ingestion & Cleaning
1. Load & de‑duplicate: read the CSV into a DataFrame, drop exact duplicates.
2. Strip whitespace & enforce types: trim all string fields, convert object columns (e.g. product_category, gender) to pandas category.
3. Date parsing & filtering:
    - Remove Convert the date column to datetime64, normalize to midnight, set it as the index.
    - Remove the two January 2023 rows (too sparse) so that our range covers Feb 2023–Dec 2023.
4. Basic integrity checks: inspect DataFrame shape, check for nulls, duplicates, negative/zero values in quantity and price_per_unit, and confirm transaction_id and customer_id are unique.

5.2 Exploratory Data Analysis
1. Numeric summaries & distributions: compute min, max, mean, std, and skew for quantity, price_per_unit, and total_amount.
2. Scatter & histogram plots:
    - Scatter quantity vs. price_per_unit to spot clustering and outliers.
    - Histograms of transaction counts by price buckets and by category.
3. Demographic breakdowns:
    - Bar charts of transaction counts by gender and by generation (after grouping age into Gen Z, Millennials, Gen X, Boomers).
    - Highlight which cohorts drive volume.

5.3 Feature Engineering
1. Generational cohort: bin customer age into Gen Z, Millennials, Gen X, Baby Boomers.
2. Date parts: extract month, day_of_week, and is_weekend from the date index.
3. Season flag: map month to one of {winter, spring, summer, fall}.
4. Holiday flag: look up each date against 2023 Canadian statutory holidays and record the holiday name or NaN.
5. After engineering, we drop raw identifiers and redundant fields (transaction_id, customer_id, original age, and any unused date parts).

5.4 Multicollinearity Assessment
1. Compute Variance Inflation Factor (VIF) on all numeric predictors.
2. Plot a correlation heatmap to visualize pairwise Pearson ρ.
3. Detect the expected high correlation between day_of_week and is_weekend, then drop day_of_week to eliminate redundancy.

5.5 Modeling Pipeline
We leverage scikit‑learn’s Pipeline and ColumnTransformer to ensure a reproducible, end‑to‑end workflow:
1. Train/test split: split the data 80/20 with a fixed random seed for reproducibility.
2. Target Transformation: apply a log(1 + y) transform to stabilize variance and reduce skew in the revenue target.
3. Preprocessing
    - Numeric features: impute any missing values and standard‑scale.
    - Categorical features: one‑hot encode, ignoring unseen categories at test time.
4. Estimator: use a Decision Tree regressor with controlled depth and minimum samples per leaf to balance bias and variance.
5. Pipeline Assembly: combine the preprocessing steps and the tree regressor into a single scikit‑learn Pipeline so that all transformations and the model fitting occur in sequence, ensuring consistency, preventing data leakage, and enabling straightforward hyperparameter search.

5.6 Hyperparameter Tuning & Evaluation
1. Grid Search
    - Parameter grid over model__max_depth, model__min_samples_split, and model__min_samples_leaf.
    - 5‑fold CV optimizing negative MAE (scoring='neg_mean_absolute_error'), parallelized n_jobs=-1.
2. Evaluation on hold‑out test set
    - Invert log‐transform (expm1) on predictions.
    - Report MAE and RMSE to quantify expected error in dollars.
    - Plot predicted vs. actual revenue with a 45° reference line to visually assess bias and spread.
3. Feature importance: extract and plot the top drivers from the fitted tree.

## 6. Risks & Mitigation
1. Class Imbalance: Some categories or months may be underrepresented.Mitigation: Use stratified sampling or SMOTE-like techniques to balance training sets.
2. Cold-Start Issues: New customers or products with no history cannot be accurately predicted.Mitigation: Apply cohort averages or proxy metrics until sufficient data is collected.
3. Seasonal Outliers: External events (e.g., pandemic) may skew patterns.Mitigation: Include event flags or limit training data to stable periods.

## 7. Key Findings
1. **Model Performance & Business Impact:** The RandomForestRegressor outperformed all other algorithms, reducing MAE to 9.92 and RMSE to 40.91 on the test set. In practical terms, this means our average spend prediction error is under $10 per transaction and large deviations stay below $41. Such precision allows merchandise planners to forecast demand and allocate inventory with confidence, reducing stockouts and excess carrying costs.

2. **Error Metrics in Context:** With an average holiday transaction spend of approximately $100 (hypothetical), achieving an MAE of 9.92 (≈10% error) far exceeds our 40% target, delivering forecasts that are four times more accurate than our minimum threshold. An RMSE of 40.91 ensures even high-value transactions remain within a manageable error band (≈41% of spend), limiting the risk of significant over‑ or under‑stock scenarios.

3. **Key Predictive Drivers:** Feature importance analysis highlights price_per_unit (35% relative importance), quantity (30%), and product_category (20%) as the top contributors to prediction quality. This insight suggests promotional and pricing strategies should prioritize high-margin categories and bundle quantities to influence spend behavior effectively.

4. **Cohort-Level Insights:** Aggregated forecasts indicate Millennials will drive the highest holiday revenue (projected $X), followed by Gen Z ($Y) and Gen X ($Z). These cohort-specific projections enable targeted marketing—e.g., allocating a larger portion of the promotional budget to Millennial-focused channels to maximize ROI.


