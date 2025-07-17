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

**Customer-Category Predictor**

A simple classifier that predicts which product category a customer will buy next, enabling personalized recommendations and smarter marketing outreach.

---

## Guiding Research Question

- **How accurately can we predict a customer’s next purchase category using transaction date, demographics, and order details to drive personalized recommendations?**

---

## 1. Purpose & Overview

**Business Motivation**  
E‑commerce platforms and retail apps rely on relevant product recommendations to boost engagement, conversion rates, and average order value. Our Customer‑Category Predictor will forecast the next product category a given customer is most likely to purchase, allowing businesses to:

- Surface a “Products you may like” section on websites and apps.  
- Send targeted emails or messages with appropriate discounts, offers, or product suggestions.  
- Increase customer satisfaction and lifetime value by personalizing the shopping experience.

**Project Objective**  
Build and evaluate a machine learning classifier that, given a set of customer and transaction features, outputs probabilities for four categories: **Clothing**, **Electronics**, **Beauty**, and **Other**.


## 2. Business Question & Success Criteria

- **Primary Question:** Which product category will this customer purchase next?  
- **Key Metric:** Achieve at least **60% accuracy** on the held‑out test set and calibrate predicted probabilities so they reflect true purchase likelihoods.


## 3. Goals & Objectives

1. **Exploratory Data Analysis:**  
   - Identify patterns and trends in transaction dates, demographics, and purchase quantities.  
   - Check for missing values, outliers, and class imbalance.

2. **Feature Engineering:**  
   - Transform **Date** into **Month** and **Season**.  
   - Create flags or bins for **Age** and **Gender**.  
   - Normalize **Quantity** and **Price per Unit**; optionally derive additional features (e.g., purchase frequency).

3. **Model Development:**  
   - Compare algorithms (e.g., Logistic Regression, Random Forest, ?).  
   - Use cross‑validation and hyperparameter tuning (GridSearchCV) to select the best model.

4. **Evaluation & Validation:**  
   - Split the data into Train/Validation/Test sets.  
   - Evaluate performance metrics (Accuracy, Precision, Recall, ROC‑AUC).  
   - Analyze confusion matrix and probability calibration.

5. **Documentation & Reproducibility:**  
   - Provide a Jupyter notebook or script that reproduces the entire pipeline.  
   - Write clear instructions in **README.md** for setup and execution.

6. **Team Video Reflections (Portfolio Asset):**  
   - Each member records a 3–5 minute video answering:  
     - What they learned.  
     - Challenges faced and how they were overcome.  
     - Future improvements and individual strengths.


## 4. Techniques & Technologies Techniques & Technologies

- **Languages & Libraries:** Python, pandas, NumPy, scikit-learn, Matplotlib  
- **Modeling:** Logistic Regression, Random Forest, ?  
- **Preprocessing:** scikit-learn Pipelines, One‑Hot Encoding, StandardScaler
- **Version Control:** Git  
- **Reproducible Environment:** ?


## 5. Plan & Tasks

**Day 1 (Wednesday)**  
- **Project Kickoff:** Create repository, set up initial structure, establish collaboration protocols.

**Day 2 (Thursday)**  
- **EDA & Data Cleaning:** Perform data summary, analyze missing values, and handle outliers.  
- **README Review:** Review and update README sections for clarity and completeness.
- **Assign Roles** Decide who's doing what + back-ups.

**Day 3 (Tuesday)**  
- **Feature Engineering:** Transform `Date` into `Month`/`Season`, encode `Age` and `Gender`, and normalize continuous features.

**Day 4 (Wednesday)**  
- **Initial Modeling:** Build baseline classifier (Logistic Regression), evaluate initial performance, and document results.

**Day 5 (Thursday)**  
- **Documentation & Reporting:** Draft/update Purpose, Goals, and Dataset sections in README; comment code and organize project files.  
- **Video Reflections Setup:** Outline individual video reflection prompts and assign recording responsibilities.


