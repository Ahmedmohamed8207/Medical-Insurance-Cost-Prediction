# Medical Insurance Cost Prediction with Machine Learning 🏥📉

An end-to-end Machine Learning project to predict individual annual medical healthcare expenses using demographic, physical, and behavioral features.

---

## 📌 Project Overview
Healthcare cost forecasting is critical for both insurance providers and policyholders. This project builds a predictive regression pipeline on a dataset of 1,338 records, examining how habits and physical attributes (such as smoking status and BMI) drive claim amounts, while diagnosing underfitting/overfitting and applying regularization techniques[cite: 6].

---

## 🔍 Key Pipeline Stages & Methodology
- **Exploratory Data Analysis (EDA):** Analyzed expense distributions and confirmed smoking status and BMI as the dominant drivers of high charges[cite: 6].
- **Feature Engineering:** Created domain interaction terms (`smoker_bmi`, `age_bmi`, `bmi_squared`) to capture non-linear health interactions[cite: 6].
- **Leakage-Free Preprocessing Pipelines:** Built automated processing using `ColumnTransformer` (`StandardScaler` for numeric features, `OneHotEncoder` for categoricals) fitted strictly on training splits[cite: 6].
- **Model Experimentation & Regularization:**
  - Evaluated baseline Linear Regression models[cite: 6].
  - Tested Polynomial Regression up to degree 8 to diagnose model complexity and the bias-variance tradeoff[cite: 6].
  - Implemented `Ridge` and `Lasso` regularization with hyperparameter tuning ($\alpha$) to prevent overfitting and ensure strong generalization[cite: 6].
- **Model Evaluation:** Benchmarked models using MAE, RMSE, and $R^2$ Score metrics[cite: 6].

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Machine Learning:** `scikit-learn`
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook

---

## 📂 Repository Structure
```text
├── Medical_Insurance_Prediction.ipynb   # Complete analysis, pipelines & modeling
├── insurance.csv                        # Dataset
├── README.md                            # Documentation
└── LICENSE                              # MIT License
