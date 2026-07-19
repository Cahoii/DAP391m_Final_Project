# 🚗 Car Price Prediction — Project Report

> **Course:** AI-DS Project (DAP391m) — FPT University  
> **Instructor:** Trần Văn Hà  
> **Team Members:** Tăng Toàn Thắng & Chu Thành Chuẩn  
> **Semester:** Summer 2026 (SU26)  
> **Duration:** 7 Weeks

---

## 📋 Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Dataset Description](#3-dataset-description)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Feature Engineering](#6-feature-engineering)
7. [Modeling](#7-modeling)
8. [Evaluation & Results](#8-evaluation--results)
9. [Web Application](#9-web-application)
10. [Project Structure](#10-project-structure)
11. [Project Timeline & Task Division](#11-project-timeline--task-division)
12. [Challenges & Lessons Learned](#12-challenges--lessons-learned)
13. [Conclusion & Future Work](#13-conclusion--future-work)
14. [References](#14-references)

---

## 1. Abstract

This project develops a complete **Machine Learning pipeline** to predict used car prices based on nine input features including brand, model, manufacturing year, engine size, fuel type, transmission, mileage, number of doors, and owner count. Three regression models—**Linear Regression**, **Random Forest**, and **XGBoost**—were trained and compared using RMSE, MAE, and R² metrics. The **Linear Regression** model achieved the best performance with an RMSE of **64.37 USD** and an R² of **0.9995**, demonstrating that after proper feature engineering and preprocessing, the relationship between car attributes and price is highly linear. The trained model was deployed as an interactive **Flask web application** enabling real-time price predictions. The entire workflow—from raw data ingestion to model deployment—is modularized into reusable Python modules, forming a reproducible and extensible codebase.

**Keywords:** car price prediction, machine learning, linear regression, random forest, XGBoost, feature engineering, Flask, regression

---

## 2. Introduction

### 2.1 Background

The used car market suffers from significant price opacity. Unlike new cars with manufacturer-set retail prices, used car valuations depend on a complex interplay of factors—brand prestige, vehicle age, accumulated mileage, engine specifications, and ownership history. Both buyers and sellers often lack objective tools to determine a fair market price, leading to information asymmetry and inefficient transactions.

### 2.2 Problem Statement

A prospective buyer inspecting a 5-year-old Toyota with 80,000 km on the odometer has no reliable way to know whether the asking price is reasonable. Similarly, a seller lacks data-driven justification for their listed price. This information gap creates distrust and friction in the used car marketplace.

### 2.3 Project Objectives

This project aims to:

- Build a machine learning model capable of accurately predicting used car prices from basic vehicle attributes.
- Implement a complete data pipeline covering cleaning, feature engineering, encoding, and scaling.
- Compare multiple regression algorithms to identify the best-performing approach.
- Deploy the final model as a user-friendly **web application** for real-time predictions.
- Deliver a well-documented, modular, and reproducible codebase.

### 2.4 Scope

The project focuses on regression-based price prediction using structured tabular data. The model accepts 9 categorical and numerical inputs and outputs a predicted price in USD. The dataset contains 10,000 records of used car listings.

---

## 3. Dataset Description

### 3.1 Source

The dataset was obtained from **Kaggle**: [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/sharmajicoder/used-car-price-prediction-dataset) by *sharmajicoder*. It is designed for educational and research purposes in regression modeling.

| Property | Value |
|---|---|
| **File** | `car_price_dataset.csv` |
| **Format** | CSV (Comma-Separated Values) |
| **Size** | ~574 KB |
| **Records** | 10,000 rows |
| **Features** | 9 input + 1 target = 10 columns |

### 3.2 Feature Dictionary

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `Brand` | Categorical | Car manufacturer (e.g., Toyota, BMW, Ford, Honda, Mercedes) |
| 2 | `Model` | Categorical | Specific car model name |
| 3 | `Year` | Numerical | Manufacturing year |
| 4 | `Engine_Size` | Numerical | Engine displacement in liters |
| 5 | `Fuel_Type` | Categorical | Fuel type (Petrol, Diesel, Electric, Hybrid) |
| 6 | `Transmission` | Categorical | Gearbox type (Manual, Automatic, Semi-Automatic) |
| 7 | `Mileage` | Numerical | Total kilometers driven |
| 8 | `Doors` | Numerical | Number of doors |
| 9 | `Owner_Count` | Numerical | Number of previous owners |
| 10 | **`Price`** | **Numerical (Target)** | **Car price in USD** |

### 3.3 Data Quality Snapshot

Upon initial inspection:

- **Missing values:** None detected in any column.
- **Duplicates:** No duplicate rows found.
- **Data types:** Appropriate—numerical columns stored as `float64`/`int64`, categorical as `object`.
- **Outliers:** Mild outliers observed in `Mileage` and `Price` columns (addressed during preprocessing).

---

## 4. Exploratory Data Analysis (EDA)

> Full analysis available in: `notebooks/EDA.ipynb`

### 4.1 Univariate Analysis

**Price Distribution:** The target variable `Price` exhibits an approximately normal distribution with a mean of ~8,853 USD and a median of ~8,858 USD. Values range from approximately 2,000 to 18,301 USD.

**Categorical Features:** The dataset contains 10 unique car brands, with Toyota, Ford, and BMW being the most frequent. Fuel types are dominated by Petrol (~40%), followed by Diesel (~18%). Transmission is roughly balanced between Manual and Automatic.

### 4.2 Correlation Analysis

A correlation heatmap was generated to understand linear relationships among numerical variables:

| Feature Pair | Pearson Correlation (r) | Interpretation |
|---|---|---|
| Year ↔ Price | **+0.66** | Strong positive — newer cars cost more |
| Mileage ↔ Price | **−0.55** | Strong negative — higher mileage lowers price |
| Engine_Size ↔ Price | **+0.36** | Moderate positive — larger engines command premium |
| Doors ↔ Price | ~0.00 | Negligible |
| Owner_Count ↔ Price | ~0.00 | Negligible |

### 4.3 Bivariate Analysis

- **Brand vs. Price:** Luxury brands (BMW, Mercedes, Audi) show higher median prices, while economy brands (Ford, Toyota) cluster at lower price ranges.
- **Fuel_Type vs. Price:** Electric and Hybrid vehicles command slightly higher median prices than Petrol/Diesel counterparts.
- **Mileage vs. Price:** A clear negative trend — as mileage increases, price decreases in a roughly linear fashion.
- **Year vs. Price:** A clear positive trend — newer model years fetch higher prices.

### 4.4 Outlier Detection

Boxplot analysis revealed mild outliers in `Price`, `Mileage`, and `Engine_Size`. These were not extreme and were handled through IQR-based capping during preprocessing rather than removal (to preserve sample size).

---

## 5. Data Preprocessing

The preprocessing pipeline is implemented in `utils/cleaning.py`, `utils/outlier.py`, and orchestrated via `utils/pipeline.py`.

### 5.1 Cleaning Pipeline

```
Raw Dataset (10,000 rows × 10 cols)
        │
        ▼
┌──────────────────────────┐
│ 1. Duplicate Removal     │  → No duplicates found; dataset preserved
├──────────────────────────┤
│ 2. Missing Value Handling│  → No missing values; dataset preserved
├──────────────────────────┤
│ 3. Rare Category Encoding│  → Brands/Models with <1% frequency labeled "Rare"
├──────────────────────────┤
│ 4. Outlier Capping (IQR) │  → Winsorization: values beyond [Q1−1.5·IQR, Q3+1.5·IQR] capped
├──────────────────────────┤
│ 5. Data Validation       │  → Post-cleaning summary confirms data integrity
└──────────────────────────┘
        │
        ▼
  Clean Dataset
```

### 5.2 Outlier Handling — IQR Capping

The **Interquartile Range (IQR)** method was chosen for its robustness:

$$ \text{Lower Bound} = Q1 - 1.5 \times IQR $$

$$ \text{Upper Bound} = Q3 + 1.5 \times IQR $$

Values falling outside these bounds are **capped** (not removed) to the nearest boundary. This preserves all 10,000 records while mitigating the influence of extreme values.

### 5.3 Rare Encoding

Categorical values appearing in fewer than 1% of records (e.g., exotic brands like Lamborghini, Bentley) were grouped into a single `"Rare"` category. This prevents the model from overfitting to categories with insufficient training examples.

### 5.4 Preprocessing for Modeling

| Step | Technique | Library |
|---|---|---|
| Categorical Encoding | One-Hot Encoding | `pandas.get_dummies()` |
| Numerical Scaling | RobustScaler | `sklearn.preprocessing.RobustScaler` |
| Feature Column Persistence | Pickle serialization | `joblib.dump()` |

**Why RobustScaler?** Unlike `StandardScaler`, `RobustScaler` uses the median and IQR rather than mean and standard deviation, making it less sensitive to outliers—an important consideration for car price data with its inherent skew.

### 5.5 Column Alignment

A critical technical detail: the `align_columns()` function in `utils/preprocessing.py` ensures that the feature matrix at prediction time has exactly the same columns (in the same order) as during training. This prevents silent errors when the user's single input row is transformed.

---

## 6. Feature Engineering

Feature engineering is performed by `utils/feature_engineering.py`, which creates six new features from the original data.

### 6.1 Engineered Features

| New Feature | Formula / Method | Rationale |
|---|---|---|
| `Engine_Size_Group` | Binned into Small / Medium / Large | Models learn group-level patterns rather than raw continuous values |
| `Mileage_Group` | Binned into Low / Medium / High | Categorizes usage intensity |
| `Year_Group` | Binned into 2000s / 2010s / 2020s | Captures vehicle generation |
| `Car_Age` | 2026 − `Year` | More interpretable than raw year; directly represents depreciation |
| `Mileage_Per_Year` | `Mileage` ÷ `Car_Age` | Measures annual driving intensity |
| `Engine_Efficiency` | `Mileage` ÷ `Engine_Size` | Proxy for engine wear per unit displacement |

### 6.2 Rationale

Consider two cars with identical mileage of 100,000 km:

- **Car A:** 20 years old → *Mileage_Per_Year* = 5,000 km/year (lightly used)
- **Car B:** 5 years old → *Mileage_Per_Year* = 20,000 km/year (heavily used)

Without `Mileage_Per_Year`, the model sees the same mileage value for both cars. With this feature, it can differentiate between a gently-driven older car and a heavily-used newer one—leading to more accurate price predictions.

The `Car_Age` feature replaces the raw `Year` with a more interpretable depreciation signal, while `Engine_Efficiency` captures the interaction between mileage accumulation and engine capacity.

---

## 7. Modeling

Model training is implemented in `utils/model.py` and `utils/evaluation.py`.

### 7.1 Train-Test Split

The dataset was split using an **80/20** ratio with `random_state=42` for reproducibility:

$$ X_{train}, X_{test}, y_{train}, y_{test} = \text{train\_test\_split}(X, y, \text{test\_size}=0.2, \text{random\_state}=42) $$

### 7.2 Models Tested

Three regression algorithms were selected, representing a spectrum of complexity:

| Model | Type | Key Hyperparameters |
|---|---|---|
| **Linear Regression** | Linear model | Default (OLS) |
| **Random Forest** | Ensemble (bagging) | `n_estimators=200`, `random_state=42`, `n_jobs=-1` |
| **XGBoost** | Ensemble (boosting) | `n_estimators=300`, `learning_rate=0.05`, `max_depth=5`, `objective='reg:squarederror'` |

**Why these three?** Linear Regression provides a simple, interpretable baseline; Random Forest captures non-linear interactions without extensive tuning; XGBoost represents state-of-the-art gradient boosting for tabular data.

### 7.3 Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Root Mean Squared Error — penalizes large errors; in original units (USD) |
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$ | Mean Absolute Error — average dollar deviation |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Coefficient of determination — proportion of variance explained (0 to 1) |

### 7.4 Model Persistence

To avoid redundant retraining, the system checks for existing model artifacts (`best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`) before training. If all three files exist, the trained model is loaded directly. This caching mechanism significantly accelerates development iteration and deployment.

---

## 8. Evaluation & Results

### 8.1 Model Comparison

| Model | RMSE (USD) ↓ | MAE (USD) ↓ | R² ↑ |
|---|---|---|---|
| **Linear Regression** ⭐ | **64.37** | **21.83** | **0.9995** |
| XGBoost | 128.32 | 101.32 | 0.9982 |
| Random Forest | 249.09 | 182.69 | 0.9932 |

> ⭐ **Selected Model: Linear Regression** — lowest RMSE, highest R².

### 8.2 Analysis

**Linear Regression** surprisingly outperformed both tree-based ensemble methods. This suggests that after thorough feature engineering and robust scaling, the relationship between the engineered features and car price is **predominantly linear**. The six engineered features—particularly `Car_Age`, `Mileage_Per_Year`, and `Engine_Efficiency`—effectively linearized the prediction task.

**XGBoost** ranked second with competitive metrics (R² = 0.9982), demonstrating that gradient boosting remains a strong choice for tabular regression.

**Random Forest** showed the highest error (RMSE = 249.09) among the three, though its R² of 0.9932 still indicates excellent predictive power. The higher RMSE may reflect overfitting to noise in the training data due to the relatively high number of trees (200) without depth constraints.

### 8.3 Feature Importance

Based on Random Forest feature importance analysis, the top predictors were:

1. **Engine_Efficiency** (~42%) — the strongest single predictor
2. **Year** (~22%) — newer cars consistently valued higher
3. **Car_Age** (~21%) — directly captures depreciation

These three features alone account for ~85% of predictive power, confirming the soundness of the feature engineering strategy.

### 8.4 Critical Note on R² = 0.9995

A $R^2$ value of 0.9995 is **exceptionally high** for real-world data and warrants careful scrutiny. Potential explanations include:

- **Synthetic/near-synthetic data:** The Kaggle dataset appears to have been generated with strong linear relationships between features and price, making the prediction task artificially easy.
- **Feature-target leakage:** `Engine_Efficiency` (Mileage ÷ Engine_Size) shares variance with Price in ways that may partially encode the target, though our analysis suggests this is a legitimate predictive relationship rather than true leakage.
- **Absence of noise:** Real-world car prices are influenced by unobserved factors (negotiation, location, vehicle condition, market trends). The dataset's cleanliness suggests limited real-world noise.

**Recommendation:** Future work should incorporate k-fold cross-validation and compare against external, real-world car listing data (e.g., from web scraping) to validate generalizability.

---

## 9. Web Application

### 9.1 Architecture

The trained model is served through a **Flask** web application, providing an intuitive interface for end users to obtain price predictions.

```
┌──────────────────────────────────────────────────────────┐
│                     USER BROWSER                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │              index.html (Bootstrap 5)               │  │
│  │  [Brand▼] [Model▼] [Year] [Engine_Size] ... [Submit]│  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP POST
                       ▼
┌──────────────────────────────────────────────────────────┐
│                   FLASK SERVER (app.py)                   │
│                                                          │
│  1. Parse form data (9 fields)                           │
│  2. Call predict_price(user_input)                       │
│  3. Return rendered template with prediction             │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                 PREDICTION PIPELINE (predict.py)          │
│                                                          │
│  User Input → Feature Engineering → Preprocessing        │
│  → Column Alignment → model.predict() → Price (USD)      │
└──────────────────────────────────────────────────────────┘
```

### 9.2 User Interface

The web form accepts all 9 input features:

| Field | Input Type | Notes |
|---|---|---|
| Brand | Dropdown | Dynamically populated from dataset |
| Model | Dropdown | **Filtered by selected Brand** (cascading) |
| Year | Number input | Integer |
| Engine_Size | Number input | Float (liters) |
| Fuel_Type | Dropdown | Petrol, Diesel, Electric, Hybrid |
| Transmission | Dropdown | Manual, Automatic, Semi-Automatic |
| Mileage | Number input | Float (kilometers) |
| Doors | Number input | Integer |
| Owner_Count | Number input | Integer |

**UX features:**
- Model dropdown is filtered based on the selected Brand, preventing invalid combinations.
- Form retains previously entered values after submission for easy iteration.
- Prediction result is displayed prominently below the form.

### 9.3 Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3, Flask 3.0 |
| **ML Libraries** | scikit-learn, XGBoost, pandas, numpy, joblib |
| **Frontend** | HTML5, CSS3, Bootstrap 5, Jinja2 Templates |
| **Visualization** | Matplotlib, Seaborn (EDA notebook) |
| **Deployment Target** | Heroku (Procfile configured; currently running on localhost) |

### 9.4 Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (first run only)
python train.py

# Start the web server
python app.py

# Open browser at http://127.0.0.1:5000
```

---

## 10. Project Structure

```
CarPricePrediction/
│
├── config.py                     # Centralized paths and constants
├── train.py                      # Main training entry point
├── predict.py                    # Prediction module (used by Flask)
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
├── Procfile                      # Heroku deployment configuration
│
├── data/
│   └── car_price_dataset.csv     # Raw dataset (10,000 × 10)
│
├── notebooks/
│   ├── EDA.ipynb                 # Exploratory Data Analysis
│   └── Model_Training_annotated.ipynb  # Model training & evaluation
│
├── utils/
│   ├── __init__.py               # Package marker
│   ├── data_loader.py            # CSV loading with path resolution
│   ├── cleaning.py               # Data cleaning pipeline
│   ├── outlier.py                # IQR-based outlier capping
│   ├── eda.py                    # Column type detection helper
│   ├── feature_engineering.py    # Create 6 new features
│   ├── preprocessing.py          # One-Hot Encoding + RobustScaler + alignment
│   ├── model.py                  # Train, compare, save models
│   ├── evaluation.py             # RMSE, MAE, R² computation and storage
│   └── pipeline.py               # End-to-end `prepare_dataset()` orchestrator
│
├── models/
│   ├── best_model.pkl            # Serialized trained model
│   ├── scaler.pkl                # Fitted RobustScaler
│   └── feature_columns.pkl       # Column names for prediction alignment
│
├── results/
│   ├── model_metrics.csv         # Comparison table of all models
│   └── predictions.csv           # Predictions on test set
│
├── templates/
│   └── index.html                # Flask HTML template
│
└── static/
    └── css/
        └── style.css             # Custom styles
```

### Design Principles

- **Modularity:** Each `utils/*.py` module has a single responsibility, making the codebase easy to understand, test, and extend.
- **Reusability:** The `preprocess_data(training=True/False)` function serves both training and prediction pipelines, ensuring consistency.
- **Configurability:** All paths and constants are centralized in `config.py`, avoiding hardcoded values.
- **Reproducibility:** Fixed `random_state=42` across all splits and models; model artifacts are serialized for exact reproduction of results.

---

## 11. Project Timeline & Task Division

### 11.1 Timeline (7 Weeks)

```
Week 1  ■  Understanding the Problem & Data Collection
         • Define business requirements
         • Source dataset from Kaggle
         • Establish data collection methodology

Week 2  ■  Data Preprocessing
         • Check for missing values
         • Handle outliers (IQR capping)
         • Standardize data formats
         • Split train/test sets

Week 3  ■  Data Analysis & Visualization
         • Validate data integrity
         • Analyze relationships between variables
         • Identify outliers
         • Visualize distributions with Matplotlib/Seaborn
         • Create dashboards with Plotly

Weeks 4–5 ■  Model Building
         • Train Linear Regression, Random Forest, XGBoost
         • Iterate on feature engineering
         • Tune hyperparameters

Week 6  ■  Model Evaluation
         • Compare models using RMSE, MAE, R²
         • Select optimal model (Linear Regression)
         • Analyze residuals and feature importance

Week 7  ■  Finalization
         • Write project documentation
         • Prepare final report and presentation
         • Schedule demo session
```

### 11.2 Task Division

| Member | Primary Responsibilities |
|---|---|
| **Tăng Toàn Thắng** | Data preparation, data cleaning, feature engineering, model training, model evaluation, pipeline orchestration |
| **Chu Thành Chuẩn** | Exploratory Data Analysis (EDA), data visualization, Flask web application development, frontend UI/UX, project documentation |

Both members collaborated on code integration, testing, and presentation preparation.

---

## 12. Challenges & Lessons Learned

### 12.1 Data Quality Challenges

The raw dataset—while complete (no missing values)—contained **inconsistent formatting** and **mild outliers** that required careful preprocessing. Rare car brands with very few samples necessitated rare encoding to prevent the model from learning spurious patterns. The team learned that real-world (or realistic synthetic) data rarely arrives in modeling-ready condition, and that a robust cleaning pipeline is as important as the model itself.

### 12.2 Technical Challenges

| Challenge | Solution |
|---|---|
| Feature dimension mismatch between training and prediction (single user input has different shape) | Implemented `align_columns()` to ensure exact column order and handle missing one-hot columns |
| Repeated model retraining during development | Added model existence check (`model_exists()`) to load cached artifacts |
| Unrelated dependencies in `requirements.txt` (e.g., Django, gym) | Audited and regenerated `requirements.txt` with only project-relevant packages |
| `round()` returning float (e.g., `12345.0`) instead of integer | Applied `int()` conversion after rounding for clean price display |

### 12.3 Key Takeaways

1. **Feature engineering can transform a problem.** The six engineered features—especially `Car_Age` and `Mileage_Per_Year`—were decisive in enabling Linear Regression to outperform tree-based models.
2. **RobustScaler is the right choice for skewed data.** Using median/IQR-based scaling prevented outliers from distorting the feature space.
3. **Column alignment is critical in production.** The single most subtle bug in ML deployment is feature mismatch between training and inference. Explicit alignment logic prevents silent failures.
4. **Modular code pays dividends.** Separating concerns into `utils/` modules made debugging, testing, and iteration dramatically faster.
5. **High R² demands skepticism.** An R² of 0.9995 is a red flag that should prompt cross-validation and leakage analysis before claiming success.

---

## 13. Conclusion & Future Work

### 13.1 Summary

This project successfully delivered an end-to-end car price prediction system:

- ✅ A complete **data preprocessing pipeline** handling rare encoding, outlier capping, and robust scaling.
- ✅ **Six engineered features** that captured domain knowledge about vehicle depreciation and usage patterns.
- ✅ **Three regression models** trained and compared, with **Linear Regression** selected as the best performer (RMSE = 64.37 USD, R² = 0.9995).
- ✅ A **Flask web application** with cascading dropdowns and an intuitive user interface.
- ✅ A fully **modular, documented, and reproducible** codebase.

### 13.2 Limitations

| Limitation | Impact |
|---|---|
| No k-fold cross-validation | Model performance estimated from a single 80/20 split; may not generalize |
| Default/custom hyperparameters | Models not optimized via GridSearchCV or RandomizedSearchCV |
| Potentially synthetic dataset | High R² may not translate to real-world car listings |
| No logging framework | `print()` statements used throughout; insufficient for production monitoring |
| No unit tests | No automated verification of pipeline correctness |
| Local-only deployment | Not yet accessible via public URL |

### 13.3 Future Improvements

| Priority | Task | Benefit |
|---|---|---|
| 🔴 High | Implement k-fold cross-validation | Reliable generalization estimate |
| 🔴 High | Investigate data leakage / synthetic nature of data | Confidence in real-world applicability |
| 🟡 Medium | Hyperparameter tuning with GridSearchCV | Potential accuracy gains for tree-based models |
| 🟡 Medium | Replace `print()` with Python `logging` module | Production-grade observability |
| 🟡 Medium | Add error handling in Flask routes | User-friendly error messages instead of HTTP 500 |
| 🟢 Low | Deploy to Heroku / Render | Public demo availability |
| 🟢 Low | Add SHAP analysis for model explainability | Build user trust through interpretable predictions |
| 🟢 Low | Write unit tests with `pytest` | Regression safety net for future changes |
| 🟢 Low | AJAX-based dynamic model dropdown | Improved UX for large brand-model combinations |
| 🟢 Low | CI/CD pipeline for automated retraining | Fresh predictions as new data arrives |

---

## 14. References

### Dataset

> sharmajicoder. *Used Car Price Prediction Dataset.* Kaggle.  
> https://www.kaggle.com/datasets/sharmajicoder/used-car-price-prediction-dataset

### Libraries & Tools

| Reference | Citation |
|---|---|
| **scikit-learn** | Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. |
| **XGBoost** | Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*, 785–794. |
| **Flask** | Ronacher, A. (2010). Flask — A Python Microframework. https://flask.palletsprojects.com/ |
| **pandas** | McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51–56. |
| **NumPy** | Harris, C. R., et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362. |
| **Matplotlib** | Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering, 9*(3), 90–95. |
| **Seaborn** | Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software, 6*(60), 3021. |
| **Joblib** | Varoquaux, G., et al. (2017). Joblib: lightweight pipelining in Python. https://joblib.readthedocs.io/ |
| **Bootstrap** | Otto, M., & Thornton, J. (2011). Bootstrap — Frontend Framework. https://getbootstrap.com/ |

---

> 📝 **Report prepared by:** Tăng Toàn Thắng & Chu Thành Chuẩn  
> 📅 **Date:** July 2026  
> 🏫 **FPT University — AI-DS Project (DAP391m)**
