# 🚗 Car Price Prediction — Project Report Plan

> **Course:** AI-DS Project (DAP391m) — FPT University  
> **Instructor:** Trần Văn Hà  
> **Team:** Tăng Toàn Thắng & Chu Thành Chuẩn  
> **Semester:** Summer 2026 (SS4 SU26)

---

## 📋 Report Structure (Draft)

The report will be written in **English** and organized into the following sections:

| # | Section | Description | Status |
|---|---------|-------------|--------|
| 1 | **Abstract / Executive Summary** | Brief overview of the project, objectives, and key results | ⬜ |
| 2 | **Introduction** | Problem statement, motivation, and project scope | ⬜ |
| 3 | **Dataset Description** | Source, size, features, and target variable | ⬜ |
| 4 | **Exploratory Data Analysis (EDA)** | Key findings, distributions, correlations, outliers | ⬜ |
| 5 | **Data Preprocessing** | Cleaning pipeline, missing values, outlier handling, rare encoding | ⬜ |
| 6 | **Feature Engineering** | New features created, rationale, and impact | ⬜ |
| 7 | **Modeling** | Models tested (Linear Regression, Random Forest, XGBoost), hyperparameters | ⬜ |
| 8 | **Evaluation & Results** | Comparison metrics (RMSE, MAE, R²), model selection | ⬜ |
| 9 | **Web Application** | Flask deployment, UI/UX, prediction flow | ⬜ |
| 10 | **Project Structure** | Code organization, module descriptions | ⬜ |
| 11 | **Challenges & Lessons Learned** | Difficulties faced, solutions, team reflections | ⬜ |
| 12 | **Conclusion & Future Work** | Summary, limitations, improvements | ⬜ |
| 13 | **References** | Libraries, datasets, academic sources | ⬜ |
| 14 | **Appendices** | Full code listings, additional charts | ⬜ |

---

## 📝 Detailed Section Outline

### 1. Abstract / Executive Summary
- One-paragraph summary of the entire project
- Problem → approach → results
- Final model: Linear Regression with R² = 0.9995, RMSE = 64.37

### 2. Introduction
- **2.1 Background:** Why car price prediction matters (used car market opacity)
- **2.2 Problem Statement:** Buyers/sellers struggle to determine fair prices
- **2.3 Objectives:**
  - Build ML model to predict car prices from 9 input features
  - Deploy as a web application for real-time predictions
- **2.4 Scope & Limitations**

### 3. Dataset Description
- **3.1 Data Source:** `car_price_dataset.csv` (~10,000 rows, 10 columns)
- **3.2 Feature Dictionary:**

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | Brand | Categorical | Car brand (Toyota, BMW, Ford, ...) |
| 2 | Model | Categorical | Car model name |
| 3 | Year | Numerical | Manufacturing year |
| 4 | Engine_Size | Numerical | Engine displacement (liters) |
| 5 | Fuel_Type | Categorical | Fuel type (Petrol, Diesel, Electric, ...) |
| 6 | Transmission | Categorical | Manual / Automatic |
| 7 | Mileage | Numerical | Kilometers driven |
| 8 | Doors | Numerical | Number of doors |
| 9 | Owner_Count | Numerical | Number of previous owners |
| 10 | **Price** | **Target** | **Car price (USD)** |

### 4. Exploratory Data Analysis (EDA)
- **4.1 Price Distribution** (right-skewed)
- **4.2 Numerical Feature Analysis** (histograms, boxplots)
- **4.3 Categorical Feature Analysis** (count plots, price by brand)
- **4.4 Correlation Matrix** (heatmap)
- **4.5 Outlier Detection** (boxplots, IQR method)

### 5. Data Preprocessing
- **5.1 Duplicate Removal**
- **5.2 Missing Value Handling** (Median for numeric, Mode for categorical)
- **5.3 Rare Encoding** (categories < 1% → "Rare")
- **5.4 Outlier Capping** (IQR method — Winsorization)
- **5.5 Data Validation** (post-cleaning summary)

### 6. Feature Engineering
- **6.1 Engine_Size_Group** (Small / Medium / Large)
- **6.2 Mileage_Group** (Low / Medium / High)
- **6.3 Year_Group** (2000s / 2010s / 2020s)
- **6.4 Car_Age** = 2026 − Year
- **6.5 Mileage_Per_Year** = Mileage ÷ Car_Age
- **6.6 Engine_Efficiency** = Mileage ÷ Engine_Size

### 7. Modeling
- **7.1 Train-Test Split** (80/20)
- **7.2 Encoding:** One-Hot Encoding for categorical features
- **7.3 Scaling:** RobustScaler (chosen over StandardScaler due to outliers)
- **7.4 Models Tested:**

| Model | Key Hyperparameters |
|-------|---------------------|
| Linear Regression | Default |
| Random Forest | n_estimators=200, random_state=42 |
| XGBoost | n_estimators=300, lr=0.05, max_depth=5 |

- **7.5 Evaluation Metrics:** RMSE, MAE, R²

### 8. Evaluation & Results
- **8.1 Model Comparison Table:**

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Linear Regression** ⭐ | **64.37** | **21.83** | **0.9995** |
| XGBoost | 128.32 | 101.32 | 0.9982 |
| Random Forest | 249.09 | 182.69 | 0.9932 |

- **8.2 Analysis:** Linear Regression surprisingly outperforms tree-based models
- **8.3 Actual vs Predicted Plot**
- **8.4 Residual Analysis**
- **8.5 Critical Note on R² = 0.9995** — possible data leakage or synthetic data

### 9. Web Application
- **9.1 Technology Stack:** Flask + Bootstrap 5 + Jinja2
- **9.2 Architecture Diagram** (data flow from user input → prediction)
- **9.3 User Interface** (screenshots)
- **9.4 Prediction Pipeline** (feature engineering → preprocessing → model inference)

### 10. Project Structure
- Directory tree with module descriptions
- Code organization principles

### 11. Challenges & Lessons Learned
- **11.1 Technical Challenges:**
  - Feature alignment between training and prediction
  - Model caching to avoid re-training
  - Requirements.txt issues
- **11.2 Team Collaboration:**
  - Task division between 2 members
  - Workflow and communication
- **11.3 Key Takeaways**

### 12. Conclusion & Future Work
- **12.1 Summary of Achievements**
- **12.2 Limitations:**
  - No cross-validation
  - No hyperparameter tuning
  - Potentially synthetic data
- **12.3 Future Improvements:**
  - Cross-validation
  - GridSearchCV
  - SHAP explainability
  - CI/CD pipeline

### 13. References
- scikit-learn, XGBoost, Flask, pandas, etc.

### 14. Appendices
- Full code listings (optional)
- Additional visualizations

---

## ✅ Known Information (Auto-extracted from project)

| Item | Detail |
|------|--------|
| Project name | Car Price Prediction |
| Final model | Linear Regression |
| Best RMSE | 64.37 |
| Best R² | 0.9995 |
| Dataset size | ~10,000 rows × 10 columns |
| Features (input) | 9 (Brand, Model, Year, Engine_Size, Fuel_Type, Transmission, Mileage, Doors, Owner_Count) |
| Target | Price (USD) |
| Models tested | Linear Regression, Random Forest, XGBoost |
| Web framework | Flask + Bootstrap 5 |
| Deployment target | Heroku (Procfile exists) |
| Languages | Python (backend), HTML/CSS (frontend) |

---

## ❓ Information Needed from the Team

Below are questions to fill in the gaps. Please answer them so I can write a complete and personalized report.

> ⚠️ **Note:** I will ask these questions in Vietnamese in the next message. This English version is for reference.
