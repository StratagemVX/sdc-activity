# House Prices Advanced Regression Techniques — Machine Learning Predictive Modeling

> End-to-end supervised machine learning regression pipeline predicting residential property sale prices using exploratory data analysis, feature engineering, and regularized ensemble models.

---

## 1. Project Title
**House Prices Advanced Regression Techniques** (`sdc-activity`) — Machine Learning Exploratory Data Analysis & Predictive Modeling Activity.

## 2. Project Overview
This repository contains a comprehensive data science and machine learning regression study based on the Ames Housing dataset. It demonstrates a complete predictive modeling lifecycle: handling high-cardinality categorical variables, imputing missing values with domain heuristics, log-normalizing skewed target distributions, and training regularized linear and gradient-boosted regressors.

## 3. Problem Statement
Accurately predicting residential home valuations is challenging due to the interplay of 79 explanatory variables describing structural attributes, location qualities, and zoning categories. Linear models underfit complex non-linear interactions, while raw decision trees overfit small sample sizes. This project establishes an optimal balance using cross-validated regularization.

## 4. Key Pipeline Stages
- **Data Ingestion & Quality Audit:** Ingesting 1,460 training records across 79 distinct residential characteristics.
- **Missing Data Imputation:** Specialized imputation strategies for structural features (e.g. `PoolQC`, `MiscFeature`, `Alley`).
- **Target Skew Correction:** Applying $\log(1 + x)$ transformations to correct right-skewed `SalePrice` distributions.
- **Model Training & Benchmarking:** Comparing Ridge Regression, Lasso ($L_1$), ElasticNet, and Gradient Boosted Decision Trees.
- **Evaluation Metric:** Scored against Root-Mean-Squared-Error (RMSE) between logarithmic predicted and observed prices.

## 5. Technology Stack
- **Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, SciPy
- **Data Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook (`Untitled.ipynb`)

## 6. Folder Structure
```
sdc-activity/
├── dataset/
│   ├── train.csv            # 1,460 residential records with SalePrice labels
│   ├── test.csv             # 1,459 test records for evaluation
│   ├── sample_submission.csv# Formatting benchmark
│   └── data_description.txt # Official Ames Housing codebook & data dictionary
├── Untitled.ipynb           # Primary Jupyter Notebook with complete ML pipeline
├── report.docx              # Formal academic project documentation & analysis report
├── requirements.txt         # Pinned reproducible dependencies
└── README.md                # Enterprise technical repository documentation
```

## 7. Installation & Notebook Execution
```bash
# Clone the repository
git clone https://github.com/StratagemVX/sdc-activity.git
cd sdc-activity

# Set up virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Untitled.ipynb
```

## 8. Author & Project Information
- **Engineer:** Chethan Kumar B S (`StratagemVX`)
- **Institution:** MVJ College of Engineering
- **Domain:** Machine Learning, Predictive Regression & Applied Data Science
