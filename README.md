# House Prices Advanced Regression Techniques — Machine Learning Predictive Modeling

> An end-to-end supervised machine learning study predicting residential property valuations through exploratory data analysis, missing data imputation, skewness correction, and regularized regression models.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)

---

## Executive Overview

This repository houses a comprehensive applied data science and machine learning regression study based on the Ames Housing dataset. It demonstrates a complete predictive modeling lifecycle: handling 79 diverse explanatory variables, applying domain-informed feature engineering, log-normalizing target distributions, and training regularized linear and ensemble models.

---

## Machine Learning Pipeline & Methodology

### 1. Exploratory Data Analysis & Target Normalization
The target variable (`SalePrice`) displays significant right-skewness and heteroscedasticity. A logarithmic transformation is applied:

$$y^* = \log(1 + \text{SalePrice})$$

This centers the residual variance and aligns with the competition's Root-Mean-Squared-Error (RMSE) evaluation metric.

### 2. Feature Engineering & Preprocessing
- **Missing Data Imputation:** Categorical variables where `NA` indicates absence (e.g. `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`) are encoded with dedicated absence tokens.
- **Categorical Encoding:** One-hot encoding of nominal variables and ordinal mapping of quality ratings (Ex, Gd, TA, Fa, Po).
- **Correlation Filtering:** Identifying high multicollinearity pairs (e.g., `GarageCars` vs. `GarageArea`) to reduce variance.

### 3. Model Architecture & Benchmarking
- **Ridge Regression ($L_2$ Regularization):** Shrinks coefficients of correlated features.
- **Lasso Regression ($L_1$ Regularization):** Enforces feature sparsity by penalizing redundant parameters.
- **ElasticNet:** Balances $L_1$ and $L_2$ penalties for robust generalization on small sample sizes.
- **Gradient Boosting Regressor:** Non-linear decision tree ensemble modeling complex variable interactions.

---

## Directory Structure

```text
sdc-activity/
├── dataset/
│   ├── train.csv            # 1,460 residential records with ground-truth SalePrice
│   ├── test.csv             # 1,459 test records for evaluation
│   ├── sample_submission.csv# Benchmark submission format
│   └── data_description.txt # Official Ames Housing codebook & data dictionary
├── Untitled.ipynb           # Complete Jupyter Notebook containing end-to-end ML pipeline
├── report.docx              # Formal academic project documentation & analysis report
├── requirements.txt         # Pinned reproducible dependencies
└── README.md                # Master repository technical documentation
```

---

## Setup & Notebook Execution

```bash
# Clone the repository
git clone https://github.com/StratagemVX/sdc-activity.git
cd sdc-activity

# Set up virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Untitled.ipynb
```

---

## Contributor Attribution & Profile

- **Engineer:** Chethan Kumar B S (`StratagemVX`)
- **Institution:** Department of Computer Science & Engineering, MVJ College of Engineering
- **GitHub:** [https://github.com/StratagemVX](https://github.com/StratagemVX)
- **Portfolio:** [https://chethan-kumar-portfolio.vercel.app/](https://chethan-kumar-portfolio.vercel.app/)
- **LinkedIn:** [https://linkedin.com/in/chethan-kumar-b-s-781a002bb](https://linkedin.com/in/chethan-kumar-b-s-781a002bb)

## Verification and maintenance

Use the source and dependency manifest as the authority for features and operational commands. This project handles user-facing data; verify authorization and privacy behavior when changing data flows.

### Build and test commands

No root package manifest was available during this pass. Select commands from the actual language-specific manifest and CI configuration.

### Contribution and security notes

- Keep features, architecture diagrams, screenshots, and technology details aligned with implementation.
- Do not commit secrets, private keys, production data, or local environment files; use placeholder-only examples and deployment-managed secrets.
- Validate authentication, authorization, input validation, and access policies when modifying user data.
- Include reproduction steps and expected/actual behavior in issue reports.
- Run relevant build, lint, and test checks before merging.
