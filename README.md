# House Prices Regression Study (`sdc-activity`) — Applied ML Notebook

> Exploratory data analysis and supervised regression comparison (`LinearRegression` vs. `RandomForestRegressor`) on the Ames Housing dataset (`dataset/train.csv`), built with Python, pandas, scikit-learn, Matplotlib, and Jupyter.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)

---

## Executive Overview

**`sdc-activity`** presents a comparative regression workflow in `Untitled.ipynb` predicting residential `SalePrice` from selected structural and lot features (`OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `FullBath`, `YearBuilt`, `LotArea`) in `dataset/train.csv`.

> **Repository Maintenance & CI Status (`main` vs. Open Draft PR #2 — `maintenance/repository-quality-improvements`):**
> - **PR #2** removes tracked `.ipynb_checkpoints` artifacts (`Untitled-checkpoint.ipynb`, `dataset/.ipynb_checkpoints/test-checkpoint.csv`) from version control, adds Python/Jupyter `.gitignore` rules, adds automated notebook JSON / Python AST / dataset CSV integrity tests (`tests/test_notebook_integrity.py`), and adds `.github/workflows/quality.yml` (verified passing in CI).

---

## Implemented Pipeline (`Untitled.ipynb`)

1. **Dataset Loading & Inspection:** Reads `dataset/train.csv`, inspects shape, schema info, summary statistics (`SalePrice`), and null counts across the 7 selected numeric features (`df[features]`).
2. **Train/Test Split:** Splits features and target (`test_size=0.2`, `random_state=42`).
3. **Model Training:** Fits both `sklearn.linear_model.LinearRegression` and `sklearn.ensemble.RandomForestRegressor(random_state=42)`.
4. **Evaluation & Visualization:** Compares **MAE**, **RMSE**, and **$R^2$** scores in a summary table (`pd.DataFrame`) and plots actual vs. predicted `SalePrice` values with `matplotlib.pyplot`.

---

## Directory Structure

```text
sdc-activity/
├── .github/workflows/quality.yml                 # Compile & notebook/CSV integrity CI (PR #2)
├── dataset/
│   ├── data_description.txt                      # Ames Housing feature codebook
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── tests/
│   └── test_notebook_integrity.py                # Notebook JSON, AST & CSV header tests (PR #2)
├── Untitled.ipynb                                # Primary analysis & regression notebook
├── report.docx                                   # Academic project report
├── requirements.txt                              # pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, jupyter
└── README.md
```

---

## Setup & Verification

```bash
git clone https://github.com/StratagemVX/sdc-activity.git
cd sdc-activity

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook Untitled.ipynb
```

### Verification Commands (PR #2 / CI `quality.yml`)
```bash
python -m compileall -q tests
python -m unittest discover -s tests -v
```

---

## Contributor Attribution & Profile

- **Engineer:** Chethan Kumar B S (`StratagemVX`)
- **GitHub:** [https://github.com/StratagemVX](https://github.com/StratagemVX)
- **Portfolio:** [https://chethan-kumar-portfolio.vercel.app/](https://chethan-kumar-portfolio.vercel.app/)
- **LinkedIn:** [https://linkedin.com/in/chethan-kumar-b-s-781a002bb](https://linkedin.com/in/chethan-kumar-b-s-781a002bb)
