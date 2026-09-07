# Price Of Risk - Loan Interest Rate Modeling — Final Project 

**Course:** MA 5755 — Data Analysis & Visualization

**Author:** Aswini V J (MA25M007)

This repository contains the final project work for MA5755, focused on analyzing LendingClub loan data and building predictive models for borrower interest rates.

## Project Overview

The main objective is to preprocess LendingClub loan data and compare multiple regression and machine learning approaches for predicting `int_rate` (loan interest rate). The analysis includes:
- exploratory data analysis (EDA)
- unsupervised learning and clustering experiments
- regression baselines and regularized linear models
- tree-based modeling
- neural network regression with a PyTorch MLP

## Data

The raw dataset is stored in:
- `data/raw/Loan_status_2007-2020Q3.gzip`

The processed subset generated for modeling is:
- `data/processed/lc_sample_50k_processed.csv`

### Key preprocessing steps

Implemented in `src/preprocess.py` and executed by `quick_sample.py`:
- remove rows with missing `int_rate`
- convert percentage fields to numeric values
- compute `income_to_loan` as `annual_inc / loan_amnt`
- select core loan and borrower features
- winsorize numeric inputs to reduce outliers
- drop invalid records and apply VIF-based feature selection
- sample a stable 50,000-row dataset for modeling

## Repository Structure
```
.
├── data
│   ├── processed          # generated after running quick_sample.py (excluded from Git)
│   └── raw                # place the original LendingClub gzip file here (excluded)
├── notebooks
│   ├── figures            # all generated plots (PNG)
│   ├── results            # model evaluation results (CSV)
│   ├── tables             # table images used in the report
│   ├── 01_EDA.ipynb
│   ├── 02_Unsupervised.ipynb
│   ├── 03_Regression.ipynb
│   ├── 04_Tree_Based_Methods.ipynb
│   ├── 05_MLP_PyTorch.ipynb
│   └── mlp_best.pt        # best PyTorch model weights
├── reports
│   ├── Initial_report_MA25M007.pdf
│   ├── Presentation_MA25M007.pdf
│   └── Report_MA25M007.pdf
├── src
│   └── preprocess.py      # data cleaning and VIF-based feature selection
├── README.md
└── quick_sample.py        # generates the 50k processed sample
```

## Results Summary

Model performance is compared using RMSE, MAE, and R² on the test split.

- Random Forest:  RMSE = 3.8109, MAE = 2.9126, R² = 0.3824
- MLP (PyTorch):   RMSE = 3.8337, MAE = 2.8937, R² = 0.3749
- Ridge:          RMSE = 3.8583, MAE = 2.9646, R² = 0.3669
- OLS:            RMSE = 3.8584, MAE = 2.9648, R² = 0.3669
- ElasticNet:     RMSE = 3.8587, MAE = 2.9648, R² = 0.3668
- Lasso:          RMSE = 3.8588, MAE = 2.9648, R² = 0.3667

The best predictive performance was achieved using the Random Forest model.

## Reproducibility

All analyses use a fixed random seed (`SEED = 42`) to ensure identical results.

### Step‑by‑step reconstruction

1. **Obtain the raw data:**  
   Download `Loan_status_2007-2020Q3.gzip` from [Kaggle](https://www.kaggle.com/datasets/ethon0426/lendingclub-20072020q1) and place it in `data/raw/`.

2. **Set up the Python environment:**  
   The code requires: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `torch`, `shap`.  
   If you use conda, create an environment with:

   ```bash
   conda create -n DV python=3.10 numpy pandas matplotlib seaborn scikit-learn statsmodels pytorch shap -c pytorch -c conda-forge
   conda activate DV
   ```

3. **Run preprocessing:**

   ```bash
   python quick_sample.py
   ```

   This creates `data/processed/lc_sample_50k_processed.csv` (the stratified 50k sample).
   
   *Note: The processed CSV is excluded from the repository because it is data heavy.*

4. **Execute the notebooks in order:**

   Open each notebook in Jupyter and run all cells.

   - All figures will be saved to `notebooks/figures/`
   - CSV results will be saved to `notebooks/results/`
   - The best neural network weights will be saved as `notebooks/mlp_best.pt`
## Notes

- The raw data (data/raw/) and the processed CSV (data/processed/) are excluded from Git (see .gitignore). They can be regenerated as described above.
- All figures and result CSVs are included in the repository.
- The PyTorch checkpoint mlp_best.pt is included; can be reloaded without retraining.



## Contact

Project by Aswini V J (MA25M007) – MA5755 Spring 2026.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5GgZyC9i)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23864232&assignment_repo_type=AssignmentRepo)
# Final Report of the Final Project
Please upload a 7-8 page PDF of the final report of your final project by accepting this assignment. Please also upload the final codes that you have used. See [here](https://github.com/ma5755-datavisual-iitm/MA5755/blob/main/FinalProjects/ProjectDetails_MA5755.pdf) for more details on what is expected.
