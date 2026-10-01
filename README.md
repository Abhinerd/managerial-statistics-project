# Productivity Prediction in Garment Manufacturing: Multiple Linear Regression

**Course:** Managerial Statistics  
**Objective:** Develop and evaluate an Ordinary Least Squares (OLS) Multiple Linear Regression model to predict manufacturing line productivity, assess overall and individual feature significance, and apply backward elimination for model optimization.

## Dataset
- **Source:** [Kaggle - Productivity Prediction of Garment Employees](https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees)
- **File:** `data/garments_worker_productivity.csv`

## Repository Structure
- `data/`: Contains the raw dataset.
- `src/regression_analysis.py`: End-to-end Python script fitting the initial full model, testing significance, executing backward elimination, and printing diagnostic summaries.
- `report/`: Complete written assignment report addressing Questions 1 through 4.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/regression_analysis.py