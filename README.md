# Credit Worthiness Predictor

A simple machine learning project that predicts whether a loan will be fully paid or not using a Decision Tree classifier on the Lending Club-style dataset (columns like `purpose`, `fico`, `int.rate`, etc.). The workflow is implemented in the Jupyter notebook `Credit_Worthiness.ipynb` and includes data exploration, preprocessing, modeling, evaluation, and basic hyperparameter exploration.

## Project Structure
- `Credit_Worthiness.ipynb`: Main notebook with EDA, preprocessing, model training, and evaluation
- `loan_data (1).csv`: Dataset file used by the notebook
- `requirements.txt`: Python dependencies to reproduce the environment (to be generated)
- `README.md`: This documentation

## Setup
1. Create and activate a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data
The notebook expects a CSV with columns including:
- `credit.policy`, `purpose`, `int.rate`, `installment`, `log.annual.inc`, `dti`, `fico`, `days.with.cr.line`, `revol.bal`, `revol.util`, `inq.last.6mths`, `delinq.2yrs`, `pub.rec`, `not.fully.paid`

The provided dataset file is `loan_data (1).csv` located in the repository root. In the original notebook, the data path points to a Google Drive location. To run locally, change the CSV path in the notebook to:
```python
df = pd.read_csv('loan_data (1).csv')
```

## Running the Notebook
1. Launch Jupyter:
```bash
python -m ipykernel install --user --name=credit-worthiness
jupyter notebook
```
2. Open `Credit_Worthiness.ipynb` and run the cells top-to-bottom. Ensure the data path is correctly set as shown above.

## What the Notebook Does
- Loads the dataset into a pandas DataFrame
- Encodes the categorical `purpose` column using `LabelEncoder`
- Performs EDA: histograms, pairplots, count plots, and boxplots
- Splits the dataset into train/test sets
- Trains a `DecisionTreeClassifier`
- Evaluates accuracy and prints a confusion matrix
- Explores hyperparameters (`criterion`, `max_depth`) to observe accuracy trends

## Results (example)
Using default settings in the notebook, typical outputs observed:
- Accuracy around ~74% (baseline)
- Best observed accuracy ~84% at certain `max_depth` values (on the test split)

## Reproducibility Notes
- A fixed `random_state=83` is used for the train/test split.
- Visualizations use seaborn/matplotlib and may vary slightly by library versions.

## Requirements
If you need to recreate `requirements.txt`, it should minimally include:
```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
ipykernel
```
This repository includes a `requirements.txt` with pinned versions for convenience.

## License
This project is provided as-is for educational purposes.