There are two parts. Each part has a notebook with the full code and results, and a slide deck.

## Part 1: Kaggle survey analysis
- Notebook: `kaggle_survey_eda.ipynb`
- Slides: `kaggle.pdf`
- Data file needed: `kaggle_survey_2017_2021.csv`, put it in the same folder as the notebook

## Part 2: Fraud detection
- Notebook: `fraud_detection.ipynb`. It builds the features, trains the models, picks the alert
  threshold for 1,000 investigations a month, explains the alerts, and replays six months of
  scoring and monitoring
- `monitor.py` is the monthly monitoring script. The notebook runs it, so keep it in the same folder
- Slides: `fraud_detection.pdf`
- Data files needed: `fraudTrain.csv` and `fraudTest.csv`, put them in the same folder as the notebook

## How to run
You need Python 3.12 and these packages: pandas, numpy, scikit-learn, xgboost, torch, shap, scipy,
mlflow, matplotlib, seaborn, joblib.

Open the notebook and run all the cells from the top. The fraud notebook takes a few minutes with a
GPU and about ten minutes without. It creates a few folders and a file called `mlflow.db`.

If you just want a quick check, set `FAST=1` before starting Jupyter and it runs on 20% of the data.

To see the training runs and the monthly monitoring results, run this in the folder:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`
