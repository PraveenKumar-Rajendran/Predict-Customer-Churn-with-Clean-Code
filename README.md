# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

This project aims to predict customer churn for a bank using machine learning techniques. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Files and Data Description

- `churn_library.py`: This Python script contains various functions to preprocess the data, perform EDA, train machine learning models (Logistic Regression and Random Forest), and generate evaluation reports and visualizations.
- `churn_script_logging_and_tests.py`: This script contains test functions to ensure the correct functionality of the functions in `churn_library.py`. It also logs the test results to a file.
- `data/bank_data.csv`: This CSV file contains the banking customer data used for training and evaluating the machine learning models.
- `images/eda/`: This directory stores the EDA visualizations generated during the analysis.
- `images/results/`: This directory stores the evaluation reports, feature importance plots, and other visualizations generated after model training.
- `logs/churn_library.log`: This log file contains the output of the test functions from `churn_script_logging_and_tests.py`.
- `models/`: This directory stores the trained machine learning models (Logistic Regression and Random Forest) in pickle format.

## Running the Scripts

1. Make sure you have the required Python libraries installed (`scikit-learn`, `shap`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `joblib`).
2. Create the `images/`, `images/eda/`, `images/results/`, `logs/`, and `models/` directories in the project root directory.
3. Place the `bank_data.csv` file in the `data/` directory.
4. Run the `churn_library.py` script to preprocess the data, perform EDA, train the models, and generate evaluation reports and visualizations.
5. Run the `churn_script_logging_and_tests.py` script to test the functions in `churn_library.py` and log the results to `logs/churn_library.log`.

When you run `churn_library.py`, it will:

- Import the `bank_data.csv` file and preprocess the data.
- Perform EDA and save the visualizations in the `images/eda/` directory.
- Encode categorical variables and perform feature engineering.
- Split the data into training and testing sets.
- Train Logistic Regression and Random Forest models.
- Generate evaluation reports (classification reports and ROC curves) and save them in the `images/results/` directory.
- Save the trained models in the `models/` directory.

When you run `churn_script_logging_and_tests.py`, it will:

- Test the functions in `churn_library.py` (`import_data`, `perform_eda`, `encoder_helper`, `perform_feature_engineering`, and `train_models`).
- Log the test results (success or failure) in the `logs/churn_library.log` file.

Note: Make sure to create the required directories (`images/`, `images/eda/`, `images/results/`, `logs/`, and `models/`) before running the scripts.



