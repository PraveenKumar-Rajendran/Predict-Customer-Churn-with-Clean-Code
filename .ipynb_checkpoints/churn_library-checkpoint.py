"""
Module: Customer Churn Prediction

This module implements various functions
to predict customer churn using machine learning techniques.


The main functions in this module are:
- import_data: Imports the data from a CSV file.
- perform_eda: Performs exploratory data analysis and saves visualizations.
- encoder_helper: Encodes categorical variables for feature engineering.
- perform_feature_engineering: Performs feature engineering
and splits the data into train and test sets,
- classification_report_image: Generates a classification report image
for model evaluation.
- feature_importance_plot: Plots the feature importance for the trained model.
- train_models: Trains and evaluates
machine learning models (Logistic Regression and Random Forest).

This module uses scikit-learn, shap, pandas, numpy, matplotlib, and seaborn libraries.

Author: Praveen Kumar
Creation Date: 1 April 2024
"""
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# import libraries
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)
    return dataframe


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    if not os.path.exists('images'):
        os.makedirs('images')

    # Histogram of Customer Age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Distribution of Customer Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('images/eda/customer_age_distribution.png')
    plt.close()

    # Bar plot of Marital Status
    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts(normalize=True).plot(kind='bar')
    plt.title('Marital Status Distribution')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion')
    plt.savefig('images/eda/marital_status_distribution.png')
    plt.close()

    # Bar plot of Normalized Marital Status
    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts(normalize=True).plot(kind='bar')
    plt.title('Normalized Marital Status Distribution')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion')
    plt.savefig('images/eda/normalized_marital_status_distribution.png')
    plt.close()

    # Density plot of Total_Trans_Ct
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Density Plot of Total Transaction Count')
    plt.xlabel('Total Transaction Count')
    plt.ylabel('Density')
    plt.savefig('images/eda/total_transaction_count_density.png')
    plt.close()

    # Heatmap of Correlation Matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Correlation Matrix')
    plt.savefig('images/eda/correlation_matrix.png')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    for column in category_lst:
        lst = []
        groups = df.groupby(column).mean()[response]
        for val in df[column]:
            lst.append(groups.loc[val])
        df[column + '_Churn'] = lst
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Feature selection and engineering, train/test split
    X = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    y = df[response]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    '''
    produces classification report for training and testing results
    and stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    train_report_lr = classification_report(
        y_train, y_train_preds_lr, output_dict=True)
    test_report_lr = classification_report(
        y_test, y_test_preds_lr, output_dict=True)

    train_report_rf = classification_report(
        y_train, y_train_preds_rf, output_dict=True)
    test_report_rf = classification_report(
        y_test, y_test_preds_rf, output_dict=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Logistic Regression
    sns.heatmap(pd.DataFrame(train_report_lr).iloc[:-1, :].T,
                annot=True, cmap='coolwarm', ax=axes[0, 0])
    axes[0, 0].set_title('LR Training Classification Report')

    sns.heatmap(pd.DataFrame(test_report_lr).iloc[:-1, :].T,
                annot=True, cmap='coolwarm', ax=axes[0, 1])
    axes[0, 1].set_title('LR Testing Classification Report')

    # Random Forest
    sns.heatmap(pd.DataFrame(train_report_rf).iloc[:-1, :].T,
                annot=True, cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title('RF Training Classification Report')

    sns.heatmap(pd.DataFrame(test_report_rf).iloc[:-1, :].T,
                annot=True, cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('RF Testing Classification Report')

    plt.tight_layout()
    plt.savefig('images/results/classification_report.png')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        'images/results/feature_importance.png')

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # ROC plot for logistic regression and random forest
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.tight_layout()
    plt.savefig('images/results/ROC_Curve.png')
    plt.close()


if __name__ == "__main__":
    # Import data
    df = import_data("./data/bank_data.csv")

    # Perform EDA
    perform_eda(df)

    # Encode categorical variables
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    df = encoder_helper(df, category_lst, 'Churn')

    # Perform feature engineering and train/test split
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    # Train models and store results
    train_models(X_train, X_test, y_train, y_test)

    # Generate classification report images
    y_train_preds_rf = joblib.load('./models/rfc_model.pkl').predict(X_train)
    y_test_preds_rf = joblib.load('./models/rfc_model.pkl').predict(X_test)
    y_train_preds_lr = joblib.load(
        './models/logistic_model.pkl').predict(X_train)
    y_test_preds_lr = joblib.load(
        './models/logistic_model.pkl').predict(X_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    print("DONE!")
