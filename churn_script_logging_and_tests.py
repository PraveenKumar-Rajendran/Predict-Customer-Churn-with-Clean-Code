"""
Module for testing the customer churn prediction library.

Tests:
- import_data
- perform_eda
- encoder_helper
- perform_feature_engineering
- train_models

Logs results to ./logs/churn_library.log

Prerequisites:
- churn_library module
- ./data/bank_data.csv file

Author: Praveen Kumar
Creation Date: 1 April 2024
"""
import os
import logging
import churn_library as cls

# Configure logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: An error occurred")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        df = encoder_helper(df, category_lst, 'Churn')
        assert 'Gender_Churn' in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: An error occurred")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        df = cls.encoder_helper(df, category_lst, 'Churn')
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, 'Churn')
        assert x_train.shape[0] > 0 and x_test.shape[0] > 0
        assert y_train.shape[0] > 0 and y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: An error occurred")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        df = cls.encoder_helper(df, category_lst, 'Churn')
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            df, 'Churn')
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: An error occurred")
        raise err


if __name__ == "__main__":
    import_data = cls.import_data
    perform_eda = cls.perform_eda
    encoder_helper = cls.encoder_helper
    perform_feature_engineering = cls.perform_feature_engineering
    train_models = cls.train_models

    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)