"""
This is the Python Test for the churn_library.py module.

This module will be used to test
    1. import_data
    2. peform_eda
    3. encode_helper
    4. perform_feature_engineering
    5. train_test_model

Author: Raju Tadisetti
Date: March 17 2023
"""
import os
import logging
import pytest
import churn_library as cls

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
    pytest.df = df

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    """
    test exploaratory data anlysis
        """
    try:
        perform_eda(pytest.df)
        logging.info("Testing exploaratory Data: SUCCESS")
    except KeyError as err:
        logging.error('Testing eda: The feature wasn not found')
        raise err
    try:
        assert os.path.isfile('./images/count.png')
        assert os.path.isfile('./images/corr.png')
        assert os.path.isfile('./images/hist.png')
    except AssertionError as err:
        logging.error("Testing eda: images weren't saved")
        raise err


def test_encoder_helper(encoder_helper, cat_column):
    '''
    test encoder helper
'''
    try:
        new_df = encoder_helper(pytest.df, cat_column)
        logging.info("Testing encoder_helper: SUCCESS")
    except KeyError as err:
        logging.error("Testing encoder helper: Failed")
        raise err
    pytest.new_df = new_df
    try:
        assert new_df.shape[0] > 0
        assert new_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test feature engineering
'''
    y = pytest.new_df['Churn']
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            pytest.new_df, y)
        logging.info("Testing feature Engineering: SUCCESS")
    except KeyError as err:
        logging.error("Testing feature Engineering: FAILED")
        raise err
    pytest.x_train = x_train
    pytest.x_test = x_test
    pytest.y_train = y_train
    pytest.y_test = y_test
    try:
        assert x_train.shape == y_train.shape
        assert x_test.shape == y_test.shape
    except AssertionError as err:
        logging.warning(" Testing feature engineering: invalid split data")


def test_train_models(train_models):
    '''
    test train models
    '''
    try:
        train_models(
            pytest.x_train,
            pytest.x_test,
            pytest.y_train,
            pytest.y_test)
        logging.info("Testing train models: SUCCESS")
    except NameError as err:
        logging.error("Model hasn't defined")
        raise err
    try:
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./images/logi.png')
        assert os.path.isfile('./images/feautre_importance.jpeg')
        assert os.path.isfile('./images/roc_curve.png')
        assert os.path.isfile('./models/logistic_model.pkl')
    except AssertionError as err:
        logging.error(
            " Testing train model: the model hasn't trained properly")


if __name__ == "__main__":
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    test_import(cls. import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper, cat_columns)
    test_perform_feature_engineering(cls.perform_feature_engineerig)
    test_train_models(cls.train_models)
