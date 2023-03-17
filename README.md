# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity


## Project Description
This project aims to predict customers churn in a bank using machine learning techniques. The project uses a dataset with various features such as customer demographics, account information, and transaction history. The objective of the project is to build a machine learning model that can accurately predict whether a customer is likely to leave the bank or not.

## Files and data description
The project contains the following files:

    data/bank_data.csv: the dataset used in the project.
    images/: a folder containing the images generated in the project.
    models/: a folder containing the saved machine learning model.
    logs/: a folder containing log files
    churn_library.py: a Python module containing the functions used in the project.
    churn_script_logging_and_tests.py: a Python script for logging and testing the functions in the churn_library module.
    README.md: this file.
    requirements.txt: a file containing the required Python packages for running the project.
## Running Files
To run the project, you need to have Python 3 and the required Python packages installed on your system. You can install the required packages by running the following command:

    pip install -r requirements.txt

Once you have installed the required packages, you can run the churn_script_logging_and_tests.py script to test the functions in the churn_library.py module:

    python churn_script_logging_and_tests.py

If all the tests pass, you can run the churn_main.py script to train and evaluate the machine learning model:

    python churn_main.py

The script will load the data, preprocess it, split it into training and testing sets, train the model, evaluate its performance, and save the model to the models/ folder. Finally, it will generate some visualizations and save them to the images/ folder.

The script will also log its output to a file called churn_libary.log in logs folder.

Once the script is finished running, you can view the visualizations generated by the script by opening the files in the images/ folder.