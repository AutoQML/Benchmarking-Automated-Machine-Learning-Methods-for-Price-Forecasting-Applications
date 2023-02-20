# Import necessary packages.
import platform

from pathlib import Path
import pandas as pd
import numpy as np

#import datetime
from datetime import date

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# for combinatorial purposses
import itertools

# Use scikit-learns for cross validation.
from sklearn.model_selection import cross_val_score

# Use PCA to detect the most important features
from sklearn.decomposition import PCA

from ZPAI_evaluate_classic_ml_models import eval_classic_ml_models
from ZPAI_evaluate_neural_nets import evaluate_neural_nets
from ZPAI_evaluate_autosklearn import evaluate_autosklearn

from time import time

import yaml

REPO_PATH = Path(__file__).parents[1]

with open(REPO_PATH / 'conf/auto_sklearn_config.yml', 'r') as file:
    auto_config = yaml.safe_load(file)

# Set the number of CPUs to be used.
# value 1 -> use 1 CPU
# value 2 -> use 2 CPUs ...
# value -1 -> use all CPUs
# value -2 -> use all but one CPU ...
n_cpus = -2

def prepare_merged_data_for_ml(machine_model, 
                               machine_type, 
                               file_path_pics, 
                               file_path_data, 
                               input_filename, 
                               summery_file, 
                               pca_num, 
                               random_state, 
                               m_date):
                               
    MACHINE_MODEL = machine_model
    machine_type = machine_type
    FILE_PATH_PICS = file_path_pics
    FILE_PATH_DATA = file_path_data
    SUMMERY_FILE = summery_file
    input_filename = input_filename
    PCA_NUM = pca_num
    RANDOM_STATE = random_state
    m_date = m_date
    # result_df = result_df
    TIME_FOR_TASK = auto_config['params']['time_for_task']
    RUN_TIME_LIMIT = auto_config['params']['run_time_limit']

    # get cuttent OS
    my_os = platform.system()

    # # get cuttent date
    # today = date.today()
    # # YY-mm-dd
    # d = today.strftime("%Y-%m-%d")

    # create result data frame to store the measurements for classic approach
    classic_index = ['CV - LinReg - Mean MAE', 'CV - LinReg - Mean RMSE', 'CV - LinReg - Mean R2Score',
                'CV - Tree - Mean MAE', 'CV - Tree - Mean RMSE', 'CV - Tree - Mean R2Score',
                'CV - RandForest - Mean MAE', 'CV - RandForest - Mean RMSE', 'CV - RandForest - Mean R2Score',
                'CV - SVR - Mean MAE', 'CV - SVR - Mean RMSE', 'CV - SVR - Mean R2Score',
                'CV - KNN - Mean MAE', 'CV - KNN - Mean RMSE', 'CV - KNN - Mean R2Score',
                'CV - AdaBoost - Mean MAE', 'CV - AdaBoost - Mean RMSE', 'CV - AdaBoost - Mean R2Score',
                'final-model', 'Test-MAE', 'Test-RMSE', 'Test-R2', 'Duration']

    # create result data frame
    classic_result_df = pd.DataFrame(index=classic_index)

    # create result data frame to store the measurements for NN
    nn_index = ['Test-MAE', 'Test-RMSE', 'Test-R2', 'Duration', 'Activation', 'Hidden-layer-size', 'Learning rate', 'Solver']
    # create result data frame
    nn_result_df = pd.DataFrame(index=nn_index)

    # create result data frame to store the measurements for autosklearn
    auto_index = ['Test-MAE', 'Test-RMSE', 'Test-R2', 'Duration']
    # create result data frame
    auto_result_df = pd.DataFrame(index=auto_index)

    ###########################
    # Data preparation
    ###########################

    categorical_cols = ['brand', 'model', 'location', 'extension']
    machine_typepd_prep_cat = pd.get_dummies(machine_type, columns=categorical_cols) # Preprocess categorical attributes

    # confert all numerial values to type int
    machine_type_preprep = machine_typepd_prep_cat.astype(int)

    # calculate the number of columns without the price column (therfore  -1)
    column_count = len(machine_type_preprep.columns) -1

    # Split the data
    X_train_preprep, X_test_preprep = train_test_split(machine_type_preprep, test_size=0.1, random_state=RANDOM_STATE)
    # X_train_preprep.info(), X_test_preprep.info()

    # Drop the price for X_train
    machine_type_X_train = X_train_preprep.drop('price', axis = 1)

    # Create the label y_train
    machine_type_y_train = X_train_preprep['price'].copy()

    #####################
    # Piplenes & scaling
    #####################

    # Build pipelines for preprocessing the attributes. Use sklearn Pipeline for pipelines and  sklearn StandardScaler for scaling the values of the attributes. 
    full_pipeline = ColumnTransformer([
            ("num", StandardScaler(), ['working_hours', 'const_year'])
        ], remainder='passthrough')

    # Prepare the training date X_train
    machine_type_prepared = full_pipeline.fit_transform(machine_type_X_train)
    machine_type_prepared = pd.DataFrame(machine_type_prepared)

    # store prepared date as csv
    filename = "{}-{}-{}.{}".format(m_date,input_filename,'prepdata','csv')
    PREPDATE_CSV = Path(FILE_PATH_DATA, filename)
    machine_type_prepared.to_csv(PREPDATE_CSV)

    # Get training data
    machine_type_X_train = machine_type_prepared.copy()

    # Do the final test with the test set with the best estimator!
    machine_type_X_test = X_test_preprep.drop('price', axis = 1)
    machine_type_y_test = X_test_preprep['price'].copy()

    X_test_prepared = full_pipeline.transform(machine_type_X_test)

    feature_set = 'all-features'

    # evaluate classical ml models like lin. regression, trees, forests, SVM
    eval_classic_ml_models(machine_type_X_train, machine_type_y_train, X_test_prepared, machine_type_y_test, SUMMERY_FILE, column_count, input_filename, FILE_PATH_PICS, classic_result_df, feature_set, RANDOM_STATE, m_date)

    # evaluate NN
    evaluate_neural_nets(machine_type_X_train, machine_type_y_train, X_test_prepared, machine_type_y_test, SUMMERY_FILE, input_filename, FILE_PATH_PICS, nn_result_df, feature_set, RANDOM_STATE, m_date)

    # evaluate AutoML - autosklearn
    # check for OS - autosklearn is not running on MAC (Darwin) at the moment
    if my_os == 'Linux':
        evaluate_autosklearn(machine_type_X_train, machine_type_y_train, X_test_prepared, machine_type_y_test, SUMMERY_FILE, input_filename, FILE_PATH_PICS, FILE_PATH_DATA, auto_result_df, feature_set, RANDOM_STATE, m_date)
    if my_os == 'Darwin':
        print("System OS: ",my_os)


    # store results within the results.csv
    # FILE_PATH_DATA = Path('./measurements', MACHINE_MODEL, 'data')
    # filename = "{}-{}-{}.{}".format(MACHINE_MODEL, d, 'classic-results','csv')
    filename = "{}-{}-{}.{}".format(m_date, input_filename, 'classic-results','csv')
    RESULT_CSV = Path(FILE_PATH_DATA, filename)
    classic_result_df.to_csv(RESULT_CSV)

    # store NN results within the results.csv
    # FILE_PATH_DATA = Path('./measurements', MACHINE_MODEL, 'data')
    # filename = "{}-{}-{}.{}".format(MACHINE_MODEL, d, 'nn-results','csv')
    filename = "{}-{}-{}.{}".format(m_date, input_filename, 'nn-results','csv')
    RESULT_CSV = Path(FILE_PATH_DATA, filename)
    nn_result_df.to_csv(RESULT_CSV)

     # store NN results within the results.csv
    # FILE_PATH_DATA = Path('./measurements', MACHINE_MODEL, 'data')
    # filename = "{}-{}-{}.{}".format(MACHINE_MODEL, d, 'auto-results','csv')
    filename = "{}-{}-{}.{}".format(m_date, input_filename, 'autosklearn-results','csv')
    RESULT_CSV = Path(FILE_PATH_DATA, filename)
    auto_result_df.to_csv(RESULT_CSV)
