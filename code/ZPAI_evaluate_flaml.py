from time import time
from pathlib import Path

import pandas as pd
import numpy as np

from ZPAI_common_functions import calculate_scores

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from flaml import AutoML
from flaml import logger
import logging
logger.setLevel(logging.WARNING)

# Set the number of CPUs to be used.
# value 1 -> use 1 CPU
# value 2 -> use 2 CPUs ...
# value -1 -> use all CPUs
# value -2 -> use all but one CPU ...
n_cpus = -2


def evaluate_flaml(X_train: pd.DataFrame, 
                         y_train: pd.DataFrame, 
                         X_test: pd.DataFrame, 
                         y_test: pd.DataFrame, 
                         summery_file: str, 
                         input_filename: str, 
                         file_path_pics: str, 
                         file_path_data: str, 
                         result_df: pd.DataFrame, 
                         feature_set: str,
                         config: dict):
                         
    X_train =  X_train
    y_train = y_train
    # column_count = column_count
    X_test = pd.DataFrame(X_test)
    y_test = y_test
    # MACHINE_MODEL = machine_model
    # machine_type = machine_type
    FILE_PATH_PICS = file_path_pics
    FILE_PATH_DATA = file_path_data
    SUMMERY_FILE = summery_file
    input_filename = input_filename
    result_df = result_df
    feature_set = feature_set

    logger.setLevel(logging.WARNING)
    
    M_DATE = config["general"]["start_date"]
    TIME_FOR_TASK = config["autosklearn"]['params']['time_for_task']

    print('\n --Flaml approach-- for feature set: {}'.format(feature_set))
    # open the summery file
    f = open(SUMMERY_FILE, "a")

    # create the data list for storing the results of the computation
    data_list = []

    ###########################
    # Flaml
    ###########################

    f.write('\n\n')
    f.write('Feature set for Flaml: \t' + feature_set)
    f.write('\n')

   
   # flaml built-in metric: https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#customize-automlfit

    automl = AutoML()

    # Training starting time
    train_start_time = time()

    automl.fit(X_train = X_train,
               y_train = y_train,
               task = "regression",
               time_budget = TIME_FOR_TASK,
               metric = "mape",
            #    max_iter = 20,
               verbose = 1)

    # Training stop time
    train_stop_time = time()

    # training duration 
    training_duration = train_stop_time - train_start_time



    # # save the leaderboard
    # filename = "{}-{}-{}-{}.{}".format(M_DATE, input_filename, 'autosklearn-leaderboard', feature_set,'csv')
    # LEADERBOARD_FILE = Path(FILE_PATH_DATA, filename)
    # model_leadership.to_csv(str(LEADERBOARD_FILE))


    train_predictions = automl.predict(X_train)
    print("Train MSE score:", mean_squared_error(y_train, train_predictions))

    # Test start time
    test_start_time = time()

    final_predictions = automl.predict(X_test) 

    # Test stop time
    test_stop_time = time()

    # test time per sample -> devide total test time by the amount of test items
    num_elements = X_test.shape[0]
    test_time_per_sample = ((test_stop_time - test_start_time) / num_elements)

    df_temp = pd.DataFrame({'Actual': y_test, 'Predicted': final_predictions})
    df_temp.Predicted = df_temp.Predicted.astype(int)

    # Display the results within a barchart diagram
    df_temp = df_temp.head(30)
    df_temp.plot(kind='bar',figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    #plt.savefig(FILE_PATH_PICS+'/'+'Test5.png')
    # filename = "{}-{}-{}-{}.{}".format(d,input_filename,feature_set,'autosklearn-test-result','png')
    filename = "{}-{}-{}-{}.{}".format(M_DATE,input_filename,'flaml-test-result',feature_set,'png')
    title = str("Flaml results for {} on feat. set {}".format(input_filename, feature_set))
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)

    # close open windows 
    plt.close()

    ###################
    # Model evaluation
    ###################
    # Calculate metric scores
    mean_abs_error, mean_abs_percentage_error, r2_score_value, RMSE = calculate_scores(y_test, final_predictions)
    
    # Normalized MRSE
    Y_MAX = y_test.max()
    Y_MIN = y_test.min()
    N_RMSE = RMSE / (Y_MAX - Y_MIN)

    # Inter quantile range RMSE
    Q1 = y_test.quantile(q=0.25)
    Q3 = y_test.quantile(q=0.75)
    IQR_RMSE = RMSE / (Q3 - Q1)
    # coefficient of variation of the RMSD
    Y_MEAN = y_test.mean()
    CV_RMSE = RMSE / Y_MEAN

    f.write('Mean Absolute Error: \t' + str(np.round(mean_abs_error, 2)))
    f.write('\n')
    f.write('Mean Absolute Percentage Error: \t' + str(np.round(mean_abs_percentage_error, 4)))
    f.write('\n')
    f.write('Root Mean Squared Error: \t' + str(np.round(RMSE, 2)))
    f.write('\n')
    f.write('N-RMSE: \t' + str(np.round(N_RMSE, 4)))
    f.write('\n')
    f.write('IQR-RMSE: \t' + str(np.round(IQR_RMSE, 4)))
    f.write('\n')
    f.write('CV-RMSE: \t' + str(np.round(CV_RMSE, 4)))
    f.write('\n')
    f.write('R2 test score: \t' + str(np.round(r2_score_value, 3)))
    f.write('\n\n')

    # add estimator score to data_list
    data_list.append(np.round(mean_abs_error, 2))
    data_list.append(np.round(mean_abs_percentage_error, 4))
    data_list.append(np.round(RMSE, 2))
    data_list.append(np.round(N_RMSE, 4))
    data_list.append(np.round(IQR_RMSE, 4))
    data_list.append(np.round(CV_RMSE, 4))
    data_list.append(round(r2_score_value, 4))

    #store training duration
    data_list.append(np.round(training_duration, 3))

    # store test duration for a sigle test point
    data_list.append(np.round(test_time_per_sample, 10))

    # put the data into the result file
    result_df[feature_set] = data_list

    # close the summery file
    f.close()
