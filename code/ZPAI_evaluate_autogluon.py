from time import time
from pathlib import Path

import pandas as pd
import numpy as np

from ZPAI_common_functions import calculate_scores

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from autogluon.tabular import TabularPredictor

# Set the number of CPUs to be used.
# value 1 -> use 1 CPU
# value 2 -> use 2 CPUs ...
# value -1 -> use all CPUs
# value -2 -> use all but one CPU ...
n_cpus = -2


def evaluate_autogluon(X_train: pd.DataFrame, 
                         X_test: pd.DataFrame,  
                         summery_file: str, 
                         input_filename: str, 
                         file_path_pics: str, 
                         file_path_data: str, 
                         result_df: pd.DataFrame, 
                         feature_set: str,
                         config: dict):
                         
    X_train =  X_train
    # column_count = column_count
    X_test = X_test
    # MACHINE_MODEL = machine_model
    # machine_type = machine_type
    FILE_PATH_PICS = file_path_pics
    FILE_PATH_DATA = file_path_data
    SUMMERY_FILE = summery_file
    input_filename = input_filename
    result_df = result_df
    feature_set = feature_set
    
    M_DATE = config["general"]["start_date"]
    TIME_FOR_TASK = config["autosklearn"]['params']['time_for_task']

    print('\n --AutoGluon approach-- for feature set: {}'.format(feature_set))
    # open the summery file
    f = open(SUMMERY_FILE, "a")

    # create the data list for storing the results of the computation
    data_list = []

    ###########################
    # Autogluon
    ###########################

    f.write('\n\n')
    f.write('Feature set for AutoGluon: \t' + feature_set)
    f.write('\n')

    label = 'price'
    train_data = X_train

    save_path = 'agModels-predictClass'  # specifies folder to store trained models

    automl = TabularPredictor(label=label, path=save_path, eval_metric='mean_absolute_percentage_error')

    # Training starting time
    train_start_time = time()

    automl.fit(train_data, time_limit=TIME_FOR_TASK)

    # Training stop time
    train_stop_time = time()

    # training duration 
    training_duration = train_stop_time - train_start_time

    y_test = X_test[label]  # values to predict
    test_data_nolab = X_test.drop(columns=[label])  # delete label column to prove we're not cheating

    # Test start time
    test_start_time = time()

    y_pred = automl.predict(test_data_nolab)

    # Test stop time
    test_stop_time = time()

    # test time per sample -> devide total test time by the amount of test items
    num_elements = X_test.shape[0] # compute numbe of test items
    average_test_time = ((test_stop_time - test_start_time) / num_elements)

    # print("Predictions:  \n", y_pred)
    perf = automl.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    print(perf)

    # print(automl.leaderboard())
    test_data = X_test
    model_leadership = automl.leaderboard(test_data,
                                          silent=True)

    # save the leaderboard
    filename = "{}-{}-{}-{}.{}".format(M_DATE, input_filename, 'autogluon-leaderboard', feature_set,'csv')
    LEADERBOARD_FILE = Path(FILE_PATH_DATA, filename)
    model_leadership.to_csv(str(LEADERBOARD_FILE))


    final_predictions = automl.predict(test_data_nolab)

    df_temp = pd.DataFrame({'Actual': y_test, 'Predicted': final_predictions})
    df_temp.Predicted = df_temp.Predicted.astype(int)

    # Display the results within a barchart diagram
    df_temp = df_temp.head(30)
    df_temp.plot(kind='bar',figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    #plt.savefig(FILE_PATH_PICS+'/'+'Test5.png')
    # filename = "{}-{}-{}-{}.{}".format(d,input_filename,feature_set,'autogluon-test-result','png')
    filename = "{}-{}-{}-{}.{}".format(M_DATE,input_filename,'autogluon-test-result',feature_set,'png')
    title = str("AutoGluon results for {} on feat. set {}".format(input_filename, feature_set))
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)

    # close open windows 
    plt.close()

    ###################
    # Model evaluation
    ###################
    # Calculate metric scores
    mean_abs_error, mean_abs_percentage_error, r2_score_value, RMSE = calculate_scores(y_test, final_predictions)
    # # Get MAE, MEPE, RMSE and R2 from TabularPredictor.evaluate_predictions according to https://auto.gluon.ai/stable/api/autogluon.predictor.html#autogluon.tabular.TabularPredictor.evaluate_predictions
    # mean_abs_error = -perf["mean_absolute_error"]
    # mean_abs_percentage_error = -perf["mean_absolute_percentage_error"]
    # RMSE = -perf["root_mean_squared_error"]
    # r2_score_value = perf["r2"]
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
    data_list.append(np.round(average_test_time, 10))

    # put the data into the result file
    result_df[feature_set] = data_list

    # close the summery file
    f.close()