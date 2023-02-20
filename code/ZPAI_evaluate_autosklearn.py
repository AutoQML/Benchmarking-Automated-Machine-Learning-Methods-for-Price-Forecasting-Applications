from time import time
from pathlib import Path

import pandas as pd
import numpy as np

from ZPAI_common_functions import calculate_scores

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from autosklearn.regression import AutoSklearnRegressor
import autosklearn

# Set the number of CPUs to be used.
# value 1 -> use 1 CPU
# value 2 -> use 2 CPUs ...
# value -1 -> use all CPUs
# value -2 -> use all but one CPU ...
n_cpus = -2


def evaluate_autosklearn(X_train: pd.DataFrame, 
                         y_train: pd.DataFrame, 
                         X_test: pd.DataFrame, 
                         y_test: pd.DataFrame, 
                         summery_file: str, 
                         input_filename: str, 
                         file_path_pics: str, 
                         file_path_data: str, 
                         result_df: pd.DataFrame, 
                         feature_set: str,
                         config: dict,
                         loop_count: int,
                         measurements: int):
                         
    X_train =  X_train
    y_train = y_train
    # column_count = column_count
    X_test = X_test
    y_test = y_test
    # MACHINE_MODEL = machine_model
    # machine_type = machine_type
    FILE_PATH_PICS = file_path_pics
    FILE_PATH_DATA = file_path_data
    SUMMERY_FILE = summery_file
    input_filename = input_filename
    result_df = result_df
    feature_set = feature_set
    LOOP_COUNT = loop_count
    ITERATION = measurements
    
    M_DATE = config["general"]["start_date"]
    TIME_FOR_TASK = config["autosklearn"]['params']['time_for_task']
    RUN_TIME_LIMIT = config["autosklearn"]['params']['run_time_limit']

    print('\n --Auto-Sklearn approach-- for feature set: {}'.format(feature_set))
    # open the summery file
    f = open(SUMMERY_FILE, "a")

    # create the data list for storing the results of the computation
    data_list = []

    ###########################
    # Autosklearn
    ###########################

    f.write('\n\n')
    f.write('Feature set for Auto-sklearn: \t' + feature_set)
    f.write('\n')

    # Available REGRESSION autosklearn.metrics.*:
    #     *mean_absolute_error
    #     *mean_squared_error
    #     *root_mean_squared_error
    #     *mean_squared_log_error
    #     *median_absolute_error
    #     *r2

    error_rate = autosklearn.metrics.make_scorer(
        name="mape_error",
        score_func= mean_absolute_percentage_error,
        optimum=0,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
    )

    tmp_folder_x = filename = "{}_{}_{}".format('/tmp/autosklearn/autosklearn_regression_tmp', ITERATION, LOOP_COUNT)

    automl = AutoSklearnRegressor(time_left_for_this_task = TIME_FOR_TASK, 
                                  per_run_time_limit=RUN_TIME_LIMIT,
                                  # metric = error_rate,
                                #   smac_scenario_args={'runcount_limit': 20},
                                #   initial_configurations_via_metalearning = 19,
                                  metric = autosklearn.metrics.mean_absolute_error,
                                  tmp_folder = tmp_folder_x, 
                                  memory_limit = None,
                                  delete_tmp_folder_after_terminate = True)

    # convert 'object' type to 'categorical' type due to a ValueError from auto-sklearn 
    list_str_obj_cols = X_train.columns[X_train.dtypes == "object"].tolist()
    for str_obj_col in list_str_obj_cols:
        X_train[str_obj_col] = X_train[str_obj_col].astype("category")

    # Training starting time
    train_start_time = time()

    automl.fit(X_train, y_train)

    # Training stop time
    train_stop_time = time()

    # training duration 
    training_duration = train_stop_time - train_start_time

    # print(automl.leaderboard())
    model_leadership = automl.leaderboard(detailed = True,
                                          ensemble_only=False,
                                          sort_order="ascending")

    # save the leaderboard
    filename = "{}-{}-{}-{}.{}".format(M_DATE, input_filename, 'autosklearn-leaderboard', feature_set,'csv')
    LEADERBOARD_FILE = Path(FILE_PATH_DATA, filename)
    model_leadership.to_csv(str(LEADERBOARD_FILE))


    # train_predictions = automl.predict(X_train)
    # print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))

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
    filename = "{}-{}-{}-{}.{}".format(M_DATE,input_filename,'autosklearn-test-result',feature_set,'png')
    title = str("Autosklearn results for {} on feat. set {}".format(input_filename, feature_set))
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)

    # close open windows 
    plt.close()

    ###################
    # Model evaluation
    ###################
    # Calculate metric scores
    mean_abs_error, mean_abs_percentage_error, r2_score_value, RMSE = calculate_scores(y_test, final_predictions)
    # # Calculate MAE and MEPE according to https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error
    # mean_abs_error = mean_absolute_error(y_test, final_predictions)
    # mean_abs_percentage_error = mean_absolute_percentage_error(y_test, final_predictions)
    # r2_score_value = r2_score(y_test, final_predictions) 

    # # calculate RMSE value and its derivative according to https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization
    # # RMSE
    # RMSE = np.sqrt(mean_squared_error(y_test, final_predictions))
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

    final_test_score = automl.score(X_test, y_test) # Return the coefficient of determination R2 of the prediction. (https://automl.github.io/auto-sklearn/master/api.html#regression)
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
