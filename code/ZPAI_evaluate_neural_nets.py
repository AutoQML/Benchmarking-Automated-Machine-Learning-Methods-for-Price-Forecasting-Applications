# Import necessary packages.
import pandas as pd
import numpy as np
from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ZPAI_common_functions import calculate_scores

# Set the number of CPUs to be used.
# value 1 -> use 1 CPU
# value 2 -> use 2 CPUs ...
# value -1 -> use all CPUs
# value -2 -> use all but one CPU ...
n_cpus = -2


def evaluate_neural_nets(X_train: pd.DataFrame, 
                         y_train: pd.DataFrame, 
                         X_test: pd.DataFrame, 
                         y_test: pd.DataFrame, 
                         summery_file: str, 
                         input_filename: str, 
                         file_path_pics: str, 
                         result_df: pd.DataFrame, 
                         feature_set: str,
                         config: dict):
    """
    Trains and evalutates neural networks on given dataset.

    Hyperparameters (number of hidden layers and learning rate) are optimized using GridSearch and 3-fold crossvalidation. 
    Each training procedure trains the network for 100.000 epochs and batch-size of 16 with early-stopping after 1000 epochs without improvement. 
    After the training, the performance of the best estimator is evaluated on the test set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Input variables of the training set
    y_train : pd.DataFrame
        Target variable of the training set
    X_test : pd.DataFrame
        Input variables of the test set
    y_test : pd.DataFrame
        Target variable of the training set
    summery_file: str
        Path to summary file
    input_filename: str
        Name of the input file
    file_path_pics: str
        Path to directory in which the plots should be saved
    result_df: pd.DataFrame
        DataFrame in which the results should be saved
    feature_set: str
        Name of the current feature set (e.g. location + extension)
    config: dict
        Global configuration dictionary
    
    Returns
    -------


    """
                         
    RANDOM_STATE = config["general"]["random_state"]
    M_DATE = config["general"]["start_date"]


    print('\n --NN approach-- for feature set: {}'.format(feature_set))
    # open the summery file
    f = open(summery_file, "a")

    # create the data list for storing the results of the computation
    data_list = []

    ###########################
    # Train the neural network
    ###########################

    f.write('\n\n')
    f.write('Feature set for NN: \t' + feature_set)
    f.write('\n')

    f.write('\n\n')
    f.write('Train a NN:')
    f.write('\n')

    parameter_space = {
        'hidden_layer_sizes': [tuple(i*[10]) for i in range(1,10,2)],
        'learning_rate_init': [1e-2, 1e-3, 1e-4, 1e-5],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha' : [0.001, 0.0001, 0.00001]
    }

    nnregr = MLPRegressor(batch_size = 16, max_iter=100000, early_stopping=True, n_iter_no_change=1000)
    rand_search = RandomizedSearchCV(nnregr, parameter_space, scoring='neg_mean_absolute_percentage_error', cv = 3, refit = True, n_iter = 20, n_jobs = -2)

     # Training starting time
    train_start_time = time()
    
    rand_search.fit(X_train, y_train)

    # Training stop time
    train_stop_time = time()

    # training duration 
    training_duration = train_stop_time - train_start_time

    nnregr = rand_search.best_estimator_

    # print("Best parameters found: ", rand_search.best_params_)
    # print("Best model trained for: ", nnregr.n_iter_, " iterations")


    # Test start time
    test_start_time = time()

    final_predictions = nnregr.predict(X_test)
    # final_predictions 

    # Test stop time
    test_stop_time = time()

    # test time per sample -> devide total test time by the amount of test items
    num_elements = X_test.shape[0]
    test_time_per_sample = ((test_stop_time - test_start_time) / num_elements)

    df_temp = pd.DataFrame({'Actual': y_test, 'Predicted': final_predictions})
    df_temp.Predicted = df_temp.Predicted.astype(int)
    df_temp.head(30)

    # Display the results within a barchart diagram
    df_temp = df_temp.head(30)
    df_temp.plot(kind='bar',figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    filename = "{}-{}-{}-{}.{}".format(M_DATE,input_filename,'nn-test-result',feature_set,'png')
    title = str("NN results for {} on feat. set {}".format(input_filename, feature_set))
    plt.title(title)
    plt.savefig(file_path_pics+'/'+filename)

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

    # final_test_score = nnregr.score(X_test, y_test)
    f.write('Mean Absolute Error: \t' + str(np.round(mean_abs_error, 2)))
    f.write('\n')
    f.write('Mean Absolute Percentage Error: \t' + str(np.round(mean_abs_percentage_error, 4)))
    f.write('\n')
    f.write('Root Mean Squared Error: \t' + str(np.round(RMSE, 2)))
    f.write('\n')
    f.write('N_RMS: \t' + str(np.round(N_RMSE, 4)))
    f.write('\n')
    f.write('IQR_RMSE: \t' + str(np.round(IQR_RMSE, 4)))
    f.write('\n')
    f.write('CV_RMS: \t' + str(np.round(CV_RMSE, 4)))
    f.write('\n')
    f.write('R2 test score: \t' + str(np.round(r2_score_value, 4)))
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

    # store activation type
    data_list.append(rand_search.best_params_.get('activation'))

    # store hidden layer size
    data_list.append(rand_search.best_params_.get('hidden_layer_sizes'))

    # store learning rate
    data_list.append(rand_search.best_params_.get('learning_rate_init'))

    # store solver
    data_list.append(rand_search.best_params_.get('solver'))

    # put the data into the result file
    result_df[feature_set] = data_list

    # close the summery file
    f.close()
