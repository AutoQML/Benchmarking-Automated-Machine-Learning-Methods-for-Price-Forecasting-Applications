import pandas as pd
import numpy as np

from datetime import date
from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from scipy.stats import randint

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from ZPAI_common_functions import display_scores

# Set the number of CPUs to be used.
# value 1 -> use 1 CPU
# value 2 -> use 2 CPUs ...
# value -1 -> use all CPUs
# value -2 -> use all but one CPU ...
n_cpus = -2

def objective(trial, 
              X, 
              y, 
              pipeline, 
              parameters):

    trial_parameters = dict()

    param_names = list(parameters.keys())

    for i in range(len(param_names)):
        variable_type = parameters[param_names[i]][0]
        if variable_type == "int":
            trial_parameters[param_names[i]] = trial.suggest_int(param_names[i], parameters[param_names[i]][1], parameters[param_names[i]][2])
        elif variable_type == "float":
            trial_parameters[param_names[i]] = trial.suggest_float(param_names[i], parameters[param_names[i]][1], parameters[param_names[i]][2])
        elif variable_type == "categorical":
            trial_parameters[param_names[i]] = trial.suggest_categorical(param_names[i], parameters[param_names[i]][1:])

    pipeline = pipeline.set_params(**trial_parameters)

    score = cross_val_score(pipeline, X, y, cv = 3, scoring = 'neg_mean_squared_error').mean()
    return score


def cross_validation(pipeline: Pipeline, 
                     parameters: dict, 
                     n_iter: int, 
                     random_state: int, 
                     X_train: pd.DataFrame, 
                     y_train: pd.DataFrame):
    """
    Nested cross-validation on training dataset.

    Nested Cross-Validation (Nested-CV) nests cross-validation and hyperparameter tuning. 
    It is used to evaluate the performance of a machine learning algorithm and also estimates the generalization error of the underlying model and its hyperparameter search.

    Outer CV: A machine learning algorithm is selected based on its performance on the outer loop of nested cross-validation. This method is repeated k times, and the final CV score is computed by taking the mean of all k scores.
    Inner CV: Then the inner-CV is applied to the (k-1) folds or groups dataset from the outer CV. The set of parameters are optimized and is then used to configure the model. The best model returned from GridSearchCV or RandomSearchCV is then evaluated using the last fold or group. 

    Parameters
    ----------
    pipeline : 
        Scikit-Learn pipeline with transformations and selected model
    parameters : 
        Dictionary with parameters names as keys and lists of parameter settings
    search_type : 
        Hyperparameter optimization method (GridSearch or RandomizedSearch)
    n_iter : 
        Number of iterations for the randomized Hyperparameter search
    X_train : 
        Input variables of the training set
    y_train : 
        Target variable of the training set
    
    Returns
    -------
    mean_mae : int
        Mean MAE of the 10-folds of the outer cross-validation
    mean_rmse : int
        Mean RMSE of the 10-folds of the outer cross-validation
    mean_r2_score : float
        Mean R2-score of the 10-folds of the outer cross-validation

    """

    # configure the outer cross-validation procedure
    cv_outer = KFold(n_splits=10, shuffle=True, random_state = random_state)

    # initialize results
    outer_results = list()
    mae_list = []
    rmse_list = []
    r2_list = []

    for train_ix, test_ix in cv_outer.split(X_train):
        
        # split data for outer cross-validation
        X_train_outer, X_test_outer = X_train.values[train_ix, :], X_train.values[test_ix, :]
        y_train_outer, y_test_outer = y_train.values[train_ix], y_train.values[test_ix]

        # Optimize hyperparameters
        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, X_train_outer, y_train_outer, pipeline, parameters), n_trials=n_iter, show_progress_bar=False)

        # Get best model
        pipeline.set_params(**study.best_trial.params)
        best_model = pipeline.fit(X_train_outer, y_train_outer)
            
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test_outer)
            
        # evaluate the model
        mae = mean_absolute_error(y_test_outer, yhat)
        rmse = mean_squared_error(y_test_outer, yhat, squared = False)
        r2 = r2_score(y_test_outer, yhat)

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

        # store the result
        outer_results.append([mae, rmse, r2])
        
        # report progress
        # print('>mae=%.3f, rmse=%.3f, R-squared= %.10f, cfg=%s' % (mae, rmse, r2, result.best_params_))

    return int(np.mean(mae_list)), int(np.mean(rmse_list)), np.round(np.mean(r2_list),4)

def get_best_model(pipeline: Pipeline, 
                   parameters: dict, 
                   X_train: pd.DataFrame, 
                   y_train: pd.DataFrame):
    """
    Get the best model for a given algorithm.

    Performs hyperparameter optimization for a given algorithm.

    Parameters
    ----------
    pipeline : 
        Scikit-Learn pipeline with transformations and selected model
    parameters : 
        Dictionary with parameters names as keys and lists of parameter settings
    X_train : 
        Input variables of the training set
    y_train : 
        Target variable of the training set
    
    Returns
    -------
    model : 
        Trained model with best hyperparameters

    """

    # Optimize hyperparameters
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, X_train, y_train, pipeline, parameters), timeout=60, show_progress_bar=False)

    # Get best model
    pipeline.set_params(**study.best_trial.params)
    model = pipeline.fit(X_train, y_train)

    return model

def eval_classic_ml_models_optuna(X_train: pd.DataFrame, 
                                  y_train: pd.DataFrame, 
                                  X_test: pd.DataFrame, 
                                  y_test: pd.DataFrame, 
                                  summery_file: str, 
                                  column_count: int, 
                                  input_filename: str, 
                                  file_path_pics: str, 
                                  result_df: pd.DataFrame, 
                                  feature_set: str,
                                  config: dict) -> None:
    """
    Runs classical ML algorithms on dataset using Optuna for the hyperparameter optimization

    Implemented algorithms: Polynomial Regression, Decision Tree, Random Forest, Support Vector Regressor, k-Nearest Neighbors and AdaBoost Regression. 
    For each algorithm, a nested cross-validation is performed on the training dataset and the mean RMSE, MAE and R2-score of the 10 folds is computed. 
    The algorithm with the best mean RMSE score for the cross-validation is selected as the best estimator and evaluated on the test set.

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
    column_count: int
        Number of input variables
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
    NUM_OPTUNA_ITER = 20

    SKLEARN_VERSION = tuple([int(x) for x in sklearn.__version__.split(".")])

    print('\n--Optuna optimization for classic approach-- for feature set: {}'.format(feature_set))
    # open the summery file
    f = open(summery_file, "a")

    # create the data list for storing the results of the computation
    data_list = []

    #
    best_model_score = None
    best_model = None

    f.write('\n\n')
    f.write('Feature set: \t' + feature_set)
    f.write('\n')

    #########################################################
    # NESTED CROSS-VALIDATION
    #########################################################

    f.write('\n\n')
    f.write('Select and train different models and calculate the MAE, RMSE and R2-Score for each model:')
    f.write('\n\n')

    ################################
    # Linear Regression
    print("      Linear Regression - ", end = '')
    f.write('***Linear regression model***')
    f.write('\n')
    linReg_pipe = Pipeline([
                    ("poly", PolynomialFeatures(include_bias=False)),
                    ('model', LinearRegression())
                    ])

    linReg_parameters = {
        "poly__degree" : ["int", 1, 3]
        }

    linRegMAE, linRegRMSE, linRegR2Score= cross_validation( pipeline = linReg_pipe, 
                                                            parameters = linReg_parameters, 
                                                            n_iter = NUM_OPTUNA_ITER,
                                                            X_train = X_train, 
                                                            y_train = y_train,
                                                            random_state = RANDOM_STATE)    

    f.write('CV - Mean MAE: \t' + str(linRegMAE))
    f.write('\nCV - Mean RMSE: \t' + str(linRegRMSE))
    f.write('\nCV - Mean R2 score: \t' + str(linRegR2Score))
    f.write('\n') 

    data_list.append(linRegMAE) 
    data_list.append(linRegRMSE) 
    data_list.append(linRegR2Score)   

    best_model_score = linRegRMSE
    best_model = "linReg"

    ###############################
    # Decision Tree
    print("Decision Tree - ", end = '')
    f.write('***Decision Tree model***')
    f.write('\n')
    tree_pipe = Pipeline([
                    ('model', DecisionTreeRegressor())
                    ])

    if SKLEARN_VERSION >= (0, 25, 0):
        tree_parameters = {"model__max_depth" : ["int", 2, 64],
                            "model__criterion": ["categorical", "squared_error", "absolute_error", "poisson"]}
    else:                        
        tree_parameters = {"model__max_depth" : ["int", 2, 64],
                            "model__criterion": ["categorical", "mae", "mse"]}

     

    treeMAE, treeRMSE, treeR2Score= cross_validation(   pipeline = tree_pipe, 
                                                        parameters = tree_parameters,
                                                        n_iter = NUM_OPTUNA_ITER, 
                                                        X_train = X_train, 
                                                        y_train = y_train,
                                                        random_state = RANDOM_STATE)    

    f.write('CV - Mean MAE: \t' + str(treeMAE))
    f.write('\nCV - Mean RMSE: \t' + str(treeRMSE))
    f.write('\nCv - Mean R2 score: \t' + str(treeR2Score))
    f.write('\n') 

    data_list.append(treeMAE) 
    data_list.append(treeRMSE) 
    data_list.append(treeR2Score) 

    if treeRMSE < best_model_score:
        best_model_score = treeRMSE
        best_model = 'tree'

    ###############################
    # Random Forest
    print("Random Forest - ", end = '')
    f.write('***Random Forest model***')
    f.write('\n')
    randForest_pipe = Pipeline([
                    ('model', RandomForestRegressor())
                    ])

    if SKLEARN_VERSION >= (0, 25, 0):
        randForest_parameters = {"model__n_estimators" : ["int", 1, 200],
                                 "model__max_features": ["int", 1, column_count],
                                 'model__min_samples_split': ["int", 2, 12],
                                 "model__criterion": ["categorical", "squared_error", "absolute_error", "poisson"],
                                 "model__max_depth": ["int", 2, 64],
                                 'model__bootstrap': ["categorical", True, False]}
    else:
        randForest_parameters = {"model__n_estimators" : ["int", 1, 200],
                                 "model__max_features": ["int", 1, column_count],
                                 'model__min_samples_split': ["int", 2, 12],
                                 "model__criterion": ["categorical", "mse", "mae"],
                                 "model__max_depth": ["int", 2, 64],
                                 'model__bootstrap': ["categorical", True, False]}

    randForestMAE, randForestRMSE, randForestR2Score= cross_validation( pipeline = randForest_pipe, 
                                                                        parameters = randForest_parameters, 
                                                                        n_iter = NUM_OPTUNA_ITER,
                                                                        X_train = X_train, 
                                                                        y_train = y_train,
                                                                        random_state = RANDOM_STATE)    

    f.write('CV - Mean MAE: \t' + str(randForestMAE))
    f.write('\nCV - Mean RMSE: \t' + str(randForestRMSE))
    f.write('\nCV - Mean R2 score: \t' + str(randForestR2Score))
    f.write('\n') 

    data_list.append(randForestMAE) 
    data_list.append(randForestRMSE) 
    data_list.append(randForestR2Score) 

    if randForestRMSE < best_model_score:
        best_model_score = randForestRMSE
        best_model = 'randForest'

    ###############################
    # SVR
    print("SVR - ", end = '')
    f.write('***Support Vector Regressor model***')
    f.write('\n')
    svrReg_pipe = Pipeline([
                    ('model', SVR())
                    ])

    svrReg_parameters = {"model__kernel" : ["categorical", 'linear', 'poly', 'rbf'],
                         "model__C": ["float", 1e-3, 1e3],
                         "model__epsilon": ["float", 1e-5, 1e-1]}

    svrRegMAE, svrRegRMSE, svrRegR2Score= cross_validation(pipeline = svrReg_pipe,
                                                            parameters = svrReg_parameters, 
                                                            n_iter = NUM_OPTUNA_ITER, 
                                                            X_train = X_train, 
                                                            y_train = y_train,
                                                            random_state = RANDOM_STATE)    

    f.write('CV - Mean MAE: \t' + str(svrRegMAE))
    f.write('\nCV - Mean RMSE: \t' + str(svrRegRMSE))
    f.write('\nCV - Mean R2 score: \t' + str(svrRegR2Score))
    f.write('\n') 

    data_list.append(svrRegMAE) 
    data_list.append(svrRegRMSE) 
    data_list.append(svrRegR2Score) 

    if svrRegRMSE < best_model_score:
        best_model_score = svrRegRMSE
        best_model = 'svrReg'

    ###############################
    # KNN
    print("KNN - ", end = '')
    f.write('***kNN Regressor model***')
    f.write('\n')
    knnReg_pipe = Pipeline([
                    ('model', KNeighborsRegressor())
                    ])

    knnReg_parameters = {"model__n_neighbors" : ["int", 2, 20],
                         "model__weights": ["categorical", 'uniform', 'distance'],
                         "model__p": ["int", 1,3]}

    knnRegMAE, knnRegRMSE, knnRegR2Score= cross_validation(pipeline = knnReg_pipe,
                                                            parameters = knnReg_parameters,
                                                            n_iter = NUM_OPTUNA_ITER,
                                                            X_train = X_train, 
                                                            y_train = y_train,
                                                            random_state = RANDOM_STATE)  

    f.write('CV - Mean MAE: \t' + str(knnRegMAE))
    f.write('\nCV - Mean RMSE: \t' + str(knnRegRMSE))
    f.write('\nCV - Mean R2 score: \t' + str(knnRegR2Score))
    f.write('\n') 

    data_list.append(knnRegMAE) 
    data_list.append(knnRegRMSE) 
    data_list.append(knnRegR2Score) 

    if knnRegRMSE < best_model_score:
        best_model_score = knnRegRMSE
        best_model = 'knnReg'

    ###############################
    # AdaBoost Regressor
    print("AdaBoost Regressor")
    f.write('***AdaBoost Regressor model***')
    f.write('\n')
    adaReg_pipe = Pipeline([
                    ('model', AdaBoostRegressor(random_state = RANDOM_STATE))
                    ])

    adaReg_parameters = {"model__n_estimators" : ["int", 1, 200]}

    adaRegMAE, adaRegRMSE, adaRegR2Score= cross_validation(pipeline = adaReg_pipe,
                                                            parameters = adaReg_parameters, 
                                                            n_iter = NUM_OPTUNA_ITER,
                                                            X_train = X_train, 
                                                            y_train = y_train,
                                                            random_state = RANDOM_STATE)    

    f.write('CV - Mean MAE: \t' + str(adaRegMAE))
    f.write('\nCV - Mean RMSE: \t' + str(adaRegRMSE))
    f.write('\nCV - Mean R2 score: \t' + str(adaRegR2Score))
    f.write('\n') 

    data_list.append(adaRegMAE) 
    data_list.append(adaRegRMSE) 
    data_list.append(adaRegR2Score) 

    if adaRegRMSE < best_model_score:
        best_model_score = adaRegRMSE
        best_model = 'adaReg'

    
    ##########################
    # Analyze  the best model
    #########################

    f.write('-----------------------\n')
    f.write('Analyze  the best model\n')
    f.write('------------------------\n\n')

    # Training starting time
    train_start_time = time()

    if best_model == "linReg":
        final_model = get_best_model(linReg_pipe, linReg_parameters,  X_train, y_train)
    elif best_model == "tree":
        final_model = get_best_model(tree_pipe, tree_parameters, X_train, y_train)
    elif best_model == "randForest":
        final_model = get_best_model(randForest_pipe, randForest_parameters, X_train, y_train)
    elif best_model == "svrReg":
        final_model = get_best_model(svrReg_pipe, svrReg_parameters, X_train, y_train)
    elif best_model == "knnReg":
        final_model = get_best_model(knnReg_pipe, knnReg_parameters, X_train, y_train)
    elif best_model == "adaReg":
        final_model = get_best_model(adaReg_pipe, adaReg_parameters, X_train, y_train)

    # Training stop time
    train_stop_time = time()

    # training duration 
    training_duration = train_stop_time - train_start_time

    data_list.append(str(final_model['model']))

    f.write('Final model: ' + str(final_model))
    f.write('\n\n')

    ######################################
    # Evaluate the system on the test set
    ######################################

    f.write('-----------------------------------\n')
    f.write('Evaluate the system on the test set\n')
    f.write('-----------------------------------\n\n')

    # # Do the final test with the test set with the best estimator!
    # machine_type_X_test = X_test_preprep.drop('price', axis = 1)
    # y_test = X_test_preprep['price'].copy()

    # X_test = full_pipeline.transform(machine_type_X_test)

    # Test start time
    test_start_time = time()

    final_predictions = final_model.predict(X_test)

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
    #plt.savefig(file_path_pics+'/'+'Test5.png')
    filename = "{}-{}-{}-{}.{}".format(M_DATE,input_filename,'optuna-test-result',feature_set,'png')
    title = str("Optuna optimized classic results for {} on feat. set {}".format(input_filename, feature_set))
    plt.title(title)
    plt.savefig(file_path_pics+'/'+filename)
    # close open windows 
    plt.close()

    ###################
    # Model evaluation
    ###################
    # Calculate MAE and MEPE according to https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error
    mean_abs_error = mean_absolute_error(y_test, final_predictions)
    mean_abs_percentage_error = mean_absolute_percentage_error(y_test, final_predictions)

    # calculate RMSE value and its derivative according to https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization
    # RMSE
    RMSE = np.sqrt(mean_squared_error(y_test, final_predictions))
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

    final_test_score = final_model.score(X_test, y_test)
    f.write('Mean Absolute Error: \t' + str(np.round(mean_abs_error, 2)))
    f.write('\n')
    f.write('Mean Absolute Percentage Error: \t' + str(np.round(mean_abs_percentage_error, 4)))
    f.write('\n')
    f.write('Root Mean Squared Error: \t' + str(np.round(RMSE, 2)))
    f.write('\n')
    f.write('Root Mean Squared Error: \t' + str(np.round(N_RMSE, 4)))
    f.write('\n')
    f.write('Root Mean Squared Error: \t' + str(np.round(IQR_RMSE, 4)))
    f.write('\n')
    f.write('Root Mean Squared Error: \t' + str(np.round(CV_RMSE, 4)))
    f.write('\n')
    f.write('Final test score: \t' + str(np.round(final_test_score, 3)))
    f.write('\n\n')

    # add estimator score to data_list
    data_list.append(np.round(mean_abs_error, 2))
    data_list.append(np.round(mean_abs_percentage_error, 4))
    data_list.append(np.round(RMSE, 2))
    data_list.append(np.round(N_RMSE, 4))
    data_list.append(np.round(IQR_RMSE, 4))
    data_list.append(np.round(CV_RMSE, 4))
    data_list.append(round(final_test_score, 4))

    #store training duration
    data_list.append(np.round(training_duration, 3))

    # store test duration for a sigle test point
    data_list.append(np.round(test_time_per_sample, 10))

    # put the data into the result file
    result_df[feature_set] = data_list

    # close the summery file
    f.close()