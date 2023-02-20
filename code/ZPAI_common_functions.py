import pandas as pd
from datetime import date
from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# load CSV data file
def load_csv_data(csv_path):
    return pd.read_csv(csv_path, delimiter=';')

# Define the helper function `display_scores` to print the results of the cross validation
def display_scores(scores, f):
        # f.write('\n\n')
        # f.write('Scores: ' + str(scores))
        f.write('\n')
        f.write('Mean: \t' + str(int(scores.mean())))
        f.write('\n')
        f.write('StD: \t' + str(int(scores.std())))
        f.write('\n\n')

def read_yaml(path):
    """
    Safe load yaml file
    """
    with open(path, 'r') as f:
        file = yaml.safe_load(f)
    return file

def get_current_date() -> str:
    """
    Get current date as string
    """
    today = date.today()
    m_date = today.strftime("%Y-%m-%d")
    return m_date

def create_path(path: str, verbose: bool):
    """
    Checks if directory exists and, if not, creates directory
    """
    if not Path.exists(path):
        Path.mkdir(path)
        if verbose:
            print("Directory " , path ,  " Created ")
    else:
        if verbose:    
            print("Directory " , path ,  " already exists")

def calculate_scores(y_test, y_predict):
    # Calculate MAE and MEPE according to https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error
    mean_abs_error = mean_absolute_error(y_test, y_predict)
    mean_abs_percentage_error = mean_absolute_percentage_error(y_test, y_predict)
    r2_score_value = r2_score(y_test, y_predict) 

    # calculate RMSE value and its derivative according to https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization
    # RMSE
    root_mean_squared_error = np.sqrt(mean_squared_error(y_test, y_predict))

    return(mean_abs_error, mean_abs_percentage_error, r2_score_value, root_mean_squared_error)