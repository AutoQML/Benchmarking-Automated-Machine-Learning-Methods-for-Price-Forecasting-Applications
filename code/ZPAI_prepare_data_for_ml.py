from pathlib import Path
import pandas as pd
import numpy as np
import itertools

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def features_pca(df: pd.DataFrame,
                 pca_num: int) -> list:
    """
    Gets the most important model-specific features.

    For a given dataset, gets all the model-specific features: all features except for base features (model, extension, location, price, const_year and working_hours).
    If there are more model-specific features than the given pca_num parameter, PCA analysis is performed to determine the pca_num most important model-specific features.
    List with the name of the base features + name of the model-specific feature from the PCA analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the dataset.
    pca_num : int
        Number of model-specific features.

    Returns
    -------
    column_names : list of str
        List of columns selected after the PCA analysis.

    """
    if 'model' in df.columns:
        # extract price column, const_year & working_hours columns from dataset before combinatorial processing - these features are fixed for all combinations
        df_temp = df.drop(['model','price', 'const_year', 'working_hours'], axis=1)

        # DataFrame for PCA only with model-specific features
        df_pca = df.drop(['model','extension', 'location', 'price', 'const_year', 'working_hours'], axis=1)
    else:
        df_temp = df.drop(['price', 'const_year', 'working_hours'], axis=1)
        df_pca = df.drop(['extension', 'location', 'price', 'const_year', 'working_hours'], axis=1)

    feature_list_important = []
    # Select the most important PCA_NUM features beside 'extension', 'location', 'const_year' & 'working_hours' by PCA
    if(len(df_pca.columns) > pca_num):

        pca = PCA(n_components=pca_num).fit(df_pca)

        # number of components
        n_pcs= pca.components_.shape[0]

        # get the index of the most important feature on EACH component i.e. largest absolute value
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

        # get the names
        initial_feature_names = list(df_pca.columns)
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

        feature_list_important = [most_important_names[i] for i in range(n_pcs)]
        feature_list_to_delete = [e for e in initial_feature_names if e not in feature_list_important]

        df_temp = df_temp.drop(feature_list_to_delete, axis=1)


    # extract the names of the columns
    column_names = list(df_temp.columns)

    return column_names

def build_dataset_from_subset(df_original: pd.DataFrame, subset: tuple):
    """
    Creates subset of dataset.

    For a given tuple of columns of the original dataset, creates subset of the original dataset.

    Parameters
    ----------
    df_original : pd.DataFrame
        DataFrame with original dataset.
    subset : tuple
        Tuple with the names of the columns of the subset.

    Returns
    -------
    df_subset : pd.DataFrame
        DataFrame with the subset of the original dataset.
    feature_set : str
        String with the name of the subset.

    """
    # extract the feature (name) combinations - name of the columns of the result.csv file
    feature_set = "-".join([str(x) for x in subset])
    if not feature_set:
        feature_set = 'basic-subset'

    # rebuild the column list
    col_list = list(subset)
    col_list.append('price')
    col_list.append('const_year')
    col_list.append('working_hours')

    if "model" in df_original.columns:
        col_list.append('model')

    df_subset = df_original[col_list].copy()

    return df_subset, feature_set

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features as one-hot encoding.

    Checks if a categorical feature (location, extension or model) exists in the given dataset.
    If categorical features exists, features are encoded based on one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with dataset to be encoded.

    Returns
    -------
    df_encoded : pd.DataFrame
        DataFrame containing the dataset with encoded categorical features.

    """
    col_list = df.columns
    categorical_cols =[]
    if 'location' in col_list:
        categorical_cols.append('location')
    if 'extension' in col_list:
        categorical_cols.append('extension')
    if 'model' in col_list:
        categorical_cols.append('model')

    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols) # Preprocess categorical attributes
    else:
        df_encoded = df.copy()

    return df_encoded


def prepare_data_for_ml(machine_model: str,
                        df_dataset: pd.DataFrame,
                        file_path_pics: str,
                        file_path_data: str,
                        input_filename: str,
                        summery_file: str,
                        config: dict,
                        random_state: int, 
                        measurements: int) -> None:
    """
    Prepares dataset and calls evaluation function.

    Generates list of n + PCA_NUM features, where n=2 for individual files (const_year & working_hours) and n=3 for merged files.
    For each possible combination of the features, a subset of the original dataset is generated.
    Categorical features are encoded and numerical features are type casted to integers.
    Dataset is splitted into training (90%) and test set (10%).
    All but the "price" column are selected as inputs. "Price" is selected as output.
    Numerical features (working_hours & const_year) are scaled using StandardScaler.
    Depending on the configurations, selected evaluation function (classical ML/neural nets/autosklearn) is called.
    Results are saved as .csv file.

    Parameters
    ----------
    machine_model : str
        Name of the current machine model.
    df_dataset : pd.DataFrame
        Dataset to be processed.
    file_path_pics : str
        Path to the folder where the plots should be saved.
    file_path_data : str
        Path to the folder where the results should be saved.
    input_filename : str
        Filename of the dataset file. Name of output files are based on the name of the input file.
    config : dict
        Dictionary with the global configurations.

    Returns
    -------


    """

    MACHINE_MODEL = machine_model
    df_dataset = df_dataset
    FILE_PATH_PICS = file_path_pics
    FILE_PATH_DATA = file_path_data
    SUMMERY_FILE = summery_file
    input_filename = input_filename

    # Get parameters from configuration file
    PCA_NUM = config["general"]["pca_num"]
    RANDOM_STATE = random_state
    MEASUREMENTS = measurements
    M_DATE = config["general"]["start_date"]
    MY_OS = config["general"]["operating_system"]
    ALGORITHMS = config["general"]["algorithms"]
    PYTHON_ENV = config["general"]["python_env"]
    AUTOML_PREPROCESS = config["general"]["automl_preprocessing"]

    ##################
    # for testing
    ##################
    loop_count = 0

    if "classical" in ALGORITHMS:
        # create result data frame to store the measurements for classic approach
        classic_index = ['CV - LinReg - Mean MAE', 'CV - LinReg - Mean MAPE', 'CV - LinReg - Mean RMSE', 'CV - LinReg - Mean R2Score',
                        'CV - Tree - Mean MAE', 'CV - Tree - Mean MAPE', 'CV - Tree - Mean RMSE', 'CV - Tree - Mean R2Score',
                        'CV - RandForest - Mean MAE', 'CV - RandForest - Mean MAPE', 'CV - RandForest - Mean RMSE', 'CV - RandForest - Mean R2Score',
                        'CV - SVR - Mean MAE', 'CV - SVR - Mean MAPE', 'CV - SVR - Mean RMSE', 'CV - SVR - Mean R2Score',
                        'CV - KNN - Mean MAE', 'CV - KNN - Mean MAPE', 'CV - KNN - Mean RMSE', 'CV - KNN - Mean R2Score',
                        'CV - AdaBoost - Mean MAE', 'CV - AdaBoost - Mean MAPE', 'CV - AdaBoost - Mean RMSE', 'CV - AdaBoost - Mean R2Score',
                        'final-model', 'Test-MAE', 'Test-MAPE','Test-RMSE', 'Test-N-RMSE', 'Test-IQR-RMSE', 'Test-CV-RMSE', 'Test-R2', 'Training-Duration', 'Test-Duration']
        # # create result data frame
        classic_result_df = pd.DataFrame(index=classic_index)

    if "nn" in ALGORITHMS:
        # create result data frame to store the measurements for NN
        nn_index = ['Test-MAE', 'Test-MAPE', 'Test-RMSE', 'Test-N-RMSE', 'Test-IQR-RMSE', 'Test-CV-RMSE', 'Test-R2', 'Training-Duration', 'Test-Duration', 'Activation', 'Hidden-layer-size', 'Learning rate', 'Solver']
        # # create result data frame
        nn_result_df = pd.DataFrame(index=nn_index)

    if "autosklearn" in ALGORITHMS:
        # create result data frame to store the measurements for autosklearn
        autosklearn_index = ['Test-MAE', 'Test-MAPE', 'Test-RMSE', 'Test-N-RMSE', 'Test-IQR-RMSE', 'Test-CV-RMSE', 'Test-R2', 'Training-Duration', 'Test-Duration']
        # # create result data frame
        autosklearn_result_df = pd.DataFrame(index=autosklearn_index)

    if "autogluon" in ALGORITHMS:
        # create result data frame to store the measurements for autosklearn
        autogluon_index = ['Test-MAE', 'Test-MAPE', 'Test-RMSE', 'Test-N-RMSE', 'Test-IQR-RMSE', 'Test-CV-RMSE', 'Test-R2', 'Training-Duration', 'Test-Duration']
        # # create result data frame
        autogluon_result_df = pd.DataFrame(index=autogluon_index)

    if "flaml" in ALGORITHMS:
        # create result data frame to store the measurements for autosklearn
        flaml_index = ['Test-MAE', 'Test-MAPE', 'Test-RMSE', 'Test-N-RMSE', 'Test-IQR-RMSE', 'Test-CV-RMSE', 'Test-R2', 'Training-Duration', 'Test-Duration']
        # # create result data frame
        flaml_result_df = pd.DataFrame(index=flaml_index)

    ###########################
    # Data preparation
    ###########################

    column_names = features_pca(df = df_dataset,
                                pca_num = PCA_NUM)

    # combinatorial walk through all possible combinations of the features / feature columns
    for L in range(0, len(column_names)+1):
        # combinatorial walk through all possible combinations of the features / feature columns
        # L = 0 just use the set features const_year & working_hours
        # L = 1 use one additional feature (each of them in combination with const_year & working_hours)
        # L = 2 use two additional features
        # L = len(column_names)+1 use all features
        for subset in itertools.combinations(column_names, L):

            loop_count = loop_count + 1

            # create datasets with different feature set combinations
            df_dataset_subset, feature_set = build_dataset_from_subset(df_original = df_dataset,
                                                                       subset = subset)

            ###########################
            # Encoding
            ###########################
            # Encode categorical attributes
            df_dataset_subset_encoded = encode_categorical_features(df = df_dataset_subset) # Encode location and extension
            df_dataset_preprocessed = df_dataset_subset_encoded.astype(int) # Numerical values as int

            # calculate the number of columns without the price column (therfore  -1)
            column_count = len(df_dataset_preprocessed.columns) -1

            #########################
            # Splitting
            ########################
            # Split the data into training and test set
            df_dataset_train, df_dataset_test = train_test_split(df_dataset_preprocessed,
                                                                 test_size=0.1,
                                                                 random_state=RANDOM_STATE)
            df_dataset_X_train = df_dataset_train.drop('price', axis = 1)
            df_dataset_y_train = df_dataset_train['price'].copy()
            df_dataset_X_test = df_dataset_test.drop('price', axis = 1)
            df_dataset_y_test = df_dataset_test['price'].copy()

            #####################
            # Scaling
            #####################

            # Build pipelines for preprocessing the attributes. Use sklearn Pipeline for pipelines and  sklearn StandardScaler for scaling the values of the attributes.
            full_pipeline = ColumnTransformer([
                    ("num", StandardScaler(), ['working_hours', 'const_year'])
                ], remainder='passthrough')

            # Prepare the training data X_train
            df_dataset_X_train_final = full_pipeline.fit_transform(df_dataset_X_train)
            df_dataset_X_train_final = pd.DataFrame(df_dataset_X_train_final)
            df_dataset_X_test_final = full_pipeline.transform(df_dataset_X_test)
            # df_dataset_X_test_final = pd.DataFrame(df_dataset_X_test_final)

            # store prepared data as csv
            filename = "{}-{}-{}.{}".format(M_DATE,input_filename,'prepdata','csv')
            PREPDATE_CSV = Path(FILE_PATH_DATA, filename)
            df_dataset_X_train_final.to_csv(PREPDATE_CSV)

            # Get training and test data
            df_dataset_X_train = df_dataset_X_train_final.copy()
            df_dataset_X_test = df_dataset_X_test_final.copy()

            ##############################
            # AutoML
            ##############################
            # Split the data into train and test sets for AutoML methods
            # omit encoding and scaling for automl methods

            df_dataset_train_automl, df_dataset_test_automl = train_test_split(df_dataset_subset,
                                                                 test_size=0.1,
                                                                 random_state=RANDOM_STATE)
            df_dataset_X_train_automl = df_dataset_train_automl.drop('price', axis = 1)
            df_dataset_y_train_automl = df_dataset_train_automl['price'].copy()
            df_dataset_X_test_automl = df_dataset_test_automl.drop('price', axis = 1)
            df_dataset_y_test_automl = df_dataset_test_automl['price'].copy()

            # set preprocessed dataset for automl methods if configured
            if AUTOML_PREPROCESS == True:
                # for auto-sklearn & FLAML
                df_dataset_X_train_automl = df_dataset_X_train
                df_dataset_y_train_automl = df_dataset_y_train
                df_dataset_X_test_automl = df_dataset_X_test
                df_dataset_y_test_automl = df_dataset_y_test
                # for autogluon
                df_dataset_train_automl = df_dataset_train
                df_dataset_test_automl = df_dataset_test

            if "classical" in ALGORITHMS:
                if PYTHON_ENV == 'automl-autosklearn':
                    # evaluate classical ml models like lin. regression, trees, forests, SVM
                    from ZPAI_evaluate_classic_ml_models import eval_classic_ml_models
                    eval_classic_ml_models(X_train = df_dataset_X_train,
                                        y_train = df_dataset_y_train,
                                        X_test = df_dataset_X_test,
                                        y_test = df_dataset_y_test,
                                        summery_file = SUMMERY_FILE,
                                        column_count = column_count,
                                        input_filename = input_filename,
                                        file_path_pics = FILE_PATH_PICS,
                                        result_df = classic_result_df,
                                        feature_set = feature_set,
                                        config = config)
                else:
                    print('Wrong python environment!')

            if "nn" in ALGORITHMS:
                if PYTHON_ENV == 'automl-autosklearn':
                    # evaluate MLP
                    from ZPAI_evaluate_neural_nets import evaluate_neural_nets
                    evaluate_neural_nets(X_train = df_dataset_X_train,
                                        y_train = df_dataset_y_train,
                                        X_test = df_dataset_X_test,
                                        y_test = df_dataset_y_test,
                                        summery_file = SUMMERY_FILE,
                                        input_filename = input_filename,
                                        file_path_pics = FILE_PATH_PICS,
                                        result_df = nn_result_df,
                                        feature_set = feature_set,
                                        config = config)
                else:
                    print('Wrong python environment!')

            if "autosklearn" in ALGORITHMS:
                if PYTHON_ENV == 'automl-autosklearn':
                    # evaluate AutoML - autosklearn
                    # check for OS - autosklearn is not running on MAC (Darwin) at the moment
                    if MY_OS == 'Linux':
                        from ZPAI_evaluate_autosklearn import evaluate_autosklearn
                        evaluate_autosklearn(X_train = df_dataset_X_train_automl,
                                            y_train = df_dataset_y_train_automl,
                                            X_test = df_dataset_X_test_automl,
                                            y_test = df_dataset_y_test_automl,
                                            summery_file = SUMMERY_FILE,
                                            input_filename = input_filename,
                                            file_path_pics = FILE_PATH_PICS,
                                            file_path_data = FILE_PATH_DATA,
                                            result_df = autosklearn_result_df,
                                            feature_set = feature_set,
                                            config = config,
                                            loop_count = loop_count, 
                                            measurements = MEASUREMENTS)
                    if MY_OS == 'Darwin':
                        print("System OS: ",MY_OS)
                else:
                    print('Wrong python environment!')

            if "autogluon" in ALGORITHMS:
                if PYTHON_ENV == 'automl-autogluon':
                    # evaluate AutoML - autogluon
                    # check for OS - autogluon is not running on MAC (Darwin) at the moment
                    if MY_OS == 'Linux':
                        from ZPAI_evaluate_autogluon import evaluate_autogluon
                        evaluate_autogluon(X_train = df_dataset_train_automl,
                                            X_test = df_dataset_test_automl,
                                            summery_file = SUMMERY_FILE,
                                            input_filename = input_filename,
                                            file_path_pics = FILE_PATH_PICS,
                                            file_path_data = FILE_PATH_DATA,
                                            result_df = autogluon_result_df,
                                            feature_set = feature_set,
                                            config = config)
                    if MY_OS == 'Darwin':
                        print("System OS: ",MY_OS)
                else:
                    print('Wrong python environment!')

            if "flaml" in ALGORITHMS:
                if PYTHON_ENV == 'automl-autosklearn':
                    # evaluate NN
                    from ZPAI_evaluate_flaml import evaluate_flaml
                    evaluate_flaml(X_train = df_dataset_X_train_automl,
                                        y_train = df_dataset_y_train_automl,
                                        X_test = df_dataset_X_test_automl,
                                        y_test = df_dataset_y_test_automl,
                                        summery_file = SUMMERY_FILE,
                                        input_filename = input_filename,
                                        file_path_pics = FILE_PATH_PICS,
                                        file_path_data = FILE_PATH_DATA,
                                        result_df = flaml_result_df,
                                        feature_set = feature_set,
                                        config = config)
                else:
                    print('Wrong python environment!')


    if "classical" in ALGORITHMS:
        # store classical results within the results.csv
        filename = "{}-{}-{}.{}".format(M_DATE, input_filename, 'classic-results','csv')
        RESULT_CSV = Path(FILE_PATH_DATA, filename)
        classic_result_df.to_csv(RESULT_CSV)

    if "nn" in ALGORITHMS:
        # store NN results within the results.csv
        filename = "{}-{}-{}.{}".format(M_DATE, input_filename, 'nn-results','csv')
        RESULT_CSV = Path(FILE_PATH_DATA, filename)
        nn_result_df.to_csv(RESULT_CSV)

    if "autosklearn" in ALGORITHMS:
        # store autosklearn results within the results.csv
        filename = "{}-{}-{}.{}".format(M_DATE, input_filename, 'autosklearn-results','csv')
        RESULT_CSV = Path(FILE_PATH_DATA, filename)
        autosklearn_result_df.to_csv(RESULT_CSV)

    if "autogluon" in ALGORITHMS:
        # store autogluon results within the results.csv
        filename = "{}-{}-{}.{}".format(M_DATE, input_filename, 'autogluon-results','csv')
        RESULT_CSV = Path(FILE_PATH_DATA, filename)
        autogluon_result_df.to_csv(RESULT_CSV)

    if "flaml" in ALGORITHMS:
        # store autogluon results within the results.csv
        filename = "{}-{}-{}.{}".format(M_DATE, input_filename, 'flaml-results','csv')
        RESULT_CSV = Path(FILE_PATH_DATA, filename)
        flaml_result_df.to_csv(RESULT_CSV)
