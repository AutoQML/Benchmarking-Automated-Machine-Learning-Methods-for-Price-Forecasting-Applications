import argparse
import yaml
import platform
from pathlib import Path
import os

from ZPAI_common_functions import get_current_date, create_path, read_yaml
from ZPAI_evaluate_dataset import evaluate_data
from ZPAI_document_results_docx import document_results_docx

def create_parser():
    parser = argparse.ArgumentParser(description="Process inputs")
    parser.add_argument("--pca", type = int, help = "Number of selected PCA components")
    parser.add_argument("--algorithms", choices = ["classical", "nn", "autosklearn", "autogluon", "flaml"], nargs="*", help = "Type of algorithms to run.")
    parser.add_argument("--models", choices = ['308', '320', '323', '329', '330', '336', '950', '966', 'D6', 'M318', 'merged', 'all'], nargs="*", help = "Selected models list")
    parser.add_argument("--outlier_detection", choices = ["True", "False"], help = "Remove outliers from dataset")
    parser.add_argument("--document_results", choices = ["True", "False"], help = "Document the results")
    parser.add_argument("--autosk_time_for_task", type = int, help = "Time limit in seconds for the search of appropriate models")
    parser.add_argument("--autosk_runtime_limit", type = int, help = "Time limit for a single call to the machine learning model")
    parser.add_argument("--start_date", type = str, help = "Start date of measurement")
    parser.add_argument("--measurements", type = int, help = "Number of measurements")
    parser.add_argument("--automl_preprocessing", type = bool, help = "Set autoML preprocessing")
    parser.add_argument("--evaluate_dataset_variance", type = bool, help = "Set evaluation of data set variance")
    parser.add_argument("--random_state", type = int, help = "Random state")

    return parser

def get_config_from_parser(parser, config):
    args = parser.parse_args()

    # List of models
    if args.models:
        if "merged" in args.models:
            config["model"]["model"]["model_list"] = ["merged-files"]
        elif "all" in args.models:
            config["model"]["model"]["model_list"] = ['Caterpillar-308', 'Caterpillar-320', 'Caterpillar-323', 'Caterpillar-329', 'Caterpillar-330', 'Caterpillar-336', 'Caterpillar-950', 'Caterpillar-966', 'Caterpillar-D6', 'Caterpillar-M318']
        else:
            models_with_prefix = ["Caterpillar-" + model for model in args.models]
            config["model"]["model"]["model_list"] = models_with_prefix

    # Algorithms
    if args.algorithms:
        config["general"]["algorithms"] = args.algorithms

    # Outlier Detection
    if args.outlier_detection:
        if args.outlier_detection == "True":
            config["general"]["bin_outlier_detect"] = True
        else:
            config["general"]["bin_outlier_detect"] = False

    # Documentation
    if args.document_results:
        if args.document_results == "True":
            config["general"]["documentation"] = True
        else:
            config["general"]["documentation"] = False

    # PCA Num
    if args.pca:
        config["general"]["pca_num"] = args.pca

    # Autosklearn
    if args.autosk_time_for_task:
        config["autosklearn"]["params"]["time_for_task"] = args.autosk_time_for_task

    if args.autosk_runtime_limit:
        config["autosklearn"]["params"]["run_time_limit"] = args.autosk_runtime_limit

    # date
    if args.start_date:
        config["general"]["start_date"] = args.start_date

    # Number of measuremets
    if args.measurements:
        config["general"]["measurement_num"] = args.measurements

    # autoML preprocessing
    if args.automl_preprocessing:
        config["general"]["automl_preprocessing"] = args.automl_preprocessing

    # Set evaluation of data set variance
    if args.evaluate_dataset_variance:
        config["general"]["evaluate_dataset_variance"] = args.evaluate_dataset_variance

    # Random State
    if args.random_state:
        config["general"]["random_state"] = args.random_state

    return config


def main():
    parser = create_parser()

    ######################################
    # LOAD CONFIGURATIONS
    ######################################
    REPO_PATH = Path(__file__).parents[1]

    # Load configuration files
    general_conf = read_yaml(REPO_PATH / 'conf/general_config.yml')
    model_conf = read_yaml(REPO_PATH / 'conf/model_config.yml')
    autosklearn_conf = read_yaml(REPO_PATH / 'conf/auto_sklearn_config.yml')

    # Create global configuration file
    CFG = dict()
    CFG["general"] = general_conf
    CFG["model"] = model_conf
    CFG["autosklearn"] = autosklearn_conf

    CFG["general"]["repo_path"] = Path(__file__).parents[1]
    CFG["general"]["start_date"] = get_current_date()
    CFG["general"]["operating_system"] = platform.system()
    # add python environment
    CFG["general"]["python_env"] = os.environ['CONDA_DEFAULT_ENV']

    CFG = get_config_from_parser(parser, CFG)

    # Get general configuration parameters
    REPO_PATH = CFG["general"]["repo_path"]
    PCA_NUM = CFG["general"]['pca_num']   # get the PCA number
    RANDOM_STATE = CFG["general"]['random_state'] # get the random state
    BIN_OUTLIER = CFG["general"]['bin_outlier_detect'] # get bin outlier detection & deletion state
    SELECTED_MACHINE_MODELS = CFG["model"]['model']['model_list'] # get list of selected models
    M_DATE = CFG["general"]["start_date"]
    DOCUMENTATION = CFG["general"]["documentation"]
    ALGORITHMS = CFG["general"]["algorithms"]
    NUM_OF_MEASUREMENTS = CFG["general"]["measurement_num"]

    ######################################
    # INITIALIZE COMMON SUMMARY
    ######################################

    # File path for the common summary
    SUMMERY_FILE_PATH = Path(REPO_PATH, 'measurements', 'summery')
    if not Path.exists(SUMMERY_FILE_PATH):
        create_path(path = SUMMERY_FILE_PATH, verbose = False)

    # File path within the summery directory for each measurement
    EXPLICIT_SUMMERY_FILE_PATH = Path(REPO_PATH, 'measurements', 'summery', M_DATE)
    if not Path.exists(EXPLICIT_SUMMERY_FILE_PATH):
        create_path(path = EXPLICIT_SUMMERY_FILE_PATH, verbose = False)

    # create summery txt file
    filename = "{}-{}.{}".format(M_DATE,'summery','txt')
    GLOBAL_TXT_SUMMERY_FILE = Path(EXPLICIT_SUMMERY_FILE_PATH, filename)
    if not Path.exists(GLOBAL_TXT_SUMMERY_FILE):
        with open(GLOBAL_TXT_SUMMERY_FILE, "w") as f:
            f.write("Measuremt date: " + M_DATE + "\n")
            f.write("Random seed: " + str(RANDOM_STATE) + "\n")
            f.write("PCA number: " + str(PCA_NUM) + "\n")
            f.write("Models: " + str(SELECTED_MACHINE_MODELS) + "\n")
            if "autosklearn" in ALGORITHMS:
                f.write("Auto-sklearn runtime " + str(autosklearn_conf['params']['time_for_task']) + "\n")
                f.write("Auto-sklearn limit : " + str(autosklearn_conf['params']['run_time_limit']) + "\n")


    # create summery yaml file
    filename = "{}-{}.{}".format(M_DATE,'summery','yml')
    GLOBAL_YAML_SUMMERY_FILE = Path(EXPLICIT_SUMMERY_FILE_PATH, filename)

    if "autosklearn" in ALGORITHMS:
        dict_file = {'measurement_date': M_DATE,
                    'random_seed': RANDOM_STATE,
                    'pca_numbers': PCA_NUM,
                    'number_of_measurements': NUM_OF_MEASUREMENTS,
                    'bin_outlier_detect': BIN_OUTLIER,
                    'autosklearn_runtime': autosklearn_conf['params']['time_for_task'],
                    'autosklearn_limit': autosklearn_conf['params']['run_time_limit'] }
    else:
        dict_file = {'measurement_date': M_DATE,
                    'random_seed': RANDOM_STATE,
                    'pca_numbers': PCA_NUM,
                    'number_of_measurements': NUM_OF_MEASUREMENTS,
                    'bin_outlier_detect': BIN_OUTLIER }

    # create first entry once at the creation of the file
    if not Path.exists(GLOBAL_YAML_SUMMERY_FILE):
        with open(GLOBAL_YAML_SUMMERY_FILE, 'w') as file:
            documents = yaml.dump(dict_file, file)


    ######################################
    # RUN PIPELINE AND ALGORITHMS FOR ALL SELECTED MODELS
    ######################################
    # outmost loop -> configure number of repetitive runs
    for measurement in range(NUM_OF_MEASUREMENTS):

        # print number of measurements
        print('\n Measurement {} of {} with random state {}'.format(measurement+1, NUM_OF_MEASUREMENTS, measurement+1))

        # iterate through all construction machine models
        for count, machine_model in enumerate(SELECTED_MACHINE_MODELS):

            # get model configuration
            print('\n Construction machine model {} of {} - {}'.format(count+1, len(SELECTED_MACHINE_MODELS), machine_model))

            evaluate_data(machine_model = machine_model,
                                    measurement = measurement + 1,
                                    GLOBAL_TXT_SUMMERY_FILE = GLOBAL_TXT_SUMMERY_FILE,
                                    GLOBAL_YAML_SUMMERY_FILE = GLOBAL_YAML_SUMMERY_FILE,
                                    config = CFG)


    ######################################
    # CREATE WORD FILE WITH ALL RESULTS
    ######################################

    # document results as docx
    if DOCUMENTATION == True:
        document_results_docx(const_machine_models = SELECTED_MACHINE_MODELS,
                              NUM_OF_MEASUREMENTS = NUM_OF_MEASUREMENTS,
                              GLOBAL_YAML_SUMMERY_FILE = GLOBAL_YAML_SUMMERY_FILE,
                              EXPLICIT_SUMMERY_FILE_PATH = EXPLICIT_SUMMERY_FILE_PATH,
                              config = CFG)

if __name__ == '__main__':
    main()
