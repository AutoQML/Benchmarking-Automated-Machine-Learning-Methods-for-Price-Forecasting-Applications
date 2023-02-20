import os

py_file = "ZPAI_main"
command = f"pydoc -w {py_file}"
os.system(command)

py_file = "ZPAI_evaluate_dataset"
command = f"pydoc -w {py_file}"
os.system(command)

py_file = "ZPAI_prepare_data_for_ml"
command = f"pydoc -w {py_file}"
os.system(command)

py_file = "ZPAI_evaluate_classic_ml_models"
command = f"pydoc -w {py_file}"
os.system(command)

py_file = "ZPAI_evaluate_neural_nets"
command = f"pydoc -w {py_file}"
os.system(command)

py_file = "ZPAI_evaluate_autosklearn"
command = f"pydoc -w {py_file}"
os.system(command)

py_file = "ZPAI_common_functions"
command = f"pydoc -w {py_file}"
os.system(command)