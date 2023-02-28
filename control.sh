!/bin/bash

clear

# base_env = conda info | grep -i 'base environment'

# get current date
START_DATE=`date +%F`
echo $START_DATE

# get conda base directory path
CONDA_BASE=$(conda info --base)
echo "Base env: $CONDA_BASE"
# source /Users/StuehlerH/opt/anaconda3/etc/profile.d/conda.sh
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "Starting control script"
echo "Operating system: $OSTYPE"

# print current conda env
echo "Conda env at start: $CONDA_DEFAULT_ENV"

# deactivate current conda env
conda deactivate
echo "Conda env after deactivate: $CONDA_DEFAULT_ENV"

# activate automl-autosklearn conda env
conda activate automl-autosklearn
echo "Conda env after activate automl-autosklearn: $CONDA_DEFAULT_ENV"
echo "Conda env: $CONDA_DEFAULT_ENV"

# start computation if os == darwin (macos)
if [[ $OSTYPE == 'darwin'* ]]
then
    echo 'macOS'
    if [ $CONDA_DEFAULT_ENV == "automl-autosklearn" ]
    then
        python code/ZPAI_main.py --start_date $START_DATE --algorithms classical --models merged --document_results True
    fi
fi

# start computation if os == linux 
if [[ $OSTYPE == 'linux'* ]] 
then
    echo 'linux'
    if [ $CONDA_DEFAULT_ENV == "automl-autosklearn" ]
    then
        python code/ZPAI_main.py --start_date $START_DATE --algorithms classical nn autosklearn flaml --models merged --measurements 5 --autosk_time_for_task 1800 --autosk_runtime_limit 180 --document_results False
    fi

    conda deactivate
    echo "Conda env after deactivate: $CONDA_DEFAULT_ENV"

    conda activate automl-autogluon
    echo "Conda env after activate automl-autogluon: $CONDA_DEFAULT_ENV"

    if [ $CONDA_DEFAULT_ENV == "automl-autogluon" ]
    then
        python code/ZPAI_main.py --start_date $START_DATE --algorithms autogluon --models merged --measurements 5 --document_results True
    fi

    conda deactivate
fi
