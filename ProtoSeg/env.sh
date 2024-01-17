# this scripts sets environment variables and activates conda environment
# run this script before any python script in this repository

# edit these variables if needed
export SOURCE_DATA_PATH='./data'
export DATA_PATH='./data'
export LOG_DIR='./logs'

export RESULTS_DIR='./results'


# set it if using neptune
export USE_NEPTUNE=0  # set to 1 to use neptune
export NEPTUNE_API_TOKEN=""  # set your neptune api token
export NEPTUNE_PROJECT=""  # set your neptune project name

# please create this conda environment using 'env.yml'
conda activate proto-seg
