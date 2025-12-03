#!/bin/sh

# Define variables for parameters
R=4
NUM_UNROLLED_STEPS=4
CONFIG_FILE="configs/mri-recon/fastmri-brain/unrolled-exps/supervised.yaml"
MAX_ATTEMPTS=8

# Define MEDDLR environment variables
MEDDLR_CACHE_DIR="/cache/meddlr"
MEDDLR_DATASETS_DIR="/data"
MEDDLR_RESULTS_DIR="/results_noise"

# Export the MEDDLR environment variables
export MEDDLR_CACHE_DIR
export MEDDLR_DATASETS_DIR
export MEDDLR_RESULTS_DIR

# Run the training script with the specified variables
python tools/train_net.py \
--config-file $CONFIG_FILE \
AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS ${R}, \
MODEL.UNROLLED.NUM_UNROLLED_STEPS $NUM_UNROLLED_STEPS \
AUG_TRAIN.UNDERSAMPLE.MAX_ATTEMPTS $MAX_ATTEMPTS \
