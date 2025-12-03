#!/bin/bash

# Define variables for parameters
R=8
NUM_UNROLLED_STEPS=4
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1
CONFIG_FILE="configs/mri-recon/mridata-3dfse-knee/n2r-unrolled-full.yaml"
METRIC="val_psnr_scan"
VARIANCE_CALCULATION_METHOD="J_sketch"
# Create a list of slices of variances to calculate between 100 and 140
slices=($(seq 100 140))
VARIANCES_LIST=$(IFS=,; echo "${slices[*]}")

# Define MEDDLR environment variables
MEDDLR_CACHE_DIR="/cache/meddlr"
MEDDLR_DATASETS_DIR="/data"
MEDDLR_RESULTS_DIR="/results_noise"

# Export the MEDDLR environment variables
export MEDDLR_CACHE_DIR
export MEDDLR_DATASETS_DIR
export MEDDLR_RESULTS_DIR

# Run the evaluation script with the specified variables
python tools/eval_net.py \
--config-file $CONFIG_FILE \
--metric $METRIC \
MODEL.UNROLLED.NUM_UNROLLED_STEPS $NUM_UNROLLED_STEPS \
AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS ${R}, \
AUG_TEST.UNDERSAMPLE.ACCELERATIONS ${R}, \
TEST.CALCULATE_PIXEL_VARIANCES True \
TEST.VARIANCES_LIST $VARIANCES_LIST \
SOLVER.TRAIN_BATCH_SIZE $TRAIN_BATCH_SIZE \
SOLVER.TEST_BATCH_SIZE $TEST_BATCH_SIZE \
TEST.VARIANCE_CALCULATION_METHOD $VARIANCE_CALCULATION_METHOD \


