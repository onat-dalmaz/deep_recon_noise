#!/bin/bash


# Define variables for parameters
R=8,
NUM_UNROLLED_STEPS=4
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1
METRIC="val_psnr_scan"
slices=($(seq 0 16))
VARIANCES_LIST=$(IFS=,; echo "${slices[*]}")
VARIANCE_CALCULATION_METHOD="J_sketch"

CONFIG_FILE="configs/mri-recon/fastmri-brain/unrolled-exps/n2r.yaml"


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
AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS ${R} \
AUG_TEST.UNDERSAMPLE.ACCELERATIONS ${R} \
TEST.CALCULATE_PIXEL_VARIANCES True \
TEST.VARIANCES_LIST $VARIANCES_LIST \
SOLVER.TRAIN_BATCH_SIZE $TRAIN_BATCH_SIZE \
SOLVER.TEST_BATCH_SIZE $TEST_BATCH_SIZE \
TEST.VARIANCE_CALCULATION_METHOD $VARIANCE_CALCULATION_METHOD



