#!/bin/sh

DATASET_PATH=../DATASET_Acdc_PFormer/DATASET_Acdc

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_acdc
export pformer_preprocessed="$DATASET_PATH"/pformer_raw/pformer_raw_data/Task01_ACDC
export pformer_raw_data_base="$DATASET_PATH"/pformer_raw

python ../pformer/run/run_training.py 3d_fullres pformer_trainer_acdc 1 0
