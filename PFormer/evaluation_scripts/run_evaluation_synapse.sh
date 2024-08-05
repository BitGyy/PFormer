#!/bin/sh

DATASET_PATH=../../DATASET_Synapse
CHECKPOINT_PATH=../output_synapse

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export unetr_pp_preprocessed="$DATASET_PATH"/pformer_raw/pformer_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/pformer_raw

python ../pformer/run/run_training.py 3d_fullres pformer_trainer_synapse 2 0 -val
