#!/bin/sh

DATASET_PATH=../DATASET_Synapse_PFormer/DATASET_Synapse

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_synapse
export pformer_preprocessed="$DATASET_PATH"/pformer_raw/pformer_raw_data/Task02_Synapse
export pformer_raw_data_base="$DATASET_PATH"/pformer_raw

python ../pformer/run/run_training.py 3d_fullres pformer_trainer_synapse 2 0
