#!/bin/bash

# Example script of using the train_evaluate.py script which pre-trains
# the model and perform a downstream evaluation of classification using
# linear evaluation and finetuning with 1% and 10% respectively.
#
# USAGE:
# ./scripts/train_evaluate.sh <DATASET> <METHOD>
#
# EXAMPLES:
# ./scripts/train_evaluate.sh eurosat simclr # Pre-train SimCLR on the EuroSAT dataset
# ./scripts/train_evaluate.sh resisc byol    # Pre-train BYOL on the Resisc-45 dataset

set -o errexit
set -o pipefail
set -o nounset

if [[ -z "$CONDA_PREFIX" ]]; then
    conda activate geossl
fi

DATASET=$1
METHOD=$2

DATA_ROOT="./data"
CHECKPOINTS_ROOT="./checkpoints/"
mkdir -p $DATA_ROOT
mkdir -p $CHECKPOINTS_ROOT

case "$DATASET" in
    "eurosat") DATASET_PATH="$DATA_ROOT/$DATASET/" ;;
    "resisc")  DATASET_PATH="$DATA_ROOT/$DATASET/" ;;
esac

CHECKPOINT_DIR="${CHECKPOINTS_ROOT}/train-${METHOD}-${DATASET}/"

python train_evaluate.py \
    --backbone_arch resnet18 \
    --method $METHOD \
    --train_batch_size 512 \
    --train_optimizer sgd \
    --train_learning_rate_weights \
    --train_learning_rate_biases \
    --train_cosine_schedule \
    --checkpoint_dir $CHECKPOINT_DIR \
    --augmentation_specs 1111 \
    --n_epochs 800 \
    $DATASET_PATH
