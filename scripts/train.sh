#!/bin/bash

EXPERIMENT_NAME=""
CONFIG_PATH=""
PRETRAINED_PATH=""
SEED=379
TASK="train"
DATA_PATH=""
EPOCHS=200
LEARNING_RATE=2e-4
RESUME_CHECKPOINTS=""
SCHEDULER_FACTOR=0.1
SCHEDULER_PATIENCE=5

python -m main \
    --experiment_name "$EXPERIMENT_NAME" \
    --config_path "$CONFIG_PATH" \
    --pretrained_path "$PRETRAINED_PATH" \
    --seed $SEED \
    --task "$TASK" \
    --data_path "$DATA_PATH" \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --resume_checkpoints "$RESUME_CHECKPOINTS" \
    --scheduler_factor $SCHEDULER_FACTOR \
    --scheduler_patience $SCHEDULER_PATIENCE
