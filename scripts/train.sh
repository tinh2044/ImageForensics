#!/bin/bash
CONFIG_PATH="configs/sample.yaml"
SEED=379
TASK="train"             # train | test |
RESUME=0               
CHECKPOINT_PATH=""    

RESUME_FLAG=""
if [ "$RESUME" -eq 1 ]; then
    RESUME_FLAG="--resume"
fi

python -m main \
    --config_path "$CONFIG_PATH" \
    --seed $SEED \
    --task "$TASK" \
    $RESUME_FLAG \
    --checkpoint_path "$CHECKPOINT_PATH"
