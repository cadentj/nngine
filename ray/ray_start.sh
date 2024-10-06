#!/bin/bash

ray start --head \
    --port=6379 \
    --include-dashboard=true \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --dashboard-gc

export HF_TOKEN="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"
huggingface-cli login --token $HF_TOKEN

serve deploy ray_config.yml

tail -f /dev/null