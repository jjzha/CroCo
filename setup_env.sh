#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

ENV_PATH=".env_rocm" # edit the example.env

# Load and Export everything in .env
if [ -f $ENV_PATH ]; then
    set -a            # Automatically export all variables
    source $ENV_PATH
    set +a            # Turn off auto-export
else
    echo "Warning: .env file not found"
fi

# Ensure all necessary directories exist
mkdir -p "$TORCH_HOME" "$HF_HOME" "$AITER_JIT_DIR/build" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
         "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "Environment loaded. HF_HOME is set to $HF_HOME"