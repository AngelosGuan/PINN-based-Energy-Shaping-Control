#!/bin/bash
# Bash script: norms_test.sh

# --- deterministic math + single-threaded BLAS/OMP ---
export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # or ":16:8"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export VECLIB_MAXIMUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export PYTHONHASHSEED="0"

python main_dof2.py --num_epoch_adam 200 --model_name default_model &
wait