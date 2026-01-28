#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_v100_32gb|gpu_a100_40gb|gpu_a100_80gb
# Bash script: main_512.sh

# --- deterministic math + single-threaded BLAS/OMP ---
export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # or ":16:8"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export VECLIB_MAXIMUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export PYTHONHASHSEED="0"

python main_dof4.py --num_epoch_adam 200 --num_epoch_bfgs 0 --model_name dof4_KMK_512 --config_opt 1 &
wait