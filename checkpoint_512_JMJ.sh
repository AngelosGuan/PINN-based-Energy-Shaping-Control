#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_v100_32gb|gpu_a100_40gb|gpu_a100_80gb
# Bash script: checkpoint_512_JMJ.sh

# --- deterministic math + single-threaded BLAS/OMP ---
export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # or ":16:8"
export PYTHONHASHSEED="0"

python train_from_savedstate_dof4.py --model_name dof4_JMJ_512 --config_opt 3 &
wait