#!/bin/bash
# Bash script: norms_test.sh

python main_dof4.py --num_epoch_adam 1000 --num_epoch_bfgs 100 --model_name res_dof4_residualonly &
wait