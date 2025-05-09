#!/bin/bash
# Bash script: norms_test.sh

python main_dof8.py --num_epoch_adam 100 --num_epoch_bfgs 0 --model_name KMKhb_dof8 &
wait