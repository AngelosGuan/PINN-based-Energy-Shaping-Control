import configs.config_dof4 as config
import core.plot as plot
import core.sampling as sampling
import core.training as training
import core.utils as utils
import models.dof4.KMKhb_fourier_dof4 as models
import models.dof4.loss_dof4 as loss
import models.dof4.dynamics as dynamics

import torch
import os
import argparse
import sys

import numpy as np
from core.utils import compute_gradient_norm
from configs.config_dof4 import MAX_GRAD, SAMPLE_EVERY, REPLACE_RATE, EARLY_STAGE_LEN, EARLY_REPLACE, SAMPLE_EVERY_EARLY, WARM_UP
import traceback

result_path = "results"

if __name__ == "__main__":

    # parse command line argument for gpu and cpu resourse
    parser = argparse.ArgumentParser(description="Training for 4DOF dynamic model")
    parser.add_argument(
        "--model_name", type=str, default="resMLP_dof4", help="Folder name to store the output within results folder."
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED, help="Seed used for random algorithms."
    )

    args, _ = parser.parse_known_args()

    assert all(c.isalnum() or c == '_' for c in args.model_name), \
    "Error: model_name can only contain letters, numbers, and underscores (_)."

    seed = args.seed
    config.SEED = seed

    # set random seed for reproductiveness
    utils.set_seed(seed)

    # get absolute storage path
    current_dir = os.getcwd()
    STORAGE_PATH = os.path.abspath(os.path.join(current_dir, result_path, args.model_name))
    PRINT_PATH = os.path.abspath(os.path.join(STORAGE_PATH,"out.txt"))
    #DATA_PATH = os.path.abspath(os.path.join(current_dir, data_path))

    # make sure paths exist
    RESULTS_PATH = os.path.abspath(os.path.join(current_dir, result_path))
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    if not os.path.exists(STORAGE_PATH):
        os.mkdir(STORAGE_PATH)
    with open(PRINT_PATH,"w") as f:
        print("",file=f)

    # detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create loss function
    loss_funcs = loss.customLoss()

    # create model
    model = models.MLP().to(device)

    # create optimizer
    adam = torch.optim.AdamW(model.parameters(), lr=config.lr_adam, weight_decay=config.l2_regu_adam)

    # load checkpoint
    total_epoch, X = plot.load_checkpoint(model, adam, storage_path, device=device)
    
    # train with custom settings


    # print residual loss
    plot.plot_loss_curve(L1_epoch, plot_title="Residual Loss v. Epoch", xlabel = "epoch", ylabel = "Residual Loss", start_idx=10, filename = "ResidualLoss.png", file_path = STORAGE_PATH)

    # print grad norm
    plot.plot_loss_curve(grad_norm_epoch, plot_title="Grad Norm v. Epoch", xlabel = "epoch", ylabel = "Grad Norm", start_idx=0, filename = "Grad_Norm.png", file_path = STORAGE_PATH)

    # verify on test sets
    # sobel
    test_set = sampling.sobol_sampling(n_samples=4096, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
    plot.plot_pde_loss_and_states(loss_funcs, model, test_set, filename="sobel_test.png", storage_path=STORAGE_PATH, print_path=PRINT_PATH)

    # uniform
    test_set = sampling.uniform_sampling(n_samples=config.testset_size, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
    plot.plot_pde_loss_and_states(loss_funcs, model, test_set, filename="uniform_test.png", storage_path=STORAGE_PATH, print_path=PRINT_PATH)

    # LHS
    test_set = sampling.lhs_sampling(n_samples=config.testset_size, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
    plot.plot_pde_loss_and_states(loss_funcs, model, test_set, filename="lhs_test.png", storage_path=STORAGE_PATH, print_path=PRINT_PATH)


    # save model
    plot.save_model_parameters(model, args.model_name, STORAGE_PATH)
    plot.save_checkpoint(model, adam, total_epoch, STORAGE_PATH)