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
    total_epoch, X = plot.load_checkpoint(model, adam, STORAGE_PATH, device=device)
    
    # train with custom settings
    ############################
    # setup custom weights 
    weights = [1.0, 0.5, 1.0/10000, 1.0/100, 1.0/100, 1.0/470]
    resonly_weights = [1.0, 0.0, 0.0, 0.0, 0.0]
    SAMPLE_EVERY = config.SAMPLE_EVERY
    REPLACE_RATE = config.REPLACE_RATE
    num_epochs_adam = 200
    fixed_trainset_size = 3000
    batch_size = config.BATCH_SIZE

    # train with custom schedule
    ############################

    # setup dataloader
    # add 10000 lhs sample
    X_fixed = sampling.lhs_sampling(n_samples=fixed_trainset_size, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
    train_set = torch.cat((X, X_fixed),dim=0)
    dataset = torch.utils.data.TensorDataset(train_set)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # gather a fixed max error set
    # largest 500 values
    num_top = 500 
    residual_over_X = loss_funcs.get_PDE_Loss_trajectory(model, X_fixed)
    # flatten residual in case it's (N,1)
    residual_flat = residual_over_X.view(-1)
    # get indices of largest residuals
    topk_vals, topk_idx = torch.topk(residual_flat, k=num_top)
    # select corresponding X points
    print(topk_vals)

    # 2. number of points with loss > 0.1
    over_01 = (residual_flat > 0.1).sum().item()

    # 3. number of points with loss > 0.5
    over_05 = (residual_flat > 0.5).sum().item()

    # 4. smallest loss value
    min_loss = residual_flat.min().item()
    print(f"Number of samples with loss > 0.1: {over_01}")
    print(f"Number of samples with loss > 0.5: {over_05}")
    print(f"Smallest residual loss: {min_loss:.6f}")


    # largest 500 values
    num_top = 500 
    residual_over_adapt = loss_funcs.get_PDE_Loss_trajectory(model, X)
    # flatten residual in case it's (N,1)
    residual_flat_adapt = residual_over_adapt.view(-1)
    # get indices of largest residuals
    topk_vals, topk_idx = torch.topk(residual_flat_adapt, k=num_top)
    # select corresponding X points
    print(topk_vals)

    # 2. number of points with loss > 0.1
    over_01 = (residual_flat_adapt > 0.1).sum().item()

    # 3. number of points with loss > 0.5
    over_05 = (residual_flat_adapt > 0.5).sum().item()

    # 4. smallest loss value
    min_loss = residual_flat_adapt.min().item()
    print(f"Number of samples with loss > 0.1: {over_01}")
    print(f"Number of samples with loss > 0.5: {over_05}")
    print(f"Smallest residual loss: {min_loss:.6f}")



    ###########################


    # print residual loss
    plot.plot_loss_curve(residual_flat.detach().cpu(), plot_title="Residual Loss over fixed set", xlabel = "data point", ylabel = "Residual Loss", start_idx=0, filename = "fixed.png", file_path = STORAGE_PATH)
    plot.plot_loss_curve(residual_flat_adapt.detach().cpu(), plot_title="Residual Loss over adaptive set", xlabel = "data point", ylabel = "Residual Loss", start_idx=0, filename = "adaptive.png", file_path = STORAGE_PATH)
