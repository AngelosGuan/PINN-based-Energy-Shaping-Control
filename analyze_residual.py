import configs.config_dof4_512_JMJ as config
import models.dof4.JwMJhb_fourier_dof4 as models


import torch
import os
import argparse
import sys

import core.plot as plot
import core.sampling as sampling
import core.training as training
import core.utils as utils
import models.dof4.loss_dof4 as loss
import models.dof4.dynamics as dynamics

import numpy as np
import matplotlib.pyplot as plt
from core.utils import compute_gradient_norm
import traceback

result_path = "results"

if __name__ == "__main__":

    # parse command line argument for gpu and cpu resourse
    parser = argparse.ArgumentParser(description="Training for 4DOF dynamic model")
    parser.add_argument(
        "--model_name", type=str, default="default", help="Folder name to store the output within results folder."
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
    


    # setup dataloader
    X_fixed = sampling.sobol_sampling(n_samples=500, input_dim=6, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)

    with torch.no_grad():
        residual = loss_funcs.get_PDE_Loss_trajectory(model, X_fixed)
        out = model.forward(X_fixed)

    # ------------------------------------------------------------
    # Make residual shape safe: convert to (N,)
    # ------------------------------------------------------------
    residual_flat = residual.detach().reshape(residual.shape[0], -1)

    if residual_flat.shape[1] > 1:
        residual_scalar = torch.norm(residual_flat, dim=1)
    else:
        residual_scalar = residual_flat.squeeze(1)

    # ------------------------------------------------------------
    # Percentile thresholds
    # ------------------------------------------------------------
    q25 = torch.quantile(residual_scalar, 0.25).item()
    q75 = torch.quantile(residual_scalar, 0.75).item()
    rmin = torch.min(residual_scalar).item()
    rmax = torch.max(residual_scalar).item()

    # 3d plot for loss distibution
    # ------------------------------------------------------------
    # 3D scatter plot
    # ------------------------------------------------------------
    X_cpu = X_fixed.detach().cpu()
    res_cpu = residual_scalar.detach().cpu()

    x_plot = X_cpu[:, 0].numpy()
    y_plot = X_cpu[:, 1].numpy()
    z_plot = X_cpu[:, 2].numpy()

    low_mask = res_cpu <= q25
    mid_mask = (res_cpu > q25) & (res_cpu < q75)
    high_mask = res_cpu >= q75

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        x_plot[high_mask.numpy()],
        y_plot[high_mask.numpy()],
        z_plot[high_mask.numpy()],
        c="red",
        s=30,
        label="Top 25% residual"
    )

    ax.scatter(
        x_plot[mid_mask.numpy()],
        y_plot[mid_mask.numpy()],
        z_plot[mid_mask.numpy()],
        c="yellow",
        s=30,
        label="25%-75% residual"
    )

    ax.scatter(
        x_plot[low_mask.numpy()],
        y_plot[low_mask.numpy()],
        z_plot[low_mask.numpy()],
        c="green",
        s=30,
        label="Bottom 25% residual"
    )

    ax.set_xlabel("X[:, 0]")
    ax.set_ylabel("X[:, 1]")
    ax.set_zlabel("X[:, 2]")
    ax.set_title("Residual Loss Distribution over First 3 Dimensions")
    ax.legend()

    plt.tight_layout()
    plt_title = os.path.abspath(os.path.join(STORAGE_PATH, "scatter_plot_residual.png"))
    plt.savefig(plt_title)
    plt.close()
    
    # print out 10 random input's output
    # Print stats and 10 random samples to file
    # ------------------------------------------------------------
    num_points = X_fixed.shape[0]
    rand_idx = torch.randperm(num_points)[:10]

    with open(PRINT_PATH, "w") as f:
        print("Residual statistics:", file=f)
        print(f"25% cutoff: {q25:.8f}", file=f)
        print(f"75% cutoff: {q75:.8f}", file=f)
        print(f"Min residual: {rmin:.8f}", file=f)
        print(f"Max residual: {rmax:.8f}", file=f)
        print("", file=f)

        print("Randomly selected 10 samples:", file=f)
        print("", file=f)

        for i, idx in enumerate(rand_idx.tolist()):
            print(f"Sample {i+1}, idx = {idx}", file=f)
            print(f"X = {X_fixed[idx].detach().cpu().numpy()}", file=f)
            print(f"Residual = {residual_scalar[idx].item():.8f}", file=f)
            print(f"Output = {out[idx].detach().cpu().numpy()}", file=f)
            print("", file=f)

