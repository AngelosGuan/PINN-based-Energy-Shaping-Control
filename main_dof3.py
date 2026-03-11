import core.plot as plot
import core.sampling as sampling
import core.training as training
import core.utils as utils
import models.dof3.loss_dof3 as loss
import models.dof3.dynamics as dynamics

import torch
import os
import argparse
import sys

########################################################################

#data_path = "data/dof8/"
result_path = "results"
########################################################################  
## main function ##
########################################################################
if __name__ == "__main__":

    # parse command line argument for gpu and cpu resourse
    parser = argparse.ArgumentParser(description="Training for 4DOF dynamic model")
    parser.add_argument(
        "--num_epoch_adam", type=int, default=100, help="Number of epochs using ADAM for training"
    )
    parser.add_argument(
        "--num_epoch_bfgs", type=int, default=0, help="Number of epochs for using L-BFGS for training"
    )
    parser.add_argument(
        "--model_name", type=str, default="MLP_dof3", help="Folder name to store the output within results folder."
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Seed used for random algorithms."
    )
    parser.add_argument(
        "--config_opt", type=int, default=0, help="config file to use, 0 for config_dof4, 1 for config_dof4_512, 2 for config_dof4_1024, ..."
    )

    args, _ = parser.parse_known_args()

    assert all(c.isalnum() or c == '_' for c in args.model_name), \
    "Error: model_name can only contain letters, numbers, and underscores (_)."

    num_epochs_adam = args.num_epoch_adam
    num_epochs_bfgs = args.num_epoch_bfgs

    # different configuration and import for different models
    config_opt = args.config_opt
    if config_opt == 1:
        import configs.config_dof3 as config
        import models.dof3.addition_fourier_dof3 as models
    else:
        # default
        import configs.config_dof3 as config
        import models.dof3.addition_fourier_dof3 as models

    if not args.seed == -1:
        config.SEED = args.seed

    # set random seed for reproductiveness
    utils.set_seed(config.SEED)

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

    # create training data from sampling
    X = sampling.sobol_sampling(n_samples=config.num_train_data, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
    
    # call train
    (train_loss_epoch, grad_norm_epoch, 
    [L1_epoch, L2_epoch, L3_epoch, L4_epoch, L5_epoch, L6_epoch], adam, X) = training.train(
        model,
        loss_funcs,
        X,
        config.BATCH_SIZE,
        num_epochs_adam,
        num_epochs_bfgs,
        config.lr_adam,
        config.l2_regu_adam,
        config.lr_lbfgs,
        config.max_iter_lbfgs,
        PRINT_PATH,
        config.SAMPLE_EVERY, 
        config.REPLACE_RATE, 
        config.EARLY_STAGE_LEN, 
        config.EARLY_REPLACE, 
        config.SAMPLE_EVERY_EARLY,
        config.WARM_UP
    )

    # print residual loss
    plot.plot_loss_curve(L1_epoch, plot_title="Residual Loss v. Epoch", xlabel = "epoch", ylabel = "Residual Loss", start_idx=10, filename = "ResidualLoss.png", file_path = STORAGE_PATH)

    # print grad norm
    plot.plot_loss_curve(grad_norm_epoch, plot_title="Grad Norm v. Epoch", xlabel = "epoch", ylabel = "Grad Norm", start_idx=0, filename = "Grad_Norm.png", file_path = STORAGE_PATH)

    # verify on test sets
    # sobel
    test_set = sampling.sobol_sampling(n_samples=4096, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS, seed=config.TEST_SEED)
    plot.plot_pde_loss_and_states(loss_funcs, model, test_set, filename="sobel_test.png", storage_path=STORAGE_PATH, print_path=PRINT_PATH)

    # uniform
    test_set = sampling.uniform_sampling(n_samples=config.testset_size, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS, seed=config.TEST_SEED)
    plot.plot_pde_loss_and_states(loss_funcs, model, test_set, filename="uniform_test.png", storage_path=STORAGE_PATH, print_path=PRINT_PATH)

    # LHS
    test_set = sampling.lhs_sampling(n_samples=config.testset_size, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS, seed=config.TEST_SEED)
    plot.plot_pde_loss_and_states(loss_funcs, model, test_set, filename="lhs_test.png", storage_path=STORAGE_PATH, print_path=PRINT_PATH)

    # matlab data
    # filepath = os.path.abspath(os.path.join(DATA_PATH, "x_8dof.mat"))
    # test_set = utils.load_data(filepath).to(device)
    # plot.plot_pde_loss_and_states(loss_funcs, model, test_set, filename="traj10_test.png", storage_path=STORAGE_PATH, print_path=PRINT_PATH)

    # save model
    plot.save_model_parameters(model, args.model_name, STORAGE_PATH)
    plot.save_checkpoint(model, adam, num_epochs_adam, X, STORAGE_PATH)
    plot.save_losses(STORAGE_PATH, num_epochs_adam, train_loss_epoch, grad_norm_epoch, L1_epoch, L2_epoch, L3_epoch, L4_epoch, L5_epoch, L6_epoch )

