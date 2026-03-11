import core.plot as plot
import core.sampling as sampling
import core.utils as utils
import models.dof4.loss_dof4 as loss
import models.dof4.dynamics as dynamics

import torch
import os
import argparse
import sys

import numpy as np
from core.utils import compute_gradient_norm
import traceback

# for debug
import time

debug = 1

result_path = "results"

if __name__ == "__main__":
    
    # parse command line argument for gpu and cpu resourse
    parser = argparse.ArgumentParser(description="Training for 4DOF dynamic model")
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

    # get total memory
    total_mem = torch.cuda.get_device_properties(device).total_memory


    # create loss function
    loss_funcs = loss.customLoss()

    # create model
    model = models.MLP().to(device)

    # create optimizer
    adam = torch.optim.AdamW(model.parameters(), lr=config.lr_adam, weight_decay=config.l2_regu_adam)

    # load checkpoint
    total_epoch, _ = plot.load_checkpoint(model, adam, STORAGE_PATH, device=device)
    
    # train with custom settings
    ############################
    SAMPLE_EVERY = config.SAMPLE_EVERY
    REPLACE_RATE = config.REPLACE_RATE
    num_epochs_adam = 200
    fixed_trainset_size = 3000
    batch_size = config.BATCH_SIZE

    # train with custom schedule
    ############################

    # setup dataloader
    # add 10000 lhs sample

    # use purely random sample from start.
    X = sampling.sobol_sampling(n_samples=config.num_train_data, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)

    X_fixed = sampling.sobol_sampling(n_samples=fixed_trainset_size, input_dim=model.INPUT_DIM, device=device, lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
    train_set = torch.cat((X, X_fixed),dim=0)
    dataset = torch.utils.data.TensorDataset(train_set)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


    # save losses for logging
    train_loss_epoch = []
    grad_norm_epoch = []
    num_losses  = 6
    losses_epoch = [[] for _ in range(num_losses)]

    ############################################
    # build proposal sampler and RAD sampler
    sobol_proposal = sampling.make_proposal(sampling.sobol_sampling, dynamics.LOWER_BOUNDS, dynamics.UPPER_BOUNDS, device, model.INPUT_DIM)

    # replace more less frequently late
    adaptive_sampler_late = sampling.AdaptiveSamplerRAD(
        residual_fn = loss_funcs.residual_pointwise,                
        proposal_sampler = sobol_proposal,
        replace_frac=REPLACE_RATE,           
        pool_mult=8,                 
        k=1.0,                       
        c=1.0,                       
        batch_size=16384,                
        eps=1e-12
    )

    # ADAM training loop
    for ep in range(num_epochs_adam):
        try: 
            train_loss = []
            loss_lists = [[] for _ in range(num_losses)]
            minibatch_grad_norms = []



            #########################
            # adaptive sampling
            # late phase (resample every SAMPLE_EVERY epoch)
            if (ep+1) % SAMPLE_EVERY==0:

                if debug:
                    start_time = time.perf_counter()

                X = adaptive_sampler_late.step(model, X)
                # setup dataloader
                train_set = torch.cat((X, X_fixed),dim=0)
                dataset = torch.utils.data.TensorDataset(train_set)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

                if debug:
                    # Record the end time
                    end_time = time.perf_counter()
                    # Calculate and print the elapsed time
                    elapsed_time = end_time - start_time
                    print(f"At Epoch {ep:d}, Adaptive Sampling Elapsed time: {elapsed_time:.4f} seconds")

            if debug:
                start_time = time.perf_counter()

            # using minibatch
            for (batch,) in dataloader:
                batch = batch.to(device)
                train_loss_batch, losses = loss_funcs.total_loss_checkpoint(model, batch)
                
                # TODO: schedule gradient descent alteratively for different loss if needed or adjust learning rate
                # backward prop
                adam.zero_grad()
                train_loss_batch.backward()
                # log gradient norm for this batch
                grad_norm = compute_gradient_norm(model)
                minibatch_grad_norms.append(grad_norm)
                adam.step()

                train_loss.append(train_loss_batch.detach().cpu())
                for i, loss_val in enumerate(losses):
                    loss_lists[i].append(loss_val)

                # clear GPU memory every minibatch to prevent overflow

                # Optional memory guard
                allocated = torch.cuda.memory_allocated(device)
                if allocated / total_mem > 0.90:   # 90% usage threshold
                    print("near OOM GPU cache clean triggered.")
                    torch.cuda.empty_cache()

                del train_loss_batch, losses

            if debug:
                # Record the end time
                end_time = time.perf_counter()
                # Calculate and print the elapsed time
                elapsed_time = end_time - start_time
                print(f"At Epoch {ep:d}, Compute Losses for all data Elapsed time: {elapsed_time:.4f} seconds")

            # log losses of all the minibatches
            train_loss_epoch.append(np.mean(train_loss))
            for i, loss_list in enumerate(loss_lists):
                losses_epoch[i].append(np.mean(loss_list))


            if debug:
                # Record the start time
                start_time = time.perf_counter()

            # compute grad norm
            avg_grad_norm  = np.mean(minibatch_grad_norms)
            grad_norm_epoch.append(avg_grad_norm)

            if debug:
                # Record the end time
                end_time = time.perf_counter()
                # Calculate and print the elapsed time
                elapsed_time = end_time - start_time
                print(f"At Epoch {ep:d}, Compute gradient loss Elapsed time: {elapsed_time:.4f} seconds")

            # print progress
            if ep % 10 == 0 or ep == num_epochs_adam - 1:
                with open(PRINT_PATH, "a") as f:
                    print(f"epoch: {total_epoch}, train loss: {np.mean(train_loss):.7f}, "
                        f"grad norm: {avg_grad_norm:.7f}, " +", ".join([f"L{i+1}: {mean:.7f}" for i, mean in enumerate([losses_epoch[j][ep] for j in range(num_losses)])]), file=f)

            del train_loss, loss_lists
            torch.cuda.empty_cache()
            # update current epoch
            total_epoch += 1

            # check for GPU memory leak
            # with open(PRINT_PATH, "a") as f:
            #     print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB", file=f)
            #     print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB", file=f)

        except Exception as e:
            with open(PRINT_PATH, "a") as f:
                print(f"Error at epoch {ep+1}: {e}", file=f)
                print(traceback.format_exc(), file=f)
            sys.exit(1)
    [L1_epoch, L2_epoch, L3_epoch, L4_epoch, L5_epoch, L6_epoch] = losses_epoch
    X = X.detach().cpu()








    ###########################


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


    # save model
    plot.save_model_parameters(model, args.model_name, STORAGE_PATH)
    plot.save_checkpoint(model, adam, total_epoch, X, STORAGE_PATH)
    plot.save_losses(STORAGE_PATH, total_epoch, train_loss_epoch, grad_norm_epoch, L1_epoch, L2_epoch, L3_epoch, L4_epoch, L5_epoch, L6_epoch)
    #plot.save_max_error_loss(STORAGE_PATH, total_epoch, L7_epoch)