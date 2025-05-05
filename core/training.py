import numpy as np
import torch
import sys
from core.utils import compute_gradient_norm, CosineAnnealingWarmupRestarts

########################################################################
# adam optimizer
def initialize_adam_optimizer(model, lr, l2_regu):
    adam = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_regu)
    scheduler = CosineAnnealingWarmupRestarts(
        adam,
        first_cycle_steps=1000,
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=1e-8,
        warmup_steps=20,
        gamma=1.0
    )
    return adam, scheduler

def initialize_lbfgs_optimizer(model, lr, max_iter):
    return torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

########################################################################
# compute weights
def calculate_weights(loss_funcs, model, X, print_path=None):
    with torch.no_grad():
        residual_loss, control_loss = loss_funcs.get_PDE_and_control_loss(model, X).detach().cpu()
        boundary_loss = loss_funcs.get_Boundary_Loss(model).detach().cpu()
        deviation_loss = loss_funcs.get_deviation_loss(model, X, 0.5).detach().cpu()
        eig_loss = loss_funcs.eig_range_loss(model, X, 0.5).detach().cpu()
        sparse_loss = loss_funcs.sparse_sample_loss(model, 100).detach().cpu()
        pos_def_loss = loss_funcs.pos_eig_loss(model, X).detach().cpu()

        eps = 1e-10
        n = float(X.shape[0])
        weights = np.array([
            1.0 / (1.0 + np.log(1.0 + residual_loss + eps)),
            1.0 / (1.0 + np.log(1.0 + control_loss + eps)),
            1.0 / (1.0 + np.log(1.0 + boundary_loss + eps)),
            1.0 / (1.0 + np.log(1.0 + deviation_loss + eps)),
            1.0 / (1.0 + np.log(1.0 + eig_loss + eps)),
            1.0 / (1.0 + np.log(1.0 + sparse_loss + eps)),
            1.0 / (1.0 + np.log(1.0 + pos_def_loss + eps))
        ])
        clamped_weights = np.clip(weights, a_min=0.5, a_max=5.0)
        # scale boundary loss by number of samples if present
        clamped_weights[2] *= n

        if getattr(model, 'hard_boundary', False):
            # boundary loss not needed for hard boundary models
            clamped_weights[2] = 0.0
        if getattr(model, 'pos_def', False):
            # positive eigenvalue loss not needed for positive definite models
            clamped_weights[6] = 0.0

        # Optional debug printing
        if print_path is not None:
            with open(print_path, "a") as f:
                print(f"W1: {clamped_weights[0]:.6f}, W2: {clamped_weights[1]:.6f}, W3: {clamped_weights[2]:.6f}, "
                      f"W4: {clamped_weights[3]:.6f}, W5: {clamped_weights[4]:.6f}, W6: {clamped_weights[5]:.6f}, "
                      f"W7: {clamped_weights[6]:.6f}", file=f)
                print(f"L1: {residual_loss:.6f}, L2: {control_loss:.6f}, L3: {boundary_loss:.6f}, "
                      f"L4: {deviation_loss:.6f}, L5: {eig_loss:.6f}, L6: {sparse_loss:.6f}, "
                      f"L7: {pos_def_loss:.6f}", file=f)
        return clamped_weights

########################################################################
# train function
def train(model, loss_funcs, X, batch_size, num_epochs_adam, num_epoch_bfgs, lr_adam, l2_regu_adam, lr_lbfgs, max_iter_lbfgs, print_path):

    # use GPU when available
    adam, scheduler = initialize_adam_optimizer(model, lr_adam, l2_regu_adam)
    
    # setup initial weights
    weights = calculate_weights(loss_funcs, model, X, print_path)

    # Initialize optimizers
    # adam
    adam, scheduler = initialize_adam_optimizer(model, lr_lbfgs, max_iter_lbfgs)
    
    # lbfgs
    lbfgs = initialize_lbfgs_optimizer(model)
    def closure():
        lbfgs.zero_grad()
        loss, _ = loss_funcs.total_loss(model, X, weights)
        loss.backward()
        return loss

    # save losses for logging
    train_loss_epoch = []
    grad_norm_epoch = []
    L1_epoch, L2_epoch, L3_epoch, L4_epoch, L5_epoch, L6_epoch, L7_epoch = [], [], [], [], [], [], []


    # ADAM training loop
    for ep in range(num_epochs_adam):
        try:
            # for last two epochs in adam jiggle learn rate
            if ep == num_epochs_adam - 2:
                for param_group in adam.param_groups:
                    param_group['lr'] = LR * 0.1  # reduce LR by factor of 10

                
            num_samples = X.shape[0]
            train_loss = []
            L1, L2, L3, L4, L5, L6, L7 = [], [], [], [], [], [], []

            # using minibatch
            for i in range(0, num_samples, BATCH_SIZE):
                batch = X[i:i+BATCH_SIZE]
                train_loss_batch, losses = loss_funcs.total_loss(model, batch, weights)
                
                # TODO: schedule gradient descent alteratively for different loss if needed or adjust learning rate
                # backward prop
                adam.zero_grad()
                train_loss_batch.backward()
                adam.step()

                train_loss.append(train_loss_batch.detach().cpu())
                L1.append(losses[0])
                L2.append(losses[1])
                L3.append(losses[2])
                L4.append(losses[3])
                L5.append(losses[4])
                L6.append(losses[5])
                L7.append(losses[6])

                # clear GPU memory every minibatch to prevent overflow
                del train_loss_batch, losses
                torch.cuda.empty_cache()

            # log losses of all the minibatches
            train_loss_epoch.append(np.mean(train_loss))
            L1_epoch.append(np.mean(L1))
            L2_epoch.append(np.mean(L2))
            L3_epoch.append(np.mean(L3))
            L4_epoch.append(np.mean(L4))
            L5_epoch.append(np.mean(L5))
            L6_epoch.append(np.mean(L6))
            L7_epoch.append(np.mean(L7))


            # Update learning rate
            scheduler.step()

            # compute grad norm
            total_norm = compute_gradient_norm(model)
            grad_norm_epoch.append(total_norm)

            # print progress
            if ep % 10 == 0 or ep == num_epochs_adam - 1:
                with open(print_path, "a") as f:
                    print(f"epoch: {ep + 1}, train loss: {np.mean(train_loss):.7f}, "
                          f"L1: {np.mean(L1):.7f}, L2: {np.mean(L2):.7f}, L3: {np.mean(L3):.7f}, "
                          f"L4: {np.mean(L4):.7f}, L5: {np.mean(L5):.7f}, L6: {np.mean(L6):.7f}, L7: {np.mean(L7):.7f}, "
                          f"grad norm: {total_norm:.7f}", file=f)

            del train_loss, L1, L2, L3, L4, L5, L6, L7
            torch.cuda.empty_cache()

            # check for GPU memory leak
            # with open(PRINT_PATH, "a") as f:
            #     print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB", file=f)
            #     print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB", file=f)

        except Exception as e:
            with open(print_path, "a") as f:
                print(f"Error at epoch {ep+1}: {e}", file=f)
            sys.exit(1)

    # L-BFGS training loop
    for ep in range(num_epoch_BFGS):
        try:

            # step with l-BFGS optimizer
            lbfgs.step(closure)

            # compute grad norm
            total_norm = compute_gradient_norm(model)
            grad_norm_epoch.append(total_norm)
                
            # recompute losses for logging without grads
            with torch.no_grad():
                train_loss, losses = loss_funcs.total_loss(model, X, weights)

                # log losses 
                train_loss_epoch.append(train_loss.detach().cpu())
                L1_epoch.append(losses[0])
                L2_epoch.append(losses[1])
                L3_epoch.append(losses[2])
                L4_epoch.append(losses[3])
                L5_epoch.append(losses[4])
                L6_epoch.append(losses[5])
                L7_epoch.append(losses[6])

                # print progress
                if ep % 10 == 0 or ep == num_epochs_bfgs - 1:
                    with open(print_path, "a") as f:
                        total_ep = ep + num_epochs_adam + 1
                        print(f"epoch: {total_ep}, train loss: {train_loss.detach().cpu():.7f}, "
                              f"L1: {losses[0]:.7f}, L2: {losses[1]:.7f}, L3: {losses[2]:.7f}, "
                              f"L4: {losses[3]:.7f}, L5: {losses[4]:.7f}, L6: {losses[5]:.7f}, L7: {losses[6]:.7f}, "
                              f"grad norm: {total_norm:.7f}", file=f)

            # clear memory here
            torch.cuda.empty_cache()

            # check for GPU memory leak
            # with open(PRINT_PATH, "a") as f:
            #     print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB", file=f)
            #     print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB", file=f)

        except Exception as e:
            with open(PRINT_PATH, "a") as f:
                print(f"Error at epoch {ep+num_epochs_adam+1}: {e}", file=f)
            sys.exit(1)

    losses_epoch = [L1_epoch, L2_epoch, L3_epoch, L4_epoch, L5_epoch, L6_epoch, L7_epoch]
    return train_loss_epoch, grad_norm_epoch, losses_epoch