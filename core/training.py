import numpy as np
import torch
import sys
from core.utils import compute_gradient_norm, CosineAnnealingWarmupRestarts, gradients_all_zero
import core.sampling as sampling 
import models.dof4.dynamics as dynamics
import traceback


########################################################################
# adam optimizer
def initialize_adam_optimizer(model, lr, l2_regu, WARM_UP):
    adam = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_regu)
    scheduler = CosineAnnealingWarmupRestarts(
        adam,
        first_cycle_steps=WARM_UP+1,
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=lr,
        warmup_steps=WARM_UP,
        gamma=1.0
    )
    return adam, scheduler

def initialize_lbfgs_optimizer(model, lr, max_iter):
    return torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

########################################################################
# train function
def train(model, loss_funcs, calculate_weights, X, batch_size, num_epochs_adam, num_epochs_bfgs, lr_adam, l2_regu_adam, lr_lbfgs, max_iter_lbfgs, print_path, SAMPLE_EVERY, REPLACE_RATE, EARLY_STAGE_LEN, EARLY_REPLACE, SAMPLE_EVERY_EARLY, WARM_UP):

    # use GPU when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setup initial weights
    weights = calculate_weights(loss_funcs, model, X, print_path)

    # setup dataloader
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize optimizers
    # adam
    adam, scheduler = initialize_adam_optimizer(model, lr_adam, l2_regu_adam, WARM_UP)
    
    # lbfgs
    lbfgs = initialize_lbfgs_optimizer(model, lr_lbfgs, max_iter_lbfgs)
    # 2DOF
    def closure():
        # needs to define closure here for other models
        lbfgs.zero_grad()
        loss, _ = loss_funcs.total_loss(model, X)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD)
        return loss


    # save losses for logging
    train_loss_epoch = []
    grad_norm_epoch = []
    num_losses  = 6
    losses_epoch = [[] for _ in range(num_losses)]

    ############################################
    # build proposal sampler and RAD sampler
    lhs_proposal = sampling.make_proposal(sampling.lhs_sampling, dynamics.LOWER_BOUNDS, dynamics.UPPER_BOUNDS, device, model.INPUT_DIM)

    # replace less more frequently early
    adaptive_sampler_early = sampling.AdaptiveSamplerRAD(
        residual_fn = loss_funcs.residual_pointwise,                
        proposal_sampler = lhs_proposal,
        replace_frac=EARLY_REPLACE,           
        pool_mult=8,                 
        k=1.0,                       
        c=1.0,                       
        batch_size=32768,                
        eps=1e-12
    )

    # replace more less frequently late
    adaptive_sampler_late = sampling.AdaptiveSamplerRAD(
        residual_fn = loss_funcs.residual_pointwise,                
        proposal_sampler = lhs_proposal,
        replace_frac=REPLACE_RATE,           
        pool_mult=8,                 
        k=1.0,                       
        c=1.0,                       
        batch_size=32768,                
        eps=1e-12
    )



    # ADAM training loop
    for ep in range(num_epochs_adam):
        try:
            # for last two epochs in adam jiggle learn rate
            # if num_epochs_adam >= 50 and ep == num_epochs_adam - 2:
            #     for param_group in adam.param_groups:
            #         param_group['lr'] = lr_adam * 0.1  # reduce LR by factor of 10

                        # for last two epochs in adam jiggle learn rate
            # if ep < 10:
            #     for param_group in adam.param_groups:
            #         param_group['lr'] = lr_adam * 10  # increase LR by factor of 10
            # if ep == 10:
            #     for param_group in adam.param_groups:
            #         param_group['lr'] = lr_adam  # normal learn rate

            
            train_loss = []
            loss_lists = [[] for _ in range(num_losses)]
            minibatch_grad_norms = []


            #########################
            # adaptive sampling
            if ep < EARLY_STAGE_LEN:
                # early phase (first 20 epoch, resample every 2 epoch)
                if (ep+1)%SAMPLE_EVERY_EARLY == 0:
                    # resample with adaptive sampling
                    X = adaptive_sampler_early.step(model, X)
                    # setup dataloader
                    dataset = torch.utils.data.TensorDataset(X)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            else:
                # late phase (resample every SAMPLE_EVERY epoch)
                if (ep+1) % SAMPLE_EVERY==0:
                    X = adaptive_sampler_late.step(model, X)
                    # setup dataloader
                    dataset = torch.utils.data.TensorDataset(X)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        

            # using minibatch
            for (batch,) in dataloader:
                batch = batch.to(device)
                train_loss_batch, losses = loss_funcs.total_loss(model, batch)
                
                # TODO: schedule gradient descent alteratively for different loss if needed or adjust learning rate
                # backward prop
                adam.zero_grad()
                train_loss_batch.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD)
                # log gradient norm for this batch
                grad_norm = compute_gradient_norm(model)
                minibatch_grad_norms.append(grad_norm)
                # if (gradients_all_zero(model)):
                #     print("All gradients vanished, abort!!!")
                #     sys.exit(1)
                adam.step()

                train_loss.append(train_loss_batch.detach().cpu())
                for i, loss_val in enumerate(losses):
                    loss_lists[i].append(loss_val)

                # clear GPU memory every minibatch to prevent overflow
                del train_loss_batch, losses
                torch.cuda.empty_cache()

            # log losses of all the minibatches
            train_loss_epoch.append(np.mean(train_loss))
            for i, loss_list in enumerate(loss_lists):
                losses_epoch[i].append(np.mean(loss_list))

            # Update learning rate only during warm up phase
            if ep < WARM_UP:
                scheduler.step()

            # compute grad norm
            avg_grad_norm  = np.mean(minibatch_grad_norms)
            grad_norm_epoch.append(avg_grad_norm)

            # print progress
            if ep % 10 == 0 or ep == num_epochs_adam - 1:
                with open(print_path, "a") as f:
                    print(f"epoch: {ep + 1}, train loss: {np.mean(train_loss):.7f}, "
                        f"grad norm: {avg_grad_norm:.7f}, " +", ".join([f"L{i+1}: {mean:.7f}" for i, mean in enumerate([losses_epoch[j][ep] for j in range(num_losses)])]), file=f)

            del train_loss, loss_lists
            torch.cuda.empty_cache()

            # check for GPU memory leak
            # with open(PRINT_PATH, "a") as f:
            #     print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB", file=f)
            #     print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB", file=f)

        except Exception as e:
            with open(print_path, "a") as f:
                print(f"Error at epoch {ep+1}: {e}", file=f)
                print(traceback.format_exc(), file=f)
            sys.exit(1)



    # L-BFGS training loop
    for ep in range(num_epochs_bfgs):
        try:

            # step with l-BFGS optimizer
            lbfgs.step(closure)

            # compute grad norm
            total_norm = compute_gradient_norm(model)
            grad_norm_epoch.append(total_norm)
                
            # recompute losses for logging without grads
            with torch.no_grad():
                # using minibatch
                train_loss = []
                loss_lists = [[] for _ in range(num_losses)]

                for (batch,) in dataloader:
                    batch = batch.to(device)
                    train_loss_batch, losses = loss_funcs.total_loss(model, batch)

                    train_loss.append(train_loss_batch.detach().cpu())
                    for i, loss_val in enumerate(losses):
                        loss_lists[i].append(loss_val)

                    # clear GPU memory every minibatch to prevent overflow
                    del train_loss_batch, losses
                    torch.cuda.empty_cache()

                # log losses of all the minibatches
                train_loss_epoch.append(np.mean(train_loss))
                for i, loss_list in enumerate(loss_lists):
                    losses_epoch[i].append(np.mean(loss_list))

                # print progress
                if ep % 10 == 0 or ep == num_epochs_bfgs - 1:
                    with open(print_path, "a") as f:
                        total_ep = ep + num_epochs_adam + 1
                        print(f"epoch: {total_ep}, train loss: {train_loss_epoch[total_ep-1]:.7f}, "
                            f"grad norm: {total_norm:.7f}, " +", ".join([f"L{i+1}: {mean:.7f}" for i, mean in enumerate([losses_epoch[j][total_ep-1] for j in range(num_losses)])]), file=f)


            # clear memory here
            torch.cuda.empty_cache()

            # check for GPU memory leak
            # with open(PRINT_PATH, "a") as f:
            #     print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB", file=f)
            #     print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB", file=f)

        except Exception as e:
            with open(PRINT_PATH, "a") as f:
                print(f"Error at epoch {ep+num_epochs_adam+1}: {e}", file=f)
                print(traceback.format_exc(), file=f)
            sys.exit(1)
    return train_loss_epoch, grad_norm_epoch, losses_epoch, adam, X.detach().cpu()