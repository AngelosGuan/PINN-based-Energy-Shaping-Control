import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss_curve(loss_epoch, plot_title, xlabel, ylabel, start_idx=10, filename=None, file_path=None):
    """
    Plot and save a loss curve.
    
    Args:
        loss_epoch (list or np.array): List of loss values per epoch.
        plot_title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        filename (str): Name of the output file (e.g., 'TrainLoss.png').
        file_path (str): Directory where the file will be saved.
    """
    os.makedirs(file_path, exist_ok=True)
    save_path = os.path.abspath(os.path.join(file_path, filename))

    num_epochs = len(loss_epoch)
    start_index = start_idx if num_epochs > start_idx else 0

    epochs = range(start_index + 1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs, loss_epoch[start_index:])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.savefig(save_path)
    plt.close()

    return 

def plot_pde_loss_and_states(loss_funcs, model, X, filename, storage_path, print_path=None):
    """
    Plot PDE residual loss and state variable traces.

    Parameters:
        loss_funcs: loss functions object with get_PDE_Loss_trajectory()
        model: the trained model
        X: input data, shape (n, m)
        filename: output plot filename 
        storage_path: directory where the plot will be saved
    """
    # Compute PDE loss per trajectory point
    L1s_alltraj = loss_funcs.get_PDE_Loss_trajectory(model, X)
    L1s_alltraj = L1s_alltraj.detach().cpu().numpy()
    X = X.cpu().numpy()
    n, m = X.shape

    # Plot setup
    plt_title = os.path.abspath(os.path.join(storage_path, filename))
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    x = np.arange(1, len(L1s_alltraj) + 1)

    # First subplot: PDE loss
    axs[0].plot(x, L1s_alltraj, color='blue')
    axs[0].set_xlabel('data index')
    axs[0].set_ylabel('PDE Loss')
    axs[0].set_title('PDE Loss vs Data Index')

    # Second subplot: state variables (arbitrary m)
    for i in range(m):
        axs[1].plot(x, X[:, i], label=f'x{i + 1}', alpha=0.6)  # add transparency
        axs[1].set_xlabel('data index')
        axs[1].set_ylabel('state variable')
        axs[1].set_title('State Variable Values vs Data Index')
        axs[1].legend()

    plt.tight_layout()
    plt.savefig(plt_title)
    plt.close()

    # Print max and avg to log file if provided
    if print_path is not None:
        max_loss = np.max(L1s_alltraj)
        avg_loss = np.mean(L1s_alltraj)
        with open(print_path, "a") as f:
            print(f"{filename}: max: {max_loss:.7f}, avg: {avg_loss:.7f}", file=f)

    # Clean up
    del X, L1s_alltraj
    torch.cuda.empty_cache()
    return 


def save_model_parameters(model, model_name, storage_path):
    """
    Save the model's state_dict to a .pth file.

    Parameters:
        model: the trained PyTorch model
        model_name: name prefix for the saved file (string, without extension)
        storage_path: directory to save the model file
    """
    filename = f"{model_name}_model.pth"
    filepath = os.path.abspath(os.path.join(storage_path, filename))
    torch.save(model.state_dict(), filepath)
    print(f"Model parameters saved to: {filepath}")
    return

def plot_comparison_surfaces(data_list, plot_title, xlabel, ylabel, zlabel, filename, storage_path, labels=None):
    """
    Plot multiple 3D surfaces over a shared state1 vs. state2 grid.
    Parameters:
        data_list: list of (values, state1, state2) tuples
                   each: values[n], state1[n], state2[n]
        plot_title: title string for the plot
        xlabel, ylabel, zlabel: axis label strings
        filename: output filename (including path)
        cmaps: optional list of colormap names (default: ['viridis', 'plasma', 'cividis', ...])
        labels: optional list of legend labels (default: ['Model 1', 'Model 2', ...])
    """
    num_models = len(data_list)
    cmaps = ['viridis', 'plasma', 'cividis', 'inferno', 'magma']
    if labels is None:
        labels = [f'Model {i+1}' for i in range(num_models)]

    assert len(labels) == num_models, f"Number of labels ({len(labels)}) must match number of data sets ({num_models})."

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i, (values, state1, state2) in enumerate(data_list):
        if len(state1.shape) > 1 and state1.shape == values.shape:
            # If already shaped like meshgrid
            surface_data = values
            X, Y = state1, state2
        else:
            # Reshape to square grid if possible
            n_points = int(np.sqrt(len(values)))
            surface_data = np.reshape(values, (n_points, n_points))
            X = np.reshape(state1, (n_points, n_points))
            Y = np.reshape(state2, (n_points, n_points))

        ax.plot_surface(X, Y, surface_data, cmap=cmaps[i % len(cmaps)], alpha=0.5, edgecolor='k', linewidth=0.2)

    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=12, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    filepath = os.path.abspath(os.path.join(storage_path, filename))
    plt.savefig(filepath)
    plt.close()
    return


def save_checkpoint(model, optimizer, epoch, X, storage_path, scheduler=None):
    """
    Save model + optimizer (+ scheduler) state_dicts to a .pth checkpoint.
    """
    filename = f"checkpoint.pth"
    filepath = os.path.abspath(os.path.join(storage_path, filename))
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_data': X
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to: {filepath}")
    return filepath

def load_checkpoint(model, optimizer, storage_path, scheduler=None, device='cpu'):
    filename = f"checkpoint.pth"
    filepath = os.path.abspath(os.path.join(storage_path, filename))
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # restore training data to desired device
    X = checkpoint['train_data'].to(device, non_blocking=True)

    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from: {filepath}, at epoch {epoch}")
    return epoch, X

def save_losses(storage_path, total_epoch, train_loss_epoch, grad_norm_epoch, L1_epoch, L2_epoch, L3_epoch, L4_epoch, L5_epoch):
    filename = f"losses.npz"
    path = os.path.abspath(os.path.join(storage_path, filename))
    np.savez_compressed(
        path,
        train_loss_epoch=np.array(train_loss_epoch),
        grad_norm_epoch=np.array(grad_norm_epoch),
        L1=np.array(L1_epoch),
        L2=np.array(L2_epoch),
        L3=np.array(L3_epoch),
        L4=np.array(L4_epoch),
        L5=np.array(L5_epoch),
        total_epoch=np.array([total_epoch], dtype=np.int32),
    )

def save_max_error_loss(storage_path, total_epoch, L7):
    filename = f"max_error_loss.npz"
    path = os.path.abspath(os.path.join(storage_path, filename))
    np.savez_compressed(
        path,
        L7=np.array(L7),
        total_epoch=np.array([total_epoch], dtype=np.int32),
    )
    
def load_metrics_npz(path):
    z = np.load(path, allow_pickle=False)
    train_loss_epoch= z["train_loss_epoch"].tolist()
    grad_norm_epoch  = z["grad_norm_epoch"].tolist()
    L1_epoch  = z["L1"].tolist()
    L2_epoch  = z["L2"].tolist()
    L3_epoch  = z["L3"].tolist()
    L4_epoch  = z["L4"].tolist()
    L5_epoch  = z["L5"].tolist()
    L6_epoch  = z["L6"].tolist()
    total_epoch = z["total_epoch"][0]
    return total_epoch, train_loss_epoch, grad_norm_epoch, L1_epoch, L2_epoch, L3_epoch, L4_epoch, L5_epoch, L6_epoch

