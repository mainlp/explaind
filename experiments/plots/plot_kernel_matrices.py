import torch
import os

import plotly.express as px


device = "cuda" if torch.cuda.is_available() else "cpu"

experiment_path = "results/modulo_val_results/"

train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

# subsample both sets
N = 500
train_samples = torch.randint(0, len(train_loader.dataset), (N,))
val_samples = torch.randint(0, len(val_loader.dataset), (N,))
# sort samples
train_samples, _ = train_samples.sort()
val_samples, _ = val_samples.sort()


# plot kernel matrices over steps
kernel_matrices = torch.load(experiment_path + "kernel_matrices.pt", map_location=device)
kernel_plot_path = experiment_path + "kernel_matrices_plots/"
os.makedirs(kernel_plot_path, exist_ok=True)
y_train = [int(train_loader.dataset[i][1].item()) for i in range(len(train_loader.dataset))]
y_val = [int(val_loader.dataset[i][1].item()) for i in range(len(val_loader.dataset))]

y_train_sorted, y_train_sorted_ids = torch.tensor(y_train).sort()
y_val_sorted, y_val_sorted_ids = torch.tensor(y_val).sort()

X_train = [train_loader.dataset[i][0] for i in range(len(train_loader.dataset))]
X_val = [val_loader.dataset[i][0] for i in range(len(val_loader.dataset))]

kernel_matrices['input_emb.weight'].shape


overall_kernel = None
for layer in kernel_matrices.keys():
    km = kernel_matrices[layer]  # matrix of shape (num_val, 1, out_dim, num_train)
    # reorder s.t. same mod results are together
    positive_matrices = []
    print(km.shape)
    for i in range(km.shape[0]):
        # set values of the not correct targets to zero
        positive_matrices.append(km[i, :, y_val[i], :])
    km = torch.stack(positive_matrices, dim=0)
    print(km.shape)

    km_sorted_mod_result = km[:, :, y_train_sorted_ids][y_val_sorted_ids]

    if overall_kernel is None:
        overall_kernel = km_sorted_mod_result
    else:
        overall_kernel += km_sorted_mod_result

    # get subsample
    # km_sorted_mod_sub = km_sorted_mod_result[:, :, :, train_samples][:, :, :, val_samples]

    print(km_sorted_mod_result.shape)

    # sum over all classes
    km_sum = km_sorted_mod_result.sum(dim=1).cpu().numpy()

    # plot matrix
    fig = px.imshow(km_sum, color_continuous_scale='Viridis', aspect='auto')
    fig.update_layout(
        title=f"Kernel matrix for layer {layer} (log of sum over outputs)",
        xaxis_title="Train samples",
        yaxis_title="Val samples",
    )
    # put y labels on axes
    val_ticks = [(i, y_val_sorted[i]) for i in range(0, len(y_val), 200)]
    train_ticks = [(i, y_train_sorted[i]) for i in range(0, len(y_train), 200)]
    fig.update_xaxes(tickvals=[i[0] for i in train_ticks], ticktext=[i[1] for i in train_ticks])
    fig.update_yaxes(tickvals=[i[0] for i in val_ticks], ticktext=[i[1] for i in val_ticks])
    # fig.write_html(kernel_plot_path + f"kernel_matrix_{layer}.html")

    fig.write_image(kernel_plot_path + f"kernel_matrix_{layer}.png")
   


# plot overall kernel matrix
overall_kernel_sum = overall_kernel.sum(dim=1).cpu().numpy()
fig = px.imshow(overall_kernel_sum, color_continuous_scale='Viridis', aspect='auto')
fig.update_layout(
    title="Overall kernel matrix (log of abs sum over outputs)",
    xaxis_title="Train samples",
    yaxis_title="Val samples",
)
# put y labels on axes
val_ticks = [(i, y_val_sorted[i]) for i in range(0, len(y_val), 200)]
train_ticks = [(i, y_train_sorted[i]) for i in range(0, len(y_train), 200)]
fig.update_xaxes(tickvals=[i[0] for i in train_ticks], ticktext=[i[1] for i in train_ticks])
fig.update_yaxes(tickvals=[i[0] for i in val_ticks], ticktext=[i[1] for i in val_ticks])
fig.write_image(kernel_plot_path + "overall_kernel_matrix.png")

