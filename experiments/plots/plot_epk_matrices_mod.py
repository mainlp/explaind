import torch
from explaind.modulo_model.model import SingleLayerTransformerClassifier

import json
from tqdm import tqdm
import plotly.graph_objects as go
import os

device = "cpu"# "cuda" if torch.cuda.is_available() else "cpu"

experiment_path = "results/modulo_val_result/"

train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

train_dataset = train_loader.dataset.data
train_sums = train_dataset[:, 0] + train_dataset[:, 2]
val_dataset = val_loader.dataset.data
val_sums = val_dataset[:, 0] + val_dataset[:, 2]

# only did first 400 samples of val
# val_dataset = val_dataset[:400]
# val_sums = val_sums[:400]

# sort both datasets by input sum
train_sums_sorted, train_sort_ids = torch.sort(train_sums, dim=0, descending=False)
train_sorted = train_dataset[train_sort_ids]

val_sums_sorted , val_sort_ids = torch.sort(val_sums, dim=0, descending=False)
val_sorted = val_dataset[val_sort_ids]


config = json.load(open(experiment_path + "config.json", 'r'))

# load model
base_model = SingleLayerTransformerClassifier(config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])

# load model checkpoints (to be found in model_path.checkpoints which is a list of state_dicts)
# model_path = ModelPath(config["model_path"], config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])

# dict of layer_name -> torch tensors of shape (num_val, 1, out_dim, num_train) = (2000, 1, 115, 4000)
# i.e. per module/layer the accumulated kernel values over all params and steps
# kernel_matrices = torch.load(experiment_path + "kernel_matrices.pt", map_location=device)

# print(kernel_matrices.keys())
# kernel_matrices['input_emb.weight'].shape


def get_comb_matrix(kernel_matrices, layer_names):
    # get the kernel matrix for the given layer names
    # kernel_matrices is a dict of layer_name -> torch tensors of shape (num_val, 1, out_dim, num_train) = (2000, 1, 115, 4000)
    # i.e. per module/layer the accumulated kernel values over all params and steps
    mat = None
    for layer_name in layer_names:
        if mat is None:
            mat = kernel_matrices[layer_name]
        else:
            mat += kernel_matrices[layer_name]

    if len(mat.shape) == 4:
        mat = mat.sum(dim=1).sum(dim=1)
    elif len(mat.shape) == 3:
        mat = mat.sum(dim=1)

    return mat


def plot_kernel_matrix(kernel_matrices, val_sums_sorted, train_sums_sorted, plot_path, log_scale=True, i=0):
    # plot kernel matrices

    combinations = {
        "embedding": ("input_emb.weight",),
        "att. encoders": ("head1_encoder.weight", "head2_encoder.weight", "head3_encoder.weight", "head4_encoder.weight"),
        "att. decoders": ("head1_decoder.weight", "head2_decoder.weight", "head3_decoder.weight", "head4_decoder.weight"),
        "linear1": ("linear1.weight", "linear1.bias"),
        "linear2": ("linear2.weight", "linear2.bias"),
        "decoder": ("decoder.weight",),
        "model": ('input_emb.weight', 
                  'head1_encoder.weight', 'head2_encoder.weight', 'head3_encoder.weight', 'head4_encoder.weight', 
                  'head1_decoder.weight', 'head2_decoder.weight', 'head3_decoder.weight', 'head4_decoder.weight', 
                  'linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias', 'decoder.weight')
    }

    for k, v in tqdm(combinations.items()):
        
        mat = get_comb_matrix(kernel_matrices, v)

        print(mat.shape)
        os.makedirs(plot_path, exist_ok=True)
        # sort axes of matrix according to the input sum
        mat = mat[val_sort_ids][:, train_sort_ids]

        # log of vals
        if log_scale:
            # avoid log(0)
            lg = torch.log(mat.abs() + 1e-10)
            mat = lg * torch.sign(mat)

        # center color bar at zero
        mmin = mat.min()
        mmax = mat.max()
        highest = max(abs(mmin), abs(mmax)).item()

        highest = 20 # highest# - (highest * 0.1)  # make the color bar a bit smaller
        highest = int(highest) + 1  # make it an integer

        # plot the kernel matrix
        # fig = px.imshow(mat, color_continuous_scale='RdBu', aspect='auto')
        fig = go.Figure(data=go.Heatmap(
            z=mat,
            colorscale='RdBu',
            colorbar=dict(title="Kernel<br>value<br>(log)"),
            zmin=-highest,
            zmax=highest,
            # coloraxis_colorbar=dict(title="Kernel value"),
        ))
        fig.update_layout(title=f"Kernel matrix for {k}", yaxis_title="Val. samples", xaxis_title="Train samples")

        # put 
        fig.update_yaxes(tickvals=list(range(0, len(val_dataset), 100)), ticktext=[f"{val_sums_sorted[i]} ({val_sums_sorted[i] % 113})" for i in range(0, len(val_dataset), 100)])
        fig.update_xaxes(tickvals=list(range(0, 4000, 100)), ticktext=[f"{train_sums_sorted[i]} ({train_sums_sorted[i] % 113})" for i in range(0, 4000, 100)])

        # rotate x axis labels
        fig.update_xaxes(tickangle=-75)

        # larger font for axes and title
        fig.update_layout(font=dict(size=20), title_font=dict(size=30))
        # font of ticks smaller
        fig.update_xaxes(tickfont=dict(size=12))
        fig.update_yaxes(tickfont=dict(size=12))

        # reverse order of y axis
        fig.update_yaxes(autorange="reversed")
    
        # make background white
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')

        fig.write_image(plot_path + f"{k}_low_res_{i}.png", width=800, height=600)
        fig.write_image(plot_path + f"{k}_{i}.png", width=800, height=600, scale=2)
        fig.write_image(plot_path + f"{k}_{i}.pdf", width=800, height=600, scale=2)


kernel_step_matrices = []
for i in range(0, 1950, 50):
    kernel_step_matrices.append(torch.load(experiment_path + f"kernel_matrices/kernel_step_matrix_{i}.pt", map_location=device))

for i in [200, 250, 500, 1100, 1900]:
    kernel_step_matrix = torch.load(experiment_path + f"kernel_matrices/kernel_step_matrix_{i}.pt", map_location=device)
    last = torch.load(experiment_path + f"kernel_matrices/kernel_step_matrix_{i - 50}.pt", map_location=device)

    print(kernel_step_matrix.keys())

    kernel_step_matrix['input_emb.weight'].shape
    
    kernel_step_matrix = {k: kernel_step_matrix[k] - last[k] for k in kernel_step_matrix.keys()}

    # plot the kernel matrix
    plot_kernel_matrix(kernel_step_matrix, val_sums_sorted, train_sums_sorted, plot_path=experiment_path + f"plots/kernel_matrices/log_kernel_step_matrix_{i*50}/",
                       log_scale=True, i=i)
    
