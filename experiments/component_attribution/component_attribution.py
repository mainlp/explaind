import torch
from path_kernels.kernels.model_paths import ModelPath
import json
import plotly.graph_objects as go
from copy import deepcopy
import numpy as np
from path_kernels.cifar.data import get_dataloader
from path_kernels.modulo_model.model import SingleLayerTransformerClassifier
from path_kernels.modulo_model.loss import RegularizedCrossEntropyLoss
from path_kernels.kernels.epk_model import ExactPathKernelModel
from path_kernels.kernels.optimizer_paths import AdamWOptimizerPath
from path_kernels.kernels.data_paths import DataPath
import os
import pandas as pd
from tqdm import tqdm
import umap


import plotly.express as px


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_kernels_and_data():
    experiment_path = "results/modulo_val_results_0.01_w/"
    model_checkpoint_path = experiment_path + "model_checkpoint.pt"
    optimizer_checkpoint_path = experiment_path + "optimizer_checkpoint.pt"
    data_checkpoint_path = experiment_path + "data_checkpoint.pt"

    # load datasets
    train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
    val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

    data_path = DataPath(train_loader,
                            checkpoint_path=data_checkpoint_path,
                            full_batch=False,
                            overwrite=False)

    config = {}
    with open(experiment_path + "config.json", "r") as f:
        config = json.load(f)

    # load model and optimizer
    model = SingleLayerTransformerClassifier(config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])
    model = ModelPath(model=model,
                    device=device, 
                    checkpoint_path=model_checkpoint_path,
                    features_to_cpu=False,
                    parallelize_over_targets=True)

    param_kernel = torch.load(experiment_path + "param_kernel.pt", weights_only=False)
    kernel_matrices = torch.load(experiment_path + "kernel_matrices.pt", weights_only=False)

    return model, data_path, param_kernel, kernel_matrices, train_loader, val_loader

def get_embedded_samples(scores, dataset, labels="accumulate"):

    targets = dataset.targets

    if labels == "positive":
        # only use the embedding of the correct class
        scores_acc = scores[torch.arange(scores.shape[0]), targets]
        print(f"positive: {scores_acc.shape}")
    elif labels == "accumulate":
        # sum up the embedding of all classes
        scores_acc = scores.sum(dim=1).sum(dim=1)
    else:
        scores_acc = scores
        
    return dataset.data, dataset.targets, scores_acc


def get_high_and_low_sim_samples(
      data, targets, input_emb, num_samples=50, num_highest=10, num_lowest=10, result_path=None, verbose=True
):
    # compute similarity matrix of all samples
    cosine_similarity_matrix = torch.einsum("ij,kj->ik", input_emb, input_emb) / (input_emb.norm(dim=1, keepdim=True) * input_emb.norm(dim=1, keepdim=True).T)

    sorted_entries, sorted_ids = torch.sort(cosine_similarity_matrix, dim=1, descending=True)

    sorted_entries = sorted_entries.cpu().numpy()
    sorted_ids = sorted_ids.cpu().numpy()

    pstr = ""

    # print highest and lowest similarity of samples
    for i in range(num_samples):
        pstr += f"Sample {i}: {data[i]} labeled {targets[i]} has high similarity with\n"
        for j in range(num_highest):
            pstr += f" - Sample {sorted_ids[i][j]}: {data[sorted_ids[i][j]]} labeled {targets[sorted_ids[i][j]]} with similarity {sorted_entries[i][j]}"
            pstr += "\n "
        pstr += "\n "

        pstr += f"and has low similarity with\n"
        for j in range(num_lowest):
            pstr += f" - Sample {sorted_ids[i][-1-j]}: {data[sorted_ids[i][-1-j]]} labeled {targets[sorted_ids[i][-1-j]]} with similarity {sorted_entries[i][-1-j]}"
            pstr += "\n "
        pstr += "\n "

        # print a label and input histogram
        data_high = data[sorted_ids[i][0:num_highest]]
        data_low = data[sorted_ids[i][-1-num_lowest:-1]]

        input_toks_high = data_high[:, 0].tolist() + data_high[:, 2].tolist()
        input_toks_low = data_low[:, 0].tolist() + data_low[:, 2].tolist()

        toks_counts = {}
        for i, tok in enumerate(input_toks_high):
            if tok not in toks_counts:
                toks_counts[tok] = 0
            toks_counts[tok] += 1
        toks_counts = sorted(toks_counts.items(), key=lambda x: x[1], reverse=True)
        pstr += f"Input histogram (high) for sample {i}:"
        for tok, count in toks_counts:
            pstr += f"{tok}({count}), "

        pstr += "\n "

        toks_counts = {}
        for i, tok in enumerate(input_toks_low):
            if tok not in toks_counts:
                toks_counts[tok] = 0
            toks_counts[tok] += 1
        toks_counts = sorted(toks_counts.items(), key=lambda x: x[1], reverse=True)
        pstr += f"Input histogram (low) for sample {i}:"
        for tok, count in toks_counts:
            pstr += f"{tok}({count}), "
        
        pstr += "\n "

        targets_high = targets[sorted_ids[i][0:num_highest]]
        targets_low = targets[sorted_ids[i][-1-num_lowest:-1]]

        labels_counts = {}
        for i, label in enumerate(targets_high):
            if label not in labels_counts:
                labels_counts[label] = 0
            labels_counts[label] += 1
        labels_counts = sorted(labels_counts.items(), key=lambda x: x[1], reverse=True)
        pstr += f"Label histogram (high) for sample {i}:"
        for label, count in labels_counts:
            pstr += f"{label}({count}), "

        pstr += "\n "
        labels_counts = {}
        for i, label in enumerate(targets_low):
            if label not in labels_counts:
                labels_counts[label] = 0
            labels_counts[label] += 1
        labels_counts = sorted(labels_counts.items(), key=lambda x: x[1], reverse=True)
        pstr += f"Label histogram (low) for sample {i}:"
        for label, count in labels_counts:
            pstr += f"{label}({count}), "
        pstr += "\n "
        pstr += "\n "
        pstr +=  "--------------------------------------------------"
        pstr += "\n "
        pstr += "\n "

    if verbose:
        print(pstr)

    if result_path is not None:
        with open(result_path, "w") as f:
            f.write(pstr)

def plot_sim_matrices_with_different_sorting(
        input_emb, data, targets, store_path, sort_by="label", name="sorted by label"):
    # sort samples by label
    q = 0.001

    if type(data) is torch.Tensor:
        print(data)
        data = data.cpu().numpy()
    if type(targets) is torch.Tensor:
        targets = targets.cpu().numpy()

    print("data", data.shape, type(data))
    print("targets", targets.shape, type(targets))
    print("data[:, 0]", data[:, 0].shape, type(data[:, 0]))
    print("data[:, 2]", data[:, 2].shape, type(data[:, 2]))

    if sort_by == "label":
        sorted_ids = np.argsort(targets + (data[:, 0] + data[:, 2]) * q)
    elif sort_by == "input0":
        sorted_ids = np.argsort(data[:, 0] + data[:, 2] * q + targets * q**2)
    elif sort_by == "input1":
        sorted_ids = np.argsort(data[:, 2] + data[:, 0] * q + targets * q**2)
    elif sort_by == "input_sum":
        sorted_ids = np.argsort(data[:, 0] + data[:, 2] + targets * q)
    else:
        raise ValueError(f"Unknown sorting method {sort_by}")

    # sort the input embedding matrix
    sorted_input_emb = input_emb[sorted_ids]
    sorted_data = data[sorted_ids]
    sorted_targets = targets[sorted_ids]
    sorted_sums = sorted_data[:, 0] + sorted_data[:, 2]

    plot_sim_matrix(sorted_input_emb, 
                    sorted_data, 
                    sorted_targets, 
                    sorted_sums,
                    store_path, name=name)





def plot_umap_projection(
        input_emb, data, targets, store_path, name="umap_projection", n_dims=2
):

    reducer = umap.UMAP(
        n_neighbors=15,
        metric="cosine",
        random_state=42,
        n_components=n_dims,
    )
    embedding = reducer.fit_transform(input_emb.cpu().numpy())

    # introduce symbols for sum//113
    symbols = [str(((data[i][0] + data[i][2]) // 113).item()) for i in range(len(data))]

    if n_dims == 2:
        fig = px.scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            color=targets,
            symbol=symbols,
            title=f"UMAP Projection {name}",
            labels={"x": "UMAP 1", "y": "UMAP 2"},
            hover_name=[f"Sample {i}: {data[i]} labeled {targets[i]}" for i in range(len(data))],
        )
    elif n_dims == 3:
        fig = px.scatter_3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            color=targets,
            symbol=symbols,
            title=f"UMAP Projection {name}",
            labels={"x": "UMAP 1", "y": "UMAP 2", "z": "UMAP 3"},
            hover_name=[f"Sample {i}: {data[i]} labeled {targets[i]}" for i in range(len(data))],
        )
    else:
        raise ValueError(f"Invalid number of dimensions {n_dims}")

    os.makedirs(store_path, exist_ok=True)

    fig.write_html(store_path + f"{name}.html")
    fig.write_image(store_path + f"{name}.png")

def concat_embs(param_kernel, layer_names):
    """
    Concatenate the embeddings of the given layer names.
    """
    emb = []
    for layer_name in layer_names:
        emb.append(param_kernel[layer_name])
    return torch.cat(emb, dim=-1)

def plot_sim_matrix(input_emb, data, targets, sums, store_path, name):

    cosine_similarity_matrix = torch.einsum("ij,kj->ik", input_emb, input_emb) / (input_emb.norm(dim=1, keepdim=True) * input_emb.norm(dim=1, keepdim=True).T)

    fig = px.imshow(cosine_similarity_matrix.cpu().numpy(), color_continuous_scale='RdBu', zmin=-1, zmax=1)
    fig.update_layout(
        title=name,
        xaxis_title="Val. sample a+b (a+b mod 113)",
        yaxis_title="Val. sample a+b (a+b mod 113)")
    # center title
    fig.update_layout(title_x=0.4)
    
    # put label on colorbar
    fig.update_coloraxes(colorbar=dict(title="Cosine"))
    # font size
    fig.update_layout(font=dict(size=15))

    # we've got len(target) samples, do a tick only every 100 samples
    #fig.update_xaxes(tickvals=list(range(0, 2000, 100)), ticktext=[f"{sums[i]} ({targets[i]})" for i in range(0, 2000, 100)])
    #fig.update_yaxes(tickvals=list(range(0, 2000, 100)), ticktext=[f"{sums[i]} ({targets[i]})" for i in range(0, 2000, 100)])
    fig.update_xaxes(tickvals=list(range(0, len(targets), 100)), ticktext=[f"{sums[i]} ({targets[i]})" for i in range(0, len(targets), 100)])
    fig.update_yaxes(tickvals=list(range(0, len(targets), 100)), ticktext=[f"{sums[i]} ({targets[i]})" for i in range(0, len(targets), 100)])

    # increase font size
    fig.update_layout(font=dict(size=24))
    fig.update_xaxes(title_font=dict(size=24))
    fig.update_yaxes(title_font=dict(size=24))

    # make

    # make ticks smaller
    fig.update_xaxes(tickfont=dict(size=12))
    #tilt to 45 degrees
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(tickfont=dict(size=12))

    # move colorbar closer to graph
    fig.update_coloraxes(colorbar=dict(yanchor="middle", y=0.5, xanchor="left", x=1.05))
    
    # fig.write_html(store_path + f"cosine_similarity_matrix_{name}.html")
    fig.write_image(store_path + f"cosine_similarity_matrix_{name}.png", width=800, height=600, scale=2)
    fig.write_image(store_path + f"cosine_similarity_matrix_{name}_low_res.png", width=800, height=600, scale=1)
    fig.write_image(store_path + f"cosine_similarity_matrix_{name}.pdf", format="pdf", width=800, height=600, scale=2)

if __name__ == "__main__":
    model, data_path, param_kernel, kernel_matrices, train_loader, val_loader = get_kernels_and_data()

    # sanity check: only samples with the same number inputs should have high similarity w.r.t. input embedding layer
    data, targets, input_emb = get_embedded_samples(param_kernel["input_emb.weight"], val_loader.dataset, labels="accumulate")

    # # compute similarity matrix of all samples
    # dot_similarity_matrix = torch.einsum("ij,kj->ik", input_emb, input_emb)

    # # print highest similarity of samples
    # topk_entries, topk_ids = torch.topk(dot_similarity_matrix, k=10, dim=1)

    # # print top-k samples for the first 50 samples
    # for i in range(50):
    #     print(f"Sample {i}: {data[i]} labeled {targets[i]} has high similarity with")
    #     for j in range(10):
    #         print(f" - Sample {topk_ids[i][j]}: {data[topk_ids[i][j]]} labeled {targets[topk_ids[i][j]]} with similarity {topk_entries[i][j]}")
    #     print()

    get_high_and_low_sim_samples(data=data, targets=targets, input_emb=input_emb, num_samples=10, num_highest=10, num_lowest=10)

    # does the oputput layer condense more around similar labels?
    data, targets, input_emb = get_embedded_samples(param_kernel["decoder.weight"], val_loader.dataset, labels="accumulate")

    get_high_and_low_sim_samples(data=data, targets=targets, input_emb=input_emb, num_samples=10, num_highest=10, num_lowest=10)

    head_emb = torch.cat([param_kernel["head2_encoder.weight"], param_kernel["head2_decoder.weight"]], dim=-1)

    data, targets, input_emb = get_embedded_samples(head_emb, val_loader.dataset, labels="accumulate")

    get_high_and_low_sim_samples(data=data, targets=targets, input_emb=input_emb, num_samples=10, num_highest=10, num_lowest=10)

    plot_dir = "results/modulo_val_results_0.01_w/plots/val_sim_scores/"
    os.makedirs(plot_dir, exist_ok=True)

    for k in tqdm(param_kernel.keys()):
        layer_dir = plot_dir + f"{k}/"
        os.makedirs(layer_dir, exist_ok=True)

        data, targets, input_emb = get_embedded_samples(param_kernel[k], val_loader.dataset, labels="accumulate")
        get_high_and_low_sim_samples(data=data, targets=targets, input_emb=input_emb, num_samples=10, num_highest=10, num_lowest=10, result_path=layer_dir + f"{k}_high_low.txt", verbose=False)
        
        for sorting in ["input_sum"]:
            plot_sim_matrices_with_different_sorting(
                input_emb, data, targets, layer_dir, sort_by=sorting
            )

        # plot the umap projection of the input embedding
        # plot_umap_projection(input_emb, data, targets, layer_dir, name=f"{k}_umap_projection")
    
    combinations = {
        # "head1": ("head1_encoder.weight", "head1_decoder.weight"),
        # "head2": ("head2_encoder.weight", "head2_decoder.weight"),
        # "head3": ("head3_encoder.weight", "head3_decoder.weight"),
        # "head4": ("head4_encoder.weight", "head4_decoder.weight"),

        "embedding": ("input_emb.weight",),

        # combine all att
        "att. encoders": ("head1_encoder.weight", "head2_encoder.weight", "head3_encoder.weight", "head4_encoder.weight"),
       
        "att. decoders": ("head1_decoder.weight", "head2_decoder.weight", "head3_decoder.weight", "head4_decoder.weight"),

        # "emb+head1_enc": ("input_emb.weight", "head1_encoder.weight"),
        # "emb+head2_enc": ("input_emb.weight", "head2_encoder.weight"),
        # "emb+head3_enc": ("input_emb.weight", "head3_encoder.weight"),
        # "emb+head4_enc": ("input_emb.weight", "head4_encoder.weight"),

        # "input_emb+head1": ("input_emb.weight", "head1_encoder.weight", "head1_decoder.weight"),
        # "input_emb+head2": ("input_emb.weight", "head2_encoder.weight", "head2_decoder.weight"),
        # "input_emb+head3": ("input_emb.weight", "head3_encoder.weight", "head3_decoder.weight"),
        # "input_emb+head4": ("input_emb.weight", "head4_encoder.weight", "head4_decoder.weight"),
        
        # "linear_layer_weights": ("linear1.weight", "linear2.weight"),
        "linear1": ("linear1.weight", "linear1.bias"),
        "linear2": ("linear2.weight", "linear2.bias"),
        # "linear_layer": ("linear1.weight", "linear2.weight", "linear1.bias", "linear2.bias"),

        # "linear+decoder": ("linear1.weight", "linear2.weight", "linear1.bias", "linear2.bias", "decoder.weight"),

        "decoder": ("decoder.weight",),

        # combine all all layers
        "model": tuple(param_kernel.keys())
    }

    for k, v in tqdm(combinations.items()):
        print(k)
        layer_dir = plot_dir + f"{k}/"
        os.makedirs(layer_dir, exist_ok=True)

        # concatenate the embeddings
        emb = concat_embs(param_kernel, v)
        data, targets, input_emb = get_embedded_samples(emb, val_loader.dataset, labels="accumulate")
        get_high_and_low_sim_samples(data=data, targets=targets, input_emb=input_emb, num_samples=10, num_highest=10, num_lowest=10, result_path=layer_dir + f"{k}_high_low.txt", verbose=False)
        
        for sorting in ["input_sum"]:
            plot_sim_matrices_with_different_sorting(
                input_emb, data, targets, layer_dir, sort_by=sorting, name=k
            )

        # plot the umap projection of the input embedding
        # plot_umap_projection(input_emb, data, targets, layer_dir, name=f"{k}_umap_projection")




    # plot umap for linear2 in 3d
    emb = concat_embs(param_kernel, ["linear2.weight"])
    data, targets, input_emb = get_embedded_samples(emb, val_loader.dataset, labels="accumulate")
    plot_umap_projection(input_emb, data, targets, plot_dir + "linear2_3d/", name="linear2_3d", n_dims=3)


    # plot UMAP for all
    emb = concat_embs(param_kernel, combinations["all"])
    data, targets, input_emb = get_embedded_samples(emb, val_loader.dataset, labels="accumulate")
    plot_umap_projection(input_emb, data, targets, plot_dir + "all/", name="all_3d", n_dims=3)

