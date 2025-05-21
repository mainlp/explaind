import torch
from explaind.model_paths import ModelPath
import json
import plotly.graph_objects as go
import numpy as np
from explaind.modulo_model.model import SingleLayerTransformerClassifier

import os
import pandas as pd
from tqdm import tqdm

import plotly.express as px


device = "cpu"

def get_kernels_and_data():
    experiment_path = "results/modulo_val_results_0.01_w/"
    model_checkpoint_path = experiment_path + "model_checkpoint.pt"
    optimizer_checkpoint_path = experiment_path + "optimizer_checkpoint.pt"
    data_checkpoint_path = experiment_path + "data_checkpoint.pt"

    # load datasets
    train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
    val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")


    config = {}
    with open(experiment_path + "config.json", "r") as f:
        config = json.load(f)

    # load model and optimizer
    base_model = SingleLayerTransformerClassifier(config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])
    model = ModelPath(model=base_model,
                    device=device, 
                    checkpoint_path=model_checkpoint_path,
                    features_to_cpu=False,
                    parallelize_over_targets=True)
    
    base_model.load_state_dict(model.checkpoints[-1])

    param_kernel = torch.load(experiment_path + "param_kernel.pt")

    return param_kernel, base_model, train_loader, val_loader

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
        scores_acc = scores.sum(dim=2)  # empty dimension
        
    return dataset.data, dataset.targets, scores_acc

def get_data():
    experiment_path = "results/modulo_val_results_0.01_w/"
    
    train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
    val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")
    
    return train_loader, val_loader

def get_cosine_of_diffs_matrix(data, freq):

    # conmvert to float
    data = data.to(torch.float32).cpu().numpy()

    sums = data[:, 0] + data[:, 2]

    # repeat the sums and plot the pairwise differences of all samples
    diffs = np.zeros((data.shape[0], data.shape[0]), dtype=data.dtype)
    for i in tqdm(range(data.shape[0])):
        diffs[i] = (sums - sums[i])

    scaled = (diffs / freq) * 2 * np.pi


    cosines = np.cos(scaled)
    # print(f"Similarities shape: {similarities.shape}")

    return cosines

def get_sin_of_diffs_matrix(data, freq):
    data = data.to(torch.float32).cpu().numpy()

    sums = data[:, 0] + data[:, 2]
    # repeat the sums and plot the pairwise differences of all samples
    diffs = np.zeros((data.shape[0], data.shape[0]), dtype=data.dtype)
    for i in tqdm(range(data.shape[0])):
        diffs[i] = (sums - sums[i])
    scaled = (diffs / freq) * 2 * np.pi
    sines = np.sin(scaled)

    return sines


def plot_sim_matrix(cosine_similarity_matrix, targets, sums, store_path, name):

    #cosine_similarity_matrix = torch.einsum("ij,kj->ik", input_emb, input_emb) / (input_emb.norm(dim=1, keepdim=True) * input_emb.norm(dim=1, keepdim=True).T)

    fig = px.imshow(cosine_similarity_matrix, color_continuous_scale='RdBu', zmin=-1, zmax=1)
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
    
    # load model and data
    param_kernel, base_model, train_loader, val_loader = get_kernels_and_data()

    # get sorted val_set
    val_set = val_loader.dataset
    val_targets = val_set.targets

    val_sums = val_set.data[:, 0] + val_set.data[:, 2]

    sorted_ids = np.argsort(val_sums)

    sorted_val_set = val_set.data[sorted_ids]
    sorted_val_targets = val_targets[sorted_ids]
    val_sums_sorted = val_sums[sorted_ids].cpu().numpy().tolist()

    results_path = "results/modulo_cosines/"
    os.makedirs(results_path, exist_ok=True)

    matrices = []

    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    my_guess = [1, 2, 3, 5, 53, 73, 79, 113]

    nandas_analysis = [14, 35, 41, 42, 52]

    for freq in nandas_analysis:
        # get cosine of sums matrix
        cosine_matrix = get_cosine_of_diffs_matrix(sorted_val_set, freq)
        matrices.append(cosine_matrix)

        # plot cosine similarity matrix
        # plot_sim_matrix(cosine_matrix, sorted_val_targets, results_path, f"freq_{freq}")


    # plot sum
    summed = sum(matrices)
    summed = summed / summed.max()
    plot_sim_matrix(summed, sorted_val_targets, val_sums_sorted, results_path, "nandas_analysis")

    # perform a linear regression to find the actual values
    embs_linear2 = torch.cat([param_kernel["linear2.weight"], param_kernel["linear2.bias"]], dim=-1)
    embs_linear2 = embs_linear2.sum(dim=1).sum(dim=1)
    sim_matrix = cosine_similarity_matrix = torch.einsum("ij,kj->ik", embs_linear2, embs_linear2) / (embs_linear2.norm(dim=1, keepdim=True) * embs_linear2.norm(dim=1, keepdim=True).T)

    sums = val_set.data[:, 0] + val_set.data[:, 2]

    X = []
    Y = []

    for i in tqdm(range(sim_matrix.shape[0])):
        for j in range(sim_matrix.shape[1]):
            # flip a 0.1 coin
            #
            X.append(sums[i] - sums[j])
            Y.append(sim_matrix[i, j].item())

    print(len(X), len(Y))

    all_primes = list(range(2, 114))

    # repeat Xs 113 times
    X = np.array(X)
    X = np.repeat(X, len(all_primes)*2).reshape(-1, len(all_primes)*2).astype(np.float32)
   
    for freq in all_primes: #tqdm(range(len(all_primes))):
        # features.append(np.cos(x / freq * 2 * np.pi))
        X[:, freq] = np.cos(X[:, freq] / all_primes[freq] * 2 * np.pi)
        X[:, freq + len(all_primes)] = np.sin(X[:, freq] / all_primes[freq] * 2 * np.pi)

    # fit a linear model
    from sklearn.linear_model import LinearRegression, Lasso
    # model = LinearRegression(fit_intercept=False, positive=True)
    model = Lasso(alpha=0.001, fit_intercept=False)
    model.fit(X, Y)
    print(model.coef_)

    # freqs of non-zero
    freqs = np.where(model.coef_ != 0)[0]
    print("Frequencies of non-zero cos coefficients: ", [all_primes[i] for i in freqs], model.coef_[freqs])
    print("Frequencies of non-zero sin coefficients: ", [all_primes[i + len(all_primes)] for i in freqs], model.coef_[freqs + len(all_primes)])


    # load coeffs from before
    coeffs = pd.read_csv("results/modulo_cosines/lr_coeffs_primes.csv")

    coeffs = coeffs[coeffs["abs_coeff"] > 0.0]

    coeffs_list = list(zip(coeffs["freq"], coeffs["coeff"]))

    matrices = []
    for freq, coeff in coeffs_list:
        # get cosine of sums matrix
        if coeff > 0:
            cosine_matrix = get_cosine_of_diffs_matrix(sorted_val_set, freq)
            matrices.append(cosine_matrix * coeff)

    summed = sum(matrices)
    # summed = summed / summed.max()
    plot_sim_matrix(summed, sorted_val_targets, val_sums_sorted, results_path, "linear_regression_all_freq")

    # plot matrix with these coeffs
    matrices = []
    for i, freq in enumerate(model.coef_):
        # get cosine of sums matrix
        #
        if i < len(all_primes):
            cosine_matrix = get_cosine_of_diffs_matrix(sorted_val_set, all_primes[i])
        else:
            cosine_matrix = get_sin_of_diffs_matrix(sorted_val_set, all_primes[i - len(all_primes)])

        matrices.append(cosine_matrix * freq)

    # plot cosine similarity matrix
    summed = sum(matrices)
    # summed = summed / summed.max()
    plot_sim_matrix(summed, sorted_val_targets, val_sums_sorted, results_path, "linear_regression_primes_freq")

    # store coeffs
    coeffs = pd.DataFrame(model.coef_, columns=["coeff"])
    coeffs["freq"] = all_primes + all_primes
    coeffs["function"] = ["cos"] * len(all_primes) + ["sin"] * len(all_primes)
    coeffs["abs_coeff"] = coeffs["coeff"].abs()
    coeffs = coeffs.sort_values(by="abs_coeff", ascending=False)
    coeffs.to_csv(results_path + "lr_coeffs_primes.csv", index=False)


    # sort by sum
    # data = val_set.data
    
    # sorted_ids = np.argsort(sums)

    # sorted_val_set = val_set.data[sorted_ids]
    # sorted_val_targets = val_targets[sorted_ids]

    # sorted_sums = sums[sorted_ids].cpu().numpy().tolist()
    # # to str
    # sorted_sums = [str(int(x)) for x in sorted_sums]


    # do a pca over the embs of linear2
    layer2_emb = torch.cat([param_kernel["linear2.weight"], param_kernel["linear2.bias"]], dim=-1)

    data, targets, layer2_emb = get_embedded_samples(layer2_emb, val_loader.dataset, labels="accumulate")
    sums = data[:, 0] + data[:, 2]

    from sklearn.decomposition import PCA

    pca = PCA(n_components=len(all_primes))
    
    transformed = pca.fit_transform(layer2_emb.cpu().numpy())

    fig = go.Figure()
    for i in range(transformed.shape[1]):
        # scatter plot
        print(i, end="\r")
        fig.add_trace(go.Scatter(x=sums, y=transformed[:, i], mode='markers', name=f"PC{i}"))
    
    fig.update_layout(
        title="PCA of Linear2 Embeddings",
        xaxis_title="Samples (sorted by target)",
        yaxis_title="PCs")
    # make markers small
    fig.update_traces(marker=dict(size=4))

    fig.write_html(results_path + "pca_linear2_emb.html")
    fig.write_image(results_path + "pca_linear2_emb.png")

    # also plot activations in the same way
    # get the activations of the activations linear2

    data = sorted_val_set.data.to(device)
    targets = sorted_val_targets

    # get encoder outs of model
    with torch.no_grad():
        outputs, activations = base_model.forward(data, output_activations=True)
        enc_activations = activations["encoder_output"]
            
    enc_activations = enc_activations[:, -1] # take last pos that is responsible for the prediction

    # plot sim matrix
    sim_matrix = cosine_similarity_matrix = torch.einsum("ij,kj->ik", enc_activations, enc_activations) / (enc_activations.norm(dim=1, keepdim=True) * enc_activations.norm(dim=1, keepdim=True).T)

    # plot sim matrix
    plot_sim_matrix(sim_matrix.cpu().numpy(), sorted_val_targets, results_path, "encoder_activations_wred_rescaled0.5")
