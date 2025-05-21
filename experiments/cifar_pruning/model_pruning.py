import torch

from explaind.model_paths import ModelPath
from explaind.cifar_model.model import construct_rn9

import json
import plotly.graph_objects as go
from copy import deepcopy
import numpy as np
import pandas as pd
import os


device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"


def mask_by_threshold(param, scores, threshold, param_shape):
    scores = scores.reshape(param_shape)
    # create a mask
    mask = torch.zeros_like(scores).to(device)
    mask[scores >= threshold] = 1
    # apply the mask to the parameter
    masked = (param * mask).to(device)
    return masked

def get_threshold(scores, k):
    # get the top k indices
    sorted_vals = scores.abs().flatten().sort(descending=True)[0]
    threshold = sorted_vals[k]
    return threshold

def eval_against_ground_truth(base_model, 
                               pruned_model_dict,
                               base_model_preds,
                               test_loader):
    # load the model
    base_model.to(device)
    base_model.load_state_dict(pruned_model_dict)

    # get predictions
    preds = []
    ground_truth = []
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            pred = base_model(x)
            preds.append(pred.cpu())
            ground_truth.append(y.cpu())

    # compare the predictions
    preds = torch.cat(preds, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)

    # accuracy
    correct = (preds.argmax(dim=1) == base_model_preds.argmax(dim=1)).float().sum()
    model_accuracy = correct / preds.shape[0]

    correct_ground_truth = (preds.argmax(dim=1) == ground_truth).float().sum()
    overall_accuracy = correct_ground_truth / preds.shape[0]

    kl_divergence = torch.nn.functional.kl_div(preds.log_softmax(dim=1), base_model_preds.log_softmax(dim=1), reduction="mean", log_target=True)
    
    # get the number of pruned parameters by layer
    no_pruned_by_layer = {}
    for name, param in pruned_model_dict.items():
        no_pruned_by_layer[name] = (param == 0).sum().item() / param.numel()

    return model_accuracy, overall_accuracy, kl_divergence, no_pruned_by_layer


def prune_by_importance_and_eval(importance_scores, results_path, state_dict, base_model, test_loader, name, fractions):

    model_accuracies_by_module = []
    overall_accuracies_by_module = []
    model_accuracies_random = []
    overall_accuracies_random = []
    model_accuracies_across = []
    overall_accuracies_across = []
    no_params_left_across = []
    no_params_left_random = []

    kl_by_module = []
    kl_across =[]
    kl_random = []

    no_params_pruned_across = []

    model_preds = []
    ground_truth = []
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            mp = base_model(x).cpu()
            gd = y.cpu()
        
        model_preds.append(mp)
        ground_truth.append(gd)
    
    model_preds = torch.cat(model_preds, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)

    for frac in fractions:
        # prune the model
        model_dict_by_module = deepcopy(state_dict)
        model_dict_across_modules = deepcopy(state_dict)
        random_pruning = deepcopy(state_dict)

        # for across all modules
        all_scores = torch.cat([param.flatten() for param in importance_scores.values()], dim=0)
        k = int(frac * all_scores.numel()) - 1
        overall_threshold = get_threshold(all_scores, k)

        for name, param in model_dict_by_module.items():
            if frac == 1.0 or name in {"9.weight"}: # , "0.0.weight"}:  # skip the last layer
                continue
            module_scores = importance_scores[name]
            k = int(frac * np.prod(module_scores.shape)) - 1

            threshold_by_module = get_threshold(module_scores, k)

            model_dict_by_module[name] = mask_by_threshold(param, module_scores, threshold_by_module, model_dict_by_module[name].shape)
            model_dict_across_modules[name] = mask_by_threshold(param, module_scores, overall_threshold, model_dict_across_modules[name].shape)

            random_mask = torch.zeros_like(param).to(device)
            # flip k random indices to 1
            if k > 0:
                random_indices = torch.randperm(module_scores.numel())[:k]
                random_mask.view(-1)[random_indices] = 1
            #
            # apply the mask to the parameter
            random_pruning[name] = (param * random_mask).to(device)

        # eval all pruned models
        model_accuracy_by_module, overall_accuracy_by_module, kl_divergence_by_module, _ = \
            eval_against_ground_truth(base_model, 
                                    model_dict_by_module,
                                    model_preds,
                                    test_loader)
        model_accuracy_across_modules, overall_accuracy_across_modules, kl_divergence_across_modules, nps_across = \
            eval_against_ground_truth(base_model,
                                    model_dict_across_modules,
                                    model_preds,
                                    test_loader)
        model_accuracy_random, overall_accuracy_random, kl_divergence_random, _ = \
            eval_against_ground_truth(base_model,
                                    random_pruning,
                                    model_preds,
                                    test_loader)

        # store the accuracies
        model_accuracies_by_module.append(model_accuracy_by_module.item())
        overall_accuracies_by_module.append(overall_accuracy_by_module.item())
        model_accuracies_random.append(model_accuracy_random.item())
        overall_accuracies_random.append(overall_accuracy_random.item())
        model_accuracies_across.append(model_accuracy_across_modules.item())
        overall_accuracies_across.append(overall_accuracy_across_modules.item())
        kl_by_module.append(kl_divergence_by_module.item())
        kl_across.append(kl_divergence_across_modules.item())
        kl_random.append(kl_divergence_random.item())
        no_params_pruned_across.append(nps_across)

        # count non-zero params for each
        no_params_random = 0
        for name, param in random_pruning.items():
            no_params_random += (param != 0).sum().item()
        no_params_across = 0
        for name, param in model_dict_across_modules.items():
            no_params_across += (param != 0).sum().item()

        no_params_left_across.append(no_params_across)
        no_params_left_random.append(no_params_random)

        print(f"Fraction: {frac:.3f} | Model Accuracy Random: {model_accuracy_random:.4f} | Overall Accuracy Random: {overall_accuracy_random:.4f} | Model Accuracy Across Modules: {model_accuracy_across_modules:.4f} | Overall Accuracy Across Modules: {overall_accuracy_across_modules:.4f}")
        print(f"Fraction: {frac:.3f} | KL Divergence Across Modules: {kl_divergence_across_modules:.4f} | KL Divergence Random: {kl_divergence_random:.4f}")
        print(f"Fraction: {frac:.3f} | No. Params Left Random: {no_params_random} | No. Params Left Across Modules: {no_params_across}")
    # save_results
    results = {
        "model_accuracies_by_module": model_accuracies_by_module,
        "overall_accuracies_by_module": overall_accuracies_by_module,
        "model_accuracies_random": model_accuracies_random,
        "overall_accuracies_random": overall_accuracies_random,
        "model_accuracies_across": model_accuracies_across,
        "overall_accuracies_across": overall_accuracies_across,
        "kl_by_module": kl_by_module,
        "kl_across": kl_across,
        "kl_random": kl_random,
        "no_params_pruned_across": no_params_pruned_across,
        "fractions": fractions,
        "name": name,
        "no_params_left_across": no_params_left_across,
        "no_params_left_random": no_params_left_random,
    }

    with open(results_path + "results.json", 'w') as f:
        json.dump(results, f)



def plot_pruning_results(results_paths, result_path, strategies=["by_module", "across_modules", "random"]):
    
    os.makedirs(result_path, exist_ok=True)

    inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
   
    # plot the scores
    test_set_acc_fig = go.Figure()
    with_model_acc_fig = go.Figure()
    kl_fig = go.Figure()

    random_kl = []
    random_model_acc = []
    random_test_acc = []

    for i, rpath in enumerate(results_paths):
        with open(rpath + "/results.json", 'r') as f:
            results = json.load(f)
        fractions = results["fractions"]
        model_accuracies_by_module = results["model_accuracies_by_module"]
        overall_accuracies_by_module = results["overall_accuracies_by_module"]
        model_accuracies_random = results["model_accuracies_random"]
        overall_accuracies_random = results["overall_accuracies_random"]
        model_accuracies_across = results["model_accuracies_across"]
        overall_accuracies_across = results["overall_accuracies_across"]
        kl_by_module = results["kl_by_module"]
        kl_across = results["kl_across"]
        kl_random = results["kl_random"]
        no_params_pruned_across = results["no_params_pruned_across"]
        name = results["name"]
        color = inclusive_colors[i % len(inclusive_colors)]

        random_kl.append(kl_random)
        random_model_acc.append(model_accuracies_random)
        random_test_acc.append(overall_accuracies_random)

        # plot the accuracies
        if "by_module" in strategies:
            with_model_acc_fig.add_trace(go.Scatter(x=fractions, y=model_accuracies_by_module, mode='lines+markers', name=f'Model Accuracy {name}', line=dict(color=color, dash='dash')))
            test_set_acc_fig.add_trace(go.Scatter(x=fractions, y=overall_accuracies_by_module, mode='lines+markers', name=f'Accuracy {name}', line=dict(color=color, dash='dash')))
            kl_fig.add_trace(go.Scatter(x=fractions, y=kl_by_module, mode='lines+markers', name=f'KL Divergence {name}', line=dict(color=color, dash='dash')))

        if "across_modules" in strategies:
            with_model_acc_fig.add_trace(go.Scatter(x=fractions, y=model_accuracies_across, mode='lines+markers', name=f'Model Accuracy {name}', line=dict(color=color)))
            test_set_acc_fig.add_trace(go.Scatter(x=fractions, y=overall_accuracies_across, mode='lines+markers', name=f'Accuracy {name}', line=dict(color=color)))
            kl_fig.add_trace(go.Scatter(x=fractions, y=kl_across, mode='lines+markers', name=f'KL Divergence {name}', line=dict(color=color)))


    # add traces of averaged random results
    random_kl = np.mean(np.array(random_kl), axis=0)
    random_model_acc = np.mean(np.array(random_model_acc), axis=0)
    random_test_acc = np.mean(np.array(random_test_acc), axis=0)
    random_kl_std = np.std(np.array(random_kl), axis=0)
    random_model_acc_std = np.std(np.array(random_model_acc), axis=0)
    random_test_acc_std = np.std(np.array(random_test_acc), axis=0)

    with_model_acc_fig.add_trace(go.Scatter(x=fractions, y=random_model_acc, mode='lines+markers', name='Random', line=dict(color="black")))
    with_model_acc_fig.add_trace(go.Scatter(x=fractions, y=random_model_acc + random_model_acc_std, mode='lines', name='Random', line=dict(color="black", dash='dash')))
    with_model_acc_fig.add_trace(go.Scatter(x=fractions, y=random_model_acc - random_model_acc_std, mode='lines', name='Random', line=dict(color="black", dash='dash')))
    test_set_acc_fig.add_trace(go.Scatter(x=fractions, y=random_test_acc, mode='lines+markers', name='Random', line=dict(color="black")))
    test_set_acc_fig.add_trace(go.Scatter(x=fractions, y=random_test_acc + random_test_acc_std, mode='lines', name='Random', line=dict(color="black", dash='dash')))
    test_set_acc_fig.add_trace(go.Scatter(x=fractions, y=random_test_acc - random_test_acc_std, mode='lines', name='Random', line=dict(color="black", dash='dash')))
    kl_fig.add_trace(go.Scatter(x=fractions, y=random_kl, mode='lines+markers', name='Random', line=dict(color="black")))
    kl_fig.add_trace(go.Scatter(x=fractions, y=random_kl + random_kl_std, mode='lines', name='Random', line=dict(color="black", dash='dash')))
    kl_fig.add_trace(go.Scatter(x=fractions, y=random_kl - random_kl_std, mode='lines', name='Random', line=dict(color="black", dash='dash')))

    test_set_acc_fig.update_layout(title='CIFAR-2 Pruning: Test Set Accuracy After Pruning)',
                    xaxis_title='Fraction of Parameters Remaining',
                    yaxis_title='Val Accuracy',)
    test_set_acc_fig.write_image(experiment_path + f"/model_accuracy_with_pruning_test_split_{fractions[0]}_{fractions[-1]}.png")
    test_set_acc_fig.write_html(experiment_path + f"/model_accuracy_with_pruning_test_split_{fractions[0]}_{fractions[-1]}.html")

    # plot the KL divergence
    # divide all by 500
    kl_fig.update_layout(title='CIFAR-2 Pruning: KL Divergence With Model on Test Split',
                    xaxis_title='Fraction of Parameters Remaining',
                    yaxis_title='KL Divergence (Mean)')

    # make y axis show 0. numbers
    # fig.update_yaxes(tickvals=np.arange(0, 0.3, 0.01), ticktext=[f"{i:.2f}" for i in np.arange(0, 0.3, 0.01)])

    kl_fig.write_image(experiment_path + f"/kl_divergence_with_pruning_test_split_{fractions[0]}_{fractions[-1]}.png")
    kl_fig.write_html(experiment_path + f"/kl_divergence_with_pruning_test_split_{fractions[0]}_{fractions[-1]}.html")


    # plot number of params pruned by layer
    # barplot for each fraction ploting the number of params pruned by layer
    for i, rpath in enumerate(results_paths):
        with open(rpath + "/results.json", 'r') as f:
            results = json.load(f)

        fig = go.Figure()
        fractions = results["fractions"]
        no_params_pruned_across = results["no_params_pruned_across"]
        name = results["name"]
        color = inclusive_colors[i % len(inclusive_colors)]
    
        df = {
            "layer": [],
            "overall_fraction": [],
            "fraction_pruned": [],
        }
        for i, nps in enumerate(no_params_pruned_across):
            for layer, fraction in nps.items():
                df["layer"].append(layer)
                df["fraction_pruned"].append(fraction)
                df["overall_fraction"].append(fractions[i])

        df = pd.DataFrame(df)

        df["overall_fraction"] = df["overall_fraction"].astype(str)

        # plot barplot
        fig = go.Figure()
        for layer in df["layer"].unique():
            fig.add_trace(go.Bar(x=df[df["layer"] == layer]["overall_fraction"], 
                                y=df[df["layer"] == layer]["fraction_pruned"], 
                                name=layer))
            
        fig.update_layout(title=f'CIFAR-2 Pruning: Number of Parameters Pruned by Layer ({name})',
                            xaxis_title='Fraction of Parameters Remaining',
                            yaxis_title='Fraction of Parameters Pruned',
                            barmode='stack')
        fig.write_image(experiment_path + f"/params_pruned_by_layer_test_split_{fractions[0]}_{fractions[-1]}_{name}.png")
        fig.write_html(experiment_path + f"/params_pruned_by_layer_test_split_{fractions[0]}_{fractions[-1]}_{name}.html")


def load_cifar2_epk_checkpoint(experiment_path, ctype="cifar2"):
    # load the model
    config_path = experiment_path + "/config.json"
    model_checkpoint_path = experiment_path + "/model_epk_0.pt"

    config = json.load(open(config_path, 'r'))

    config["type"] = ctype

    # load training path
    if config["type"] == "cifar10":
        base_model = construct_rn9()
        model = ModelPath(model=base_model, 
                        device=device, 
                        checkpoint_path=model_checkpoint_path, 
                        overwrite=False,
                        features_to_cpu=False,
                        output_dim=10,)
    elif config["type"] == "cifar2":
        base_model = construct_rn9(num_classes=2)
        model = ModelPath(model=base_model, 
                        device=device, 
                        checkpoint_path=model_checkpoint_path, 
                        overwrite=False,
                        features_to_cpu=False,
                        output_dim=2,)
        
    # load the model
    base_model.load_state_dict(model.checkpoints[-1])
    return base_model, model.checkpoints[-1]

def load_seed_set_and_prune(
    seed,
    experiment_path, 
    model, 
    param_kernel,
    fractions=None, 
    name="epk", 
    save_path="results/cifar2/pruning_results/"
):
    # load data
    test_loader = torch.load(experiment_path + f"test_set_500_{seed}.pt")
    val_ids = torch.load(experiment_path + f"val_ids_1500_{seed}.pt")

    epk_importances = {k: v[val_ids].abs().sum(dim=0).sum(dim=0).squeeze() for k, v in param_kernel.items()}

    model_to_prune = deepcopy(model)
    state_dict = deepcopy(model_to_prune.state_dict())

    save_path = save_path + f"seed_{seed}/"
    os.makedirs(save_path, exist_ok=True)
    # prune the model
    prune_by_importance_and_eval(epk_importances, save_path, state_dict, model_to_prune, test_loader, name=name + f"_seed_{seed}", fractions=fractions)

if __name__ == "__main__":
    # set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # run the script
    experiment_path = "results/cifar2/checkpoints/epk/model_epk_0"
    results_path = "results/cifar2/pruning_results/"

    base_model, state_dict = load_cifar2_epk_checkpoint(experiment_path, ctype="cifar2")

    # base_model.load_state_dict(state_dict)

    param_kernel = torch.load(experiment_path + "/param_kernel.pt", map_location=device)

    # val_loader = get_dataloader(batch_size=200, num_workers=1, split='test', shuffle=False, augment=False, type="cifar2")[0]
    # # sample  a test set from the val loader
    # test_ids = torch.randperm(len(val_loader.dataset))
    # val_ids = test_ids[:1500]
    # test_ids = test_ids[1500:]
    # test_set = torch.utils.data.Subset(val_loader.dataset, test_ids)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, shuffle=False, num_workers=1)

    # epk_importances = {k: v[val_ids].abs().sum(dim=0).sum(dim=0).squeeze() for k, v in param_kernel.items()}

    results_path = "results/cifar2/pruning_results/ours/"
    pruning_path = "results/cifar2/pruning_results/"
    os.makedirs(results_path, exist_ok=True)

    # fractions = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # fractions = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03]
    # fractions = [0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    fractions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
    
    for seed in [1, 2, 3, 4, 5]:
        load_seed_set_and_prune(seed, pruning_path, base_model, param_kernel, fractions=fractions, name="epk", save_path=results_path)

    fractions_rem = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03]
    
    results_path = "results/cifar2/pruning_results/ours_fine/"
    for seed in [1, 2, 3, 4, 5]:
        load_seed_set_and_prune(seed, pruning_path, base_model, param_kernel, fractions=fractions_rem, name="epk", save_path=results_path)

