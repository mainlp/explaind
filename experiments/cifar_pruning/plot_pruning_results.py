from explaind.cifar_model.model import construct_rn9
import json
import plotly.graph_objects as go
import pandas as pd

# load all the pruning results
ours_coarse = []
ours_fine = []
baselines_coarse = []
baselines_fine = []

inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", #"#F0E442", 
                    "#0072B2", "#D55E00", "#CC79A7"]

method2color = {
    "ExPLAIND": inclusive_colors[0],
    "Random": inclusive_colors[1],
    "Baseline": inclusive_colors[2],
}

for seed in [1, 2, 3, 4, 5]:
    with open(f'results/cifar2/pruning_results/ours/seed_{seed}/results.json', 'r') as f:
        results = json.load(f)
        results['seed'] = seed
        ours_coarse.append(results)

    with open(f'results/cifar2/pruning_results/ours_fine/seed_{seed}/results.json', 'r') as f:
        results = json.load(f)
        results['seed'] = seed
        ours_fine.append(results)

    with open(f'results/cifar2/pruning_results/baseline_magn/results_0.7_0.99_seed_{seed}.json', 'r') as f:
        results = json.load(f)
        results['seed'] = seed
        baselines_coarse.append(results)

    with open(f'results/cifar2/pruning_results/baseline_magn/results_0.97_0.999_seed_{seed}.json', 'r') as f:
        results = json.load(f)
        results['seed'] = seed
        baselines_fine.append(results)
       
print(len(ours_coarse), len(ours_fine), len(baselines_coarse), len(baselines_fine))

# prepare all the data into dicts and put into df
df_coarse = []
df_fine = []

for i, result in enumerate(ours_fine):
    for j, frac in enumerate(result["fractions"]):
        df_fine.append({
            "fraction": 1 - frac,
            "val. acc.": result["overall_accuracies_across"][j],
            "model acc.": result['model_accuracies_across'][j],
            "KL div.": result["kl_across"][j],
            "seed": result["seed"],
            "name": result["name"],
            "method": "ExPLAIND",
            "num_params": result["no_params_left_across"][j],
        })
        df_fine.append({
            "fraction": 1 - frac,
            "val. acc.": result["overall_accuracies_random"][j],
            "model acc.": result['model_accuracies_random'][j],
            "KL div.": result["kl_random"][j],
            "seed": result["seed"],
            "method": "Random",
            "name": result["name"],
            "num_params": result["no_params_left_random"][j],
        })

for i, result in enumerate(ours_coarse):
    for j, frac in enumerate(result["fractions"]):
        df_coarse.append({
            "fraction": 1 - frac,
            "val. acc.": result["overall_accuracies_across"][j],
            "model acc.": result['model_accuracies_across'][j],
            "KL div.": result["kl_across"][j],
            "seed": result["seed"],
            "name": result["name"],
            "method": "ExPLAIND",
            "num_params": result["no_params_left_across"][j],
        })
        df_coarse.append({
            "fraction": 1 - frac,
            "val. acc.": result["overall_accuracies_random"][j],
            "model acc.": result['model_accuracies_random'][j],
            "KL div.": result["kl_random"][j],
            "seed": result["seed"],
            "method": "Random",
            "name": result["name"],
            "num_params": result["no_params_left_random"][j],
        })

for i, result in enumerate(baselines_fine):
    for j, frac in enumerate(result["fractions"]):
        df_fine.append({
            "fraction": frac,
            "val. acc.": result["overall_accs"][j],
            "model acc.": result['model_accs'][j],
            "KL div.": result["kl_divs"][j],
            "seed": result["seed"],
            "name": "baseline",
            "method": "Baseline",
            "num_params": result["nparams_l"][j][-1],
        })

for i, result in enumerate(baselines_coarse):
    for j, frac in enumerate(result["fractions"]):
        df_coarse.append({
            "fraction": frac,
            "val. acc.": result["overall_accs"][j],
            "model acc.": result['model_accs'][j],
            "KL div.": result["kl_divs"][j],
            "seed": result["seed"],
            "name": "baseline",
            "method": "Baseline",
            "num_params": result["nparams_l"][j][-1],
        })

df_coarse = pd.DataFrame(df_coarse)
df_fine = pd.DataFrame(df_fine)

len(df_coarse), len(df_fine)

# get model and count params
model = construct_rn9(num_classes=2)
num_params = sum(p.numel() for p in model.parameters())

df_coarse["fraction_pruned"] = 1 - (df_coarse["num_params"] / num_params)
df_fine["fraction_pruned"] = 1 - (df_fine["num_params"] / num_params)

# plot the data
# get mean and std over seeds
df_coarse_mean = df_coarse.groupby(["fraction", "method"]).agg(
    {
        "val. acc.": ["mean", "std"],
        "model acc.": ["mean", "std"],
        "KL div.": ["mean", "std"],
        "fraction_pruned": ["mean", "std"],
    }
).reset_index()

df_fine_mean = df_fine.groupby(["fraction", "method"]).agg(
    {
        "val. acc.": ["mean", "std"],
        "model acc.": ["mean", "std"],
        "KL div.": ["mean", "std"],
        "fraction_pruned": ["mean", "std"],
    }
).reset_index()



# rename columns
df_coarse_mean.columns = ["fraction", "method", "Test accuracy", "Test accuracy std", "Model accuracy", "Model accuracy std", "KL divergence", "KL divergence std", "fraction_pruned", "fraction_pruned std"]
df_fine_mean.columns = ["fraction", "method", "Test accuracy", "Test accuracy std", "Model accuracy", "Model accuracy std", "KL divergence", "KL divergence std", "fraction_pruned", "fraction_pruned std"]
# sort by fraction pruned
df_coarse_mean = df_coarse_mean.sort_values(by=["fraction_pruned"])
df_fine_mean = df_fine_mean.sort_values(by=["fraction_pruned"])

# make plots
def plot_pruning_results(df, title, save_path, field="val. acc."):
    fig = go.Figure()

    # add lines for each method
    for method in df["method"].unique():
        df_method = df[df["method"] == method]
        fig.add_trace(
            go.Scatter(
                x=df_method["fraction_pruned"],
                y=df_method[field],
                mode="lines+markers",
                name=method,
                line=dict(color=method2color[method]),
                error_y=dict(
                    type="data",
                    array=df_method[f"{field} std"],
                    visible=True,
                ),
                error_x=dict(
                    type="data",
                    array=df_method["fraction_pruned std"],
                    visible=True,
                )
            )
        )

    # update layout
    fig.update_layout(
        title=title,
        xaxis_title="Frac. of param. pruned",
        yaxis_title=field,
        legend_title="Method",
    )

    # increase font size
    fig.update_layout(
        font=dict(
            family="Arial",
            size=26,
            color="Black"
        )
    )



    # make background white with black grid
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="LightGray",
            zeroline=True,
            zerolinecolor="LightGray",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="LightGray",
            zeroline=True,
            zerolinecolor="LightGray",
        ),
    )
    fig.update_layout(legend=dict(itemsizing='constant'))

    # save the figure
    fig.write_image(save_path.replace(".png", "_low_res.png"), width=800, height=600)
    fig.write_image(save_path, width=800, height=600, scale=2)
    fig.write_image(save_path.replace(".png", ".pdf"))


for stat in ["Test accuracy", "Model accuracy", "KL divergence"]:
    plot_pruning_results(df_fine_mean, f"Pruning results (fine) {stat}", f"results/cifar2/pruning_results/pruning_fine_{stat}.png", field=stat)
    plot_pruning_results(df_coarse_mean, f"Pruning results (coarse) {stat}", f"results/cifar2/pruning_results/pruning_coarse_{stat}.png", field=stat)