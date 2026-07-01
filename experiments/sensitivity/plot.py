# import torch
# from torch import nn
# import torch.nn.functional as F
# import time
import json
import os
import numpy as np
import argparse
import plotly.express as px
import pandas as pd

# from experiments.train_models.mnist10_model import train_mnist, MNISTDataset

parser = argparse.ArgumentParser(description='Analyze ExPLAIND sensitivity.')
parser.add_argument('--experiment_path', type=str, default="results/mnist_sparse/", help='Path to the experiment folder')
args = parser.parse_args()

experiment_path = args.experiment_path

# device = "cuda" if torch.cuda.is_available() else "cpu"

print(experiment_path)
with open(experiment_path + "sensitivity_results_no_scal.json", "r") as f:
    data_dict = json.load(f)

data_dict.pop("path")

df = pd.DataFrame(data_dict)
# plot it
fig = px.line(df, x="frac", y="acc", error_y="std", title="ExPLAIND Accuracy subsampling checkpoints")
fig.update_layout(xaxis_title="Fraction of checkpoints", yaxis_title="EPK Accuracy")

# times new roman
fig.update_layout(font=dict(family="Times New Roman", size=18))
# choose whit background thme
fig.update_layout(template="plotly_white")

# make ticks at 0.5, 0.7, 0.9, 0.95, 0.99
# fig.update_xaxes(tickvals=alphas[:-1], ticktext=[str(a) for a in alphas][:-1])

fig.write_image(experiment_path + "sensitivity_plot.png", scale=2)
fig.write_image(experiment_path + "sensitivity_plot.pdf", scale=2)

print(experiment_path)
with open(experiment_path + "sensitivity_results_rescale.json", "r") as f:
    data_dict = json.load(f)

data_dict.pop("path")

df = pd.DataFrame(data_dict)
# plot it
fig = px.line(df, x="frac", y="acc", error_y="std", title="ExPLAIND Accuracy subsampling checkpoints")
fig.update_layout(xaxis_title="Fraction of checkpoints", yaxis_title="EPK Accuracy")

# times new roman
fig.update_layout(font=dict(family="Times New Roman", size=18))
# choose whit background thme
fig.update_layout(template="plotly_white")

# make ticks at 0.5, 0.7, 0.9, 0.95, 0.99
# fig.update_xaxes(tickvals=alphas[:-1], ticktext=[str(a) for a in alphas][:-1])



fig.write_image(experiment_path + "sensitivity_rescale_plot.png", scale=2)
fig.write_image(experiment_path + "sensitivity_rescale_plot.pdf", scale=2)