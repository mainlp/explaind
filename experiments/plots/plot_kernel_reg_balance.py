import torch
import os
from tqdm import tqdm

import plotly.express as px
import pandas as pd

device = "cpu"

experiment_path = "results/modulo_val_results/"
plot_dir = "results/modulo_val_results/plots/kernel_reg_balance/"

inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", # "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7"]

train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

# load step matrices
kernel_path = 'results/modulo_val_results/kernel_matrices/'

step_kernels = []  # will be list of dicts of layer_name -> torch tensors of shape (num_val, num_params) = (2000, num_params)
step_regs = []

# takes about 3 minutes (loading 35 GB worth of tensors)
for i in tqdm(range(0, 1939, 50)):
    step_kernels.append(torch.load(kernel_path + f"param_kernel_step_{i}.pt", map_location=device, weights_only=True))
    step_regs.append(torch.load(kernel_path + f"param_reg_kernel_step_{i}.pt", map_location=device, weights_only=True))





combinations = {
        "embedding": ("input_emb.weight",),
        "att. encoders": ("head1_encoder.weight", "head2_encoder.weight", "head3_encoder.weight", "head4_encoder.weight"),
        "att. decoders": ("head1_decoder.weight", "head2_decoder.weight", "head3_decoder.weight", "head4_decoder.weight"),
        "linear1": ("linear1.weight", "linear1.bias"),
        "linear2": ("linear2.weight", "linear2.bias"),
        "decoder": ("decoder.weight",),
        # "model": ('input_emb.weight', 
        #           'head1_encoder.weight', 'head2_encoder.weight', 'head3_encoder.weight', 'head4_encoder.weight', 
        #           'head1_decoder.weight', 'head2_decoder.weight', 'head3_decoder.weight', 'head4_decoder.weight', 
        #           'linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias', 'decoder.weight')
    }

df = []

for i in tqdm(range(len(step_kernels))):
    step_kernel = step_kernels[i]
    step_reg = step_regs[i]


    step_dict = {}
    for layer_name, kernel in step_kernel.items():

        reg_kernel = step_reg[layer_name] * 0.001  # reg terms are not scaled by lr

        print(reg_kernel.shape, kernel.shape)

        k_l1 = kernel.sum(dim=1).abs().sum().item()
        r_l1 = reg_kernel.sum(dim=1).abs().sum().item()

        diff = (kernel.abs() - reg_kernel.abs()).sum().item()

        step_dict[layer_name] = {
            "kernel_l1": k_l1,
            "reg_kernel_l1": r_l1,
            "diff": diff
        }
    # sum up the l1 norms according to the combinations
    for k, v in combinations.items():
        kernel_l1 = 0
        reg_kernel_l1 = 0
        for layer_name in v:
            kernel_l1 += step_dict[layer_name]["kernel_l1"]
            reg_kernel_l1 += step_dict[layer_name]["reg_kernel_l1"]
        df.append({
            "Step": i,
            "Layer": k,
            "Kernel L1 Norm": kernel_l1,
            "Reg. L1 Norm": reg_kernel_l1,
            "Differences": step_dict[layer_name]["diff"]
        })
    
df_plot = pd.DataFrame(df)
df_plot["Kernel/Reg. L1 Norm"] = df_plot["Kernel L1 Norm"] / df_plot["Reg. L1 Norm"]
df_plot["Step"] = df_plot["Step"] * 50

# plot this
fig = px.line(df_plot, x="Step", y="Kernel/Reg. L1 Norm", color="Layer")
fig.update_layout(
    title="Kernel L1 Norm / Reg. L1 Norm",
    xaxis_title="Step",
    yaxis_title="Kernel L1 Norm / Reg. L1 Norm",
    legend_title="Layer",
)
fig.update_traces(mode="lines+markers")

# use inclusive colors
for i, color in enumerate(inclusive_colors):
    fig.data[i].line.color = color
    fig.data[i].marker.color = color

# log scale
fig.update_yaxes(type="log")

# make white background and grey grid lines
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="lightgrey"),
    yaxis=dict(showgrid=True, gridcolor="lightgrey"),)  


os.makedirs(plot_dir, exist_ok=True)

fig.write_image(plot_dir + "kernel_reg_balance_low_res.png", width=800, height=600)
fig.write_image(plot_dir + "kernel_reg_balance.png", width=800, height=600, scale=2)
fig.write_image(plot_dir + "kernel_reg_balance.pdf", width=800, height=600, scale=2)

# df_plot["Differences"] = df_plot["Differences"].apply(lambda x: np.log(np.abs(x) + 1e-10) * np.sign(x))

# plto the differences
fig = px.line(df_plot, x="Step", y="Differences", color="Layer")

fig.update_layout(
    title="",
    xaxis_title="<b>Step",
    yaxis_title="$\Large D(\Theta)$",
    legend_title="Layer",
)
fig.update_traces(mode="lines")

# fig.update_yaxes(type="log")

# use inclusive colors
order = ["embedding", "att. encoders", "att. decoders", "linear1", "linear2", "decoder"]
color_mapping = {
    "embedding": inclusive_colors[0],
    "att. encoders": inclusive_colors[1],
    "att. decoders": inclusive_colors[2],
    "linear1": inclusive_colors[3],
    "linear2": inclusive_colors[4],
    "decoder": inclusive_colors[5],
    "model": inclusive_colors[6],
}

for i, color in enumerate(inclusive_colors):
    fig.data[i].line.color = color_mapping[fig.data[i].name]
    fig.data[i].marker.color = color_mapping[fig.data[i].name]

# make white background and grey grid lines
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
fig.update_traces(line=dict(width=2))
fig.update_layout(legend=dict(itemsizing='constant'))
#make bold
fig.update_layout(legend_font=dict(size=24))
fig.update_layout(font=dict(size=26))
fig.update_xaxes(title_font=dict(size=26))

# showledgend False
fig.update_layout(showlegend=False)

fig.update_layout(font=dict(size=22))
fig.update_xaxes(title_font=dict(size=22))
fig.update_yaxes(title_font=dict(size=22))
fig.update_layout(legend_font=dict(size=16))
# smaller tick font
fig.update_xaxes(tickfont=dict(size=12))
fig.update_yaxes(tickfont=dict(size=12))

fig.write_image(plot_dir + "kernel_reg_balance_differences_low_res.png", width=400, height=340)
fig.write_image(plot_dir + "kernel_reg_balance_differences.png", width=400, height=340, scale=2)
fig.write_image(plot_dir + "kernel_reg_balance_differences.pdf", width=400, height=340, scale=2)
fig.write_html(plot_dir + "kernel_reg_balance_differences.html")

# make aplot with reg and kernel as separate lines

# layers = df_plot["Layer"].unique()
# fig = go.Figure()
# for i, layer in enumerate(layers):
#     fig.add_trace(
#         go.Scatter(
#             x=df_plot[df_plot["Layer"] == layer]["Step"],
#             y=df_plot[df_plot["Layer"] == layer]["Kernel L1 Norm"],
#             mode="lines+markers",
#             name=f"{layer} kernel",
#             line=dict(color=inclusive_colors[i], width=2),
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=df_plot[df_plot["Layer"] == layer]["Step"],
#             y=df_plot[df_plot["Layer"] == layer]["Reg. L1 Norm"],
#             mode="lines+markers",
#             name=f"{layer} reg.",
#             line=dict(color=inclusive_colors[i], width=2, dash="dash"),
#         )
#     )

# fig.update_layout(
#     title="Kernel L1 Norm and Reg. L1 Norm",
#     xaxis_title="Step",
#     yaxis_title="L1 Norm",
#     legend_title="Layer",
# )

# fig.update_traces(mode="lines+markers")

# # white background and grey grid lines
# fig.update_layout(
#     plot_bgcolor="white",
#     paper_bgcolor="white",
#     xaxis=dict(showgrid=True, gridcolor="lightgrey"),
#     yaxis=dict(showgrid=True, gridcolor="lightgrey"),)

# # log scale
# fig.update_yaxes(type="log")

# fig.write_image(plot_dir + "kernel_reg_balance_separate_low_res.png", width=800, height=600)
# fig.write_image(plot_dir + "kernel_reg_balance_separate.png", width=800, height=600, scale=2)
# fig.write_image(plot_dir + "kernel_reg_balance_separate.pdf", width=800, height=600, scale=2)
