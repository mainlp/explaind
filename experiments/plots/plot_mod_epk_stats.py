import torch
from explaind.modulo_model.model import SingleLayerTransformerClassifier
import json
import plotly.graph_objects as go

device = "cuda" if torch.cuda.is_available() else "cpu"

inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", #"#F0E442", 
                    "#0072B2", "#D55E00", "#CC79A7"]

experiment_path = "results/modulo_val_results_0.01_w/"

train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

config = json.load(open(experiment_path + "config.json", 'r'))

# load model
base_model = SingleLayerTransformerClassifier(config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])

kernel_abs_sums = torch.load(experiment_path + "kernel_abs_sums.pt", map_location=device)

kernel_abs_sums_accumulated = {}
for layer_name, kernel in kernel_abs_sums.items():
    kernel_abs_sums_accumulated[layer_name] = kernel.sum(dim=0).sum(dim=-1).sum(dim=-1)

kernel_abs_sums_accumulated["linear2"] = kernel_abs_sums_accumulated["linear2.weight"] + kernel_abs_sums_accumulated["linear2.bias"]
del kernel_abs_sums_accumulated["linear2.weight"]
del kernel_abs_sums_accumulated["linear2.bias"]
kernel_abs_sums_accumulated["linear1"] = kernel_abs_sums_accumulated["linear1.weight"] + kernel_abs_sums_accumulated["linear1.bias"]
del kernel_abs_sums_accumulated["linear1.weight"]
del kernel_abs_sums_accumulated["linear1.bias"]
kernel_abs_sums_accumulated["att. encoders"] = kernel_abs_sums_accumulated["head1_encoder.weight"] + kernel_abs_sums_accumulated["head2_encoder.weight"] + kernel_abs_sums_accumulated["head3_encoder.weight"] + kernel_abs_sums_accumulated["head4_encoder.weight"]
del kernel_abs_sums_accumulated["head1_encoder.weight"]
del kernel_abs_sums_accumulated["head2_encoder.weight"]
del kernel_abs_sums_accumulated["head3_encoder.weight"]
del kernel_abs_sums_accumulated["head4_encoder.weight"]
kernel_abs_sums_accumulated["att. decoders"] = kernel_abs_sums_accumulated["head1_decoder.weight"] + kernel_abs_sums_accumulated["head2_decoder.weight"] + kernel_abs_sums_accumulated["head3_decoder.weight"] + kernel_abs_sums_accumulated["head4_decoder.weight"]
del kernel_abs_sums_accumulated["head1_decoder.weight"]
del kernel_abs_sums_accumulated["head2_decoder.weight"]
del kernel_abs_sums_accumulated["head3_decoder.weight"]
del kernel_abs_sums_accumulated["head4_decoder.weight"]
del kernel_abs_sums_accumulated["positional_encoding.pe"]
kernel_abs_sums_accumulated["embedding"] = kernel_abs_sums_accumulated["input_emb.weight"]
del kernel_abs_sums_accumulated["input_emb.weight"]

kernel_abs_sums_accumulated["decoder"] = kernel_abs_sums_accumulated["decoder.weight"]

print(kernel_abs_sums_accumulated.keys())

order = ['embedding', 'att. encoders', 'att. decoders', 'linear1', 'linear2', 'decoder']

# plot kernel_abs_sums_accumulated
abs_sums_fig = go.Figure()
for layer_name in order:
    kernel = kernel_abs_sums_accumulated[layer_name].cpu().numpy()
    # print change of sum
    kernel[1:] = kernel[1:] - kernel[:-1]
    # apply smoothing
    for k in range(2, len(kernel)-2):
        kernel[k] = (kernel[k-2] + kernel[k-1] + kernel[k] + kernel[k+1] + kernel[k+2]) / 5
    print(f"Layer {layer_name}: {kernel.shape}")
    abs_sums_fig.add_trace(go.Scatter(x=list(range(len(kernel))),
                                      y=kernel, 
                                      mode='lines', 
                                      name=layer_name, 
                                      line=dict(color=inclusive_colors[order.index(layer_name)])))
    
abs_sums_fig.update_layout(title='', xaxis_title='<b>Step', yaxis_title='$\Large\Psi_s(\Theta_{layer})$')
# make font size bigger
abs_sums_fig.update_layout(font=dict(size=22))
abs_sums_fig.update_xaxes(title_font=dict(size=22))
# move legend into graph
# abs_sums_fig.update_layout(legend=dict(
#     orientation="v",
#     yanchor="bottom",
#     y=0.74,
#     xanchor="left",
#     x=0.5,
# ))
# remove legend
abs_sums_fig.update_layout(showlegend=False)
# make backround white, lines grey
abs_sums_fig.update_layout(
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
# make lines thicker
abs_sums_fig.update_traces(line=dict(width=1.2))
abs_sums_fig.update_layout(legend=dict(itemsizing='constant'))
# don't show title
abs_sums_fig.update_layout(showtitle=False)
# #make bold
abs_sums_fig.update_layout(legend_font=dict(size=18))
#smller tick font
abs_sums_fig.update_xaxes(tickfont=dict(size=12))
abs_sums_fig.update_yaxes(tickfont=dict(size=12))
abs_sums_fig.write_image(experiment_path + "plots/kernel_abs_sums_accumulated.png", format="png", width=400, height=340, scale=2)
abs_sums_fig.write_image(experiment_path + "plots/kernel_abs_sums_accumulated.pdf", format="pdf", width=400, height=340)


##########################################################################################################################

step_l1_norms = torch.load(experiment_path + "Step_l1_norm.pt", map_location="cpu")
reg_term_l1_norm = torch.load(experiment_path + "regularization_term_l1_norm.pt", map_location="cpu")

# plot both (they're lists of len 1939)
step_l1_norms = step_l1_norms.cpu().numpy()
reg_term_l1_norm = reg_term_l1_norm.cpu().numpy()
step_plot = go.Figure()
step_plot.add_trace(go.Scatter(x=list(range(len(step_l1_norms))), y=step_l1_norms, mode='lines', name='Kernel', line=dict(color=inclusive_colors[0])))
step_plot.add_trace(go.Scatter(x=list(range(len(reg_term_l1_norm))), y=reg_term_l1_norm, mode='lines', name='Regularization', line=dict(color=inclusive_colors[1])))

step_plot.update_layout(title='', xaxis_title='<b>Step', yaxis_title='$\Large\Psi_s(\Theta_{model})$')


# background white, lines grey
step_plot.update_layout(
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

# move legend into graph
step_plot.update_layout(legend=dict(
    orientation="v",
    yanchor="top",
    y=1.0,
    xanchor="left",
    x=0.01,
))
# make lines thicker
step_plot.update_traces(line=dict(width=1.2))
step_plot.update_layout(legend=dict(itemsizing='constant'))

# make font size bigger
step_plot.update_layout(font=dict(size=22))
step_plot.update_xaxes(title_font=dict(size=22))
step_plot.update_yaxes(title_font=dict(size=22))
step_plot.update_layout(legend_font=dict(size=16))
# smaller tick font
step_plot.update_xaxes(tickfont=dict(size=12))
step_plot.update_yaxes(tickfont=dict(size=12))

step_plot.write_image(experiment_path + "plots/l1_norms.png", format="png", width=400, height=340, scale=2)
step_plot.write_image(experiment_path + "plots/l1_norms.pdf", format="pdf", width=400, height=340)

##############################################################################################

