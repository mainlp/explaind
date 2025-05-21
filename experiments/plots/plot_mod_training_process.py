import torch
import json
import pandas as pd
import plotly.graph_objects as go


device = "cuda" if torch.cuda.is_available() else "cpu"

inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

experiment_path = "results/modulo_val_results/"

train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

config = json.load(open(experiment_path + "config.json", 'r'))

train_stats_path = "results/modulo_val_results_0.01_w/plots/train_stats"
train_acc = pd.read_csv(train_stats_path + "/accuracy_train.csv")
train_loss = pd.read_csv(train_stats_path + "/loss_train.csv")
val_acc = pd.read_csv(train_stats_path + "/accuracy_val.csv")
val_loss = pd.read_csv(train_stats_path + "/loss_val.csv")
reg = pd.read_csv(train_stats_path + "/reg_train.csv")

# combine into one df
df = {
    "step": train_acc["Step"],
    "train_acc": train_acc["Value"],
    "train_loss": train_loss["Value"],
    "val_acc": val_acc["Value"],
    "val_loss": val_loss["Value"],
    "reg": reg["Value"],
}
df = pd.DataFrame(df)

# make plots of loss and acc and reg
loss_fig = go.Figure()
loss_fig.add_trace(go.Scatter(x=df["step"], y=df["train_loss"], mode='lines', name='Train', line=dict(color=inclusive_colors[0])))
loss_fig.add_trace(go.Scatter(x=df["step"], y=df["val_loss"], mode='lines', name='Test', line=dict(color=inclusive_colors[1])))
loss_fig.update_layout(title='', xaxis_title='<b>Step', yaxis_title='<b>Loss')

# make font size bigger
loss_fig.update_layout(font=dict(size=22))
loss_fig.update_xaxes(title_font=dict(size=22))
loss_fig.update_yaxes(title_font=dict(size=22))
# move legend into graph
loss_fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=0.65,
    xanchor="right",
    x=1.0
))
#background white, lines grey
loss_fig.update_layout(
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
#make lines thicker
loss_fig.update_traces(line=dict(width=2))
loss_fig.update_layout(legend=dict(itemsizing='constant'))

loss_fig.update_layout(font=dict(size=22))
loss_fig.update_xaxes(title_font=dict(size=22))
loss_fig.update_yaxes(title_font=dict(size=22))
loss_fig.update_layout(legend_font=dict(size=16))
# smaller tick font
loss_fig.update_xaxes(tickfont=dict(size=12))
loss_fig.update_yaxes(tickfont=dict(size=12))

loss_fig.write_image(train_stats_path + "/loss_train_val.png", format="png", width=400, height=340)
loss_fig.write_image(train_stats_path + "/loss_train_val.pdf", format="pdf", width=400, height=340, scale=2)
loss_fig.write_image(train_stats_path + "/loss_train_val.pdf", format="pdf", width=400, height=340)

# do the same for acc
acc_fig = go.Figure()
acc_fig.add_trace(go.Scatter(x=df["step"], y=df["train_acc"], mode='lines', name='Train', line=dict(color=inclusive_colors[0])))
acc_fig.add_trace(go.Scatter(x=df["step"], y=df["val_acc"], mode='lines', name='Test', line=dict(color=inclusive_colors[1])))
acc_fig.update_layout(title="", xaxis_title='<b>Step', yaxis_title='<b>Accuracy')
# make font size bigger
acc_fig.update_layout(font=dict(size=22))
acc_fig.update_xaxes(title_font=dict(size=22))
# hide legend
acc_fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=0.05,
    xanchor="right",
    x=0.97,
))
acc_fig.update_layout(showlegend=False)
# background white, lines grey
acc_fig.update_layout(
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

#make lines thicker
acc_fig.update_traces(line=dict(width=2))

acc_fig.update_layout(font=dict(size=22))
acc_fig.update_xaxes(title_font=dict(size=22))
acc_fig.update_yaxes(title_font=dict(size=22))
acc_fig.update_layout(legend_font=dict(size=16))
# smaller tick font
acc_fig.update_xaxes(tickfont=dict(size=12))
acc_fig.update_yaxes(tickfont=dict(size=12))

acc_fig.write_image(train_stats_path + "/acc_train_val.png", format="png", width=400, height=340)
acc_fig.write_image(train_stats_path + "/acc_train_val.pdf", format="pdf", width=400, height=340)
acc_fig.write_html(train_stats_path + "/acc_train_val.html")

# make plot of reg
reg_fig = go.Figure()
reg_fig.add_trace(go.Scatter(x=df["step"], y=df["reg"], mode='lines', name='', line=dict(color=inclusive_colors[0])))
reg_fig.update_layout(title='', xaxis_title='<b>Step', yaxis_title='<b>Weight Decay')

# make font size bigger
reg_fig.update_layout(font=dict(size=22))
reg_fig.update_xaxes(title_font=dict(size=22))
# do not show legend
reg_fig.update_layout(showlegend=False)
# set y max to 130
reg_fig.update_yaxes(range=[98, 130])
# background white, lines grey
reg_fig.update_layout(
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
reg_fig.update_traces(line=dict(width=2))

reg_fig.update_layout(font=dict(size=22))
reg_fig.update_xaxes(title_font=dict(size=22))
reg_fig.update_yaxes(title_font=dict(size=22))
reg_fig.update_layout(legend_font=dict(size=16))
# smaller tick font
reg_fig.update_xaxes(tickfont=dict(size=12))
reg_fig.update_yaxes(tickfont=dict(size=12))


reg_fig.write_image(train_stats_path + "/reg_train.png", format="png", width=400, height=340)
reg_fig.write_image(train_stats_path + "/reg_train.pdf", format="pdf", width=400, height=340)
reg_fig.write_html(train_stats_path + "/reg_train.html")