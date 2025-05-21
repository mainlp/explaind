import json
import plotly.graph_objects as go
import os
import pandas as pd


device = "cpu"# "cuda" if torch.cuda.is_available() else "cpu"

experiment_path = "results/modulo_val_results_wreg/plots/inject_pipeline_train/"


# load pipeline results
# runs1700 = []
# runs1939 = []

# for i in range(0, 1801, 50):
#     checkpoints_dir = experiment_path + f"pipeline_train_{i}_1700/"

#     for f in os.listdir(checkpoints_dir):
#         if f.endswith(".json"):
#             with open(checkpoints_dir + f, 'r') as json_file:
#                 data = json.load(json_file)
#                 data["step"] = i
#                 runs1700.append(data)

# for i in range(0, 1801, 50):
#     checkpoints_dir = experiment_path + f"pipeline_train_{i}_-1/"
#     for f in os.listdir(checkpoints_dir):
#         if f.endswith(".json"):
#             with open(checkpoints_dir + f, 'r') as json_file:
#                 data = json.load(json_file)
#                 data["step"] = i
#                 runs1939.append(data)

# df = []

# for entry in runs1700:
#     df.append({"step": entry["step"], "pipeline step": "1700", "num. epochs": entry["final_epoch"], "test acc.": entry["val_accuracy"]})

# for entry in runs1939:
#     df.append({"step": entry["step"], "pipeline step": "final", "num. epochs": entry["final_epoch"], "test acc.": entry["val_accuracy"]})

# df = pd.DataFrame(df)
# df["Step"] = df["step"].astype(int)
# df["Num. epochs"] = df["num. epochs"].astype(int)
# df["Test acc."] = df["test acc."].astype(float)
# df["Pipeline step"] = df["pipeline step"].astype(str)
# df = df.sort_values(by=["step", "pipeline step"])

# # get std and mean
# df_group = df.groupby(["Step", "Pipeline step"]).agg({"Num. epochs": ["mean", "std"]}).reset_index()

# df_group.columns = ["Step", "Pipeline step", "Num. epochs", "Num. epochs std"]

# # plot lines for 1700 and final
# fig = go.Figure()

# inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# df_1700 = df_group[df_group["Pipeline step"] == "1700"]
# fig.add_trace(go.Scatter(
#     x=df_1700["Step"],
#     y=df_1700["Num. epochs"],
#     mode='lines',
#     name='Step 1700',
#     line=dict(color=inclusive_colors[6])
# ))

# # add error lines
# fig.add_trace(go.Scatter(
#     x=df_1700["Step"],
#     y=df_1700["Num. epochs"] + df_1700["Num. epochs std"],
#     mode='lines',
#     line=dict(color=inclusive_colors[6], dash='dot'),
#     showlegend=False,
#     name='1700 error',
#     hoverinfo='none',
# ))
# fig.add_trace(go.Scatter(
#     x=df_1700["Step"],
#     y=df_1700["Num. epochs"] - df_1700["Num. epochs std"],
#     mode='lines',
#     line=dict(color=inclusive_colors[6], dash='dot'),
#     showlegend=False,
#     name='1700 error',
#     hoverinfo='none',
# ))

# df_final = df_group[df_group["Pipeline step"] == "final"]
# fig.add_trace(go.Scatter(
#     x=df_final["Step"],
#     y=df_final["Num. epochs"],
#     mode='lines',
#     name='Step 1939',
#     line=dict(color=inclusive_colors[2])
# ))
# # add error lines
# fig.add_trace(go.Scatter(
#     x=df_final["Step"],
#     y=df_final["Num. epochs"] + df_final["Num. epochs std"],
#     mode='lines',
#     line=dict(color=inclusive_colors[2], dash='dot'),
#     showlegend=False,
#     name='1939 error',
#     hoverinfo='none',
# ))
# fig.add_trace(go.Scatter(
#     x=df_final["Step"],
#     y=df_final["Num. epochs"] - df_final["Num. epochs std"],
#     mode='lines',
#     line=dict(color=inclusive_colors[2], dash='dot'),
#     showlegend=False,
#     name='1939 error',
#     hoverinfo='none',
# ))

# fig.add_trace(go.Scatter(
#     x=list(range(0, 1801, 50)),
#     y=list(range(1939, 138, -50)),
#     mode='markers',
#     name='Original run',
#     marker=dict(color=inclusive_colors[0], size=3),
# ))

# # larger font
# fig.update_layout(
#     title="Number of epochs until convergence",
#     xaxis_title="Step",
#     yaxis_title="Number of epochs",
#     legend=dict(
#         orientation="v",
#         yanchor="bottom",
#         y=0.55,
#         xanchor="right",
#         x=0.9
#     ),
# )

# # white background, grey lines
# fig.update_layout(
#     plot_bgcolor='white',
#     paper_bgcolor='white',
#     xaxis=dict(showgrid=True, gridcolor='lightgrey'),
#     yaxis=dict(showgrid=True, gridcolor='lightgrey'),
# )

# # have axes
# fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
# fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
# # have ticks
# fig.update_xaxes(showticklabels=True, ticks="outside", tickwidth=1, tickcolor='black')
# fig.update_yaxes(showticklabels=True, ticks="outside", tickwidth=1, tickcolor='black')

# # increase font sizes
# fig.update_xaxes(title_font=dict(size=28), tickfont=dict(size=20))
# fig.update_yaxes(title_font=dict(size=28), tickfont=dict(size=20))
# fig.update_layout(title_font=dict(size=28))
# fig.update_layout(legend_font=dict(size=26))

# # decouple line wdiths from legend
# fig.update_traces(line=dict(width=3))
# fig.update_layout(legend=dict(itemsizing='constant'))

# fig.write_image(experiment_path + "num_epochs_until_convergence.png", width=800, height=600)
# fig.write_image(experiment_path + "num_epochs_until_convergence.pdf", width=800, height=600)
# fig.write_html(experiment_path + "num_epochs_until_convergence.html")

def load_pipeline_results():
    # load pipeline results
    df = []

    for i in range(1939, 300, -100):
        checkpoints_dir = experiment_path + f"pipeline_train_rand_{i}/"

        for f in os.listdir(checkpoints_dir):
            if f.endswith(".json"):
                with open(checkpoints_dir + f, 'r') as json_file:
                    data = json.load(json_file)
                    data["step"] = i
                    for j, acc in enumerate(data["val_accuracies"]):
                        df.append({
                            "step": i,
                            "layers": "att.+lin1",
                            "num. epochs": data["final_epoch"],
                            "test acc.": acc,
                            "train acc.": data["train_accuracies"][j],
                            "epoch": j
                        })

        checkpoints_dir = experiment_path + f"att_train_rand_{i}/"
        for f in os.listdir(checkpoints_dir):
            if f.endswith(".json"):
                with open(checkpoints_dir + f, 'r') as json_file:
                    data = json.load(json_file)
                    data["step"] = i
                    for j, acc in enumerate(data["val_accuracies"]):
                        df.append({
                            "step": i,
                            "layers": "att.",
                            "num. epochs": data["final_epoch"],
                            "test acc.": acc,
                            "train acc.": data["train_accuracies"][j],
                            "epoch": j
                        })

        checkpoints_dir = experiment_path + f"outer_train_rand_{i}/"
        for f in os.listdir(checkpoints_dir):
            if f.endswith(".json"):
                with open(checkpoints_dir + f, 'r') as json_file:
                    data = json.load(json_file)
                    data["step"] = i
                    for j, acc in enumerate(data["val_accuracies"]):
                        df.append({
                            "step": i,
                            "layers": "emb.+lin2+dec.",
                            "num. epochs": data["final_epoch"],
                            "test acc.": acc,
                            "train acc.": data["train_accuracies"][j],
                            "epoch": j
                        })

    df = pd.DataFrame(df)

    # sort values by step and layers
    df = df.sort_values(by=["step", "layers", "epoch"])

    return df


def plot_results_for(df, steps=[1939], configs=["att.+lin1"], legend=False):
    inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    fig = go.Figure()

    for config in configs:
        for step in steps:
            df_ = df[(df["layers"] == config) & (df["step"] == step)]

            # group by epoch and get mean
            df_ = df_.groupby(["epoch"]).agg({"test acc.": ["mean", "std"],
                                            "train acc.": ["mean", "std"]}).reset_index()

            # make x acis only go to 500 (and cut remaining points)
            df_ = df_[df_["epoch"] <= 500]

            # fig.add_trace(go.Scatter(
            #     x=df_["epoch"],
            #     y=df_["test acc."],
            #     mode='markers',
            #     name=f"{config} {step}",
            #     line=dict(color=inclusive_colors[configs.index(config)])
            # ))

            # add errorbars
            fig.add_trace(go.Scatter(
                x=df_["epoch"],
                y=df_["test acc."]["mean"],
                mode='lines',
                name="test",
                line=dict(color=inclusive_colors[configs.index(config)]),
            ))

            df_["upper error"] = df_["test acc."]["mean"] + df_["test acc."]["std"]
            df_["lower error"] = df_["test acc."]["mean"] - df_["test acc."]["std"]

            fig.add_trace(go.Scatter(
                x=df_["epoch"],
                y=df_["upper error"],
                mode='lines',
                line=dict(color=inclusive_colors[configs.index(config)], dash='2px'),
                showlegend=False,
            ))

            fig.add_trace(go.Scatter(
                x=df_["epoch"],
                y=df_["lower error"],
                mode='lines',
                line=dict(color=inclusive_colors[configs.index(config)], dash='2px'),
                showlegend=False,
            ))

            fig.add_trace(go.Scatter(
                x=df_["epoch"],
                y=df_["train acc."]["mean"],
                mode='lines',
                name="train",
                line=dict(color=inclusive_colors[configs.index(config)+1]),
            ))

            df_["upper error"] = df_["train acc."]["mean"] + df_["train acc."]["std"]
            df_["lower error"] = df_["train acc."]["mean"] - df_["train acc."]["std"]

            fig.add_trace(go.Scatter(
                x=df_["epoch"],
                y=df_["upper error"],
                mode='lines',
                line=dict(color=inclusive_colors[configs.index(config)+1], dash='2px'),
                showlegend=False,
            ))

            fig.add_trace(go.Scatter(
                x=df_["epoch"],
                y=df_["lower error"],
                mode='lines',
                line=dict(color=inclusive_colors[configs.index(config)+1], dash='2px'),
                showlegend=False,
            ))



    # larger font
    fig.update_layout(
        title="",
        xaxis_title="<b>Step",
        yaxis_title="<b>Accuracy",
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.2,
            xanchor="right",
            x=0.9
        ),
    )


    # white background, grey lines
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    )

    # have axes
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
    # have ticks
    fig.update_xaxes(showticklabels=True, ticks="outside", tickwidth=1, tickcolor='black')
    fig.update_yaxes(showticklabels=True, ticks="outside", tickwidth=1, tickcolor='black')

    # increase font sizes
    fig.update_layout(legend=dict(itemsizing='constant'))

    # make font size bigger
    fig.update_layout(font=dict(size=22))
    fig.update_xaxes(title_font=dict(size=22))
    fig.update_yaxes(title_font=dict(size=22))
    fig.update_layout(legend_font=dict(size=16))
    # smaller tick font
    fig.update_xaxes(tickfont=dict(size=12))
    fig.update_yaxes(tickfont=dict(size=12))

    # decouple line wdiths from legend
    fig.update_traces(line=dict(width=1.2))
    fig.update_layout(legend=dict(itemsizing='constant'))

    if not legend:
        fig.update_layout(showlegend=False)

    return fig


def save_fig(fig, name):
    fig.write_image(experiment_path + name + ".png", width=400, height=280)
    fig.write_image(experiment_path + name + ".pdf", width=400, height=280)

    fig.write_html(experiment_path + name + ".html")


if __name__ == "__main__":
    df = load_pipeline_results()

    # plot results for 1700 and final
    fig = plot_results_for(df, steps=[1939], configs=["att.+lin1"], legend=True)
    save_fig(fig, "pipeline_train_1939_att_lin1")

    fig = plot_results_for(df, steps=[1939], configs=["att."], legend=True)
    save_fig(fig, "pipeline_train_1939_att")

    fig = plot_results_for(df, steps=[1939], configs=["emb.+lin2+dec."], legend=True)
    save_fig(fig, "pipeline_train_1939_emb_lin2_dec")

    fig = plot_results_for(df, steps=[1839], configs=["att.+lin1"])
    save_fig(fig, "pipeline_train_1839_att_lin1")

    fig = plot_results_for(df, steps=[1839], configs=["att."])
    save_fig(fig, "pipeline_train_1839_att")

    fig = plot_results_for(df, steps=[1839], configs=["emb.+lin2+dec."])
    save_fig(fig, "pipeline_train_1839_emb_lin2_dec")

    fig = plot_results_for(df, steps=[1739], configs=["att.+lin1"])
    save_fig(fig, "pipeline_train_1739_att_lin1")
    fig = plot_results_for(df, steps=[1739], configs=["att."])
    save_fig(fig, "pipeline_train_1739_att")

    fig = plot_results_for(df, steps=[1739], configs=["emb.+lin2+dec."])
    save_fig(fig, "pipeline_train_1739_emb_lin2_dec")

    fig = plot_results_for(df, steps=[1539], configs=["att.+lin1"])
    save_fig(fig, "pipeline_train_1539_att_lin1")

    fig = plot_results_for(df, steps=[1539], configs=["att."])
    save_fig(fig, "pipeline_train_1539_att")

    fig = plot_results_for(df, steps=[1539], configs=["emb.+lin2+dec."])
    save_fig(fig, "pipeline_train_1539_emb_lin2_dec")

    fig = plot_results_for(df, steps=[1139], configs=["att.+lin1"])
    save_fig(fig, "pipeline_train_1139_att_lin1")

    fig = plot_results_for(df, steps=[1139], configs=["att."])
    save_fig(fig, "pipeline_train_1139_att")

    fig = plot_results_for(df, steps=[1139], configs=["emb.+lin2+dec."])
    save_fig(fig, "pipeline_train_1139_emb_lin2_dec")

    fig = plot_results_for(df, steps=[539], configs=["att.+lin1"])
    save_fig(fig, "pipeline_train_539_att_lin1")

    fig = plot_results_for(df, steps=[539], configs=["att."])
    save_fig(fig, "pipeline_train_539_att")

    fig = plot_results_for(df, steps=[539], configs=["emb.+lin2+dec."])
    save_fig(fig, "pipeline_train_539_emb_lin2_dec")

    fig = plot_results_for(df, steps=[439], configs=["att.+lin1"])
    save_fig(fig, "pipeline_train_439_att_lin1")

    fig = plot_results_for(df, steps=[439], configs=["att."])
    save_fig(fig, "pipeline_train_439_att")

    fig = plot_results_for(df, steps=[439], configs=["emb.+lin2+dec."])
    save_fig(fig, "pipeline_train_439_emb_lin2_dec")

    fig = plot_results_for(df, steps=[339], configs=["emb.+lin2+dec."])
    save_fig(fig, "pipeline_train_339_emb_lin2_dec")

    fig = plot_results_for(df, steps=[339], configs=["att.+lin1"])
    save_fig(fig, "pipeline_train_339_att_lin1")

    fig = plot_results_for(df, steps=[339], configs=["att."])
    save_fig(fig, "pipeline_train_339_att")
