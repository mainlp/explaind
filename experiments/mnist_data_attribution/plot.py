import pandas as pd
import plotly.express as px
import json

results_path = "results/mnist_lds_ground_truth/lds_all.json"

with open(results_path, "r") as f:
    lds_all = json.load(f)

alphas = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]
data_dict = {"alpha": [], "LDS": [], "method": []}
for method, scores in lds_all.items():
    if method in ["explaind_0", "explaind_no_init", "explaind_proba"]:
       continue
    if method == "explaind_1":
        method = "explaind"
    for alpha, score in zip(alphas, scores):
        data_dict["alpha"].append(alpha)
        data_dict["LDS"].append(score)
        data_dict["method"].append(method)

df = pd.DataFrame(data_dict)
# plot it
fig = px.line(df, x="alpha", y="LDS", color="method", markers=True, title="LDS scores for different methods and sparsities")
fig.update_layout(xaxis_title="<b>Alpha</b>", yaxis_title="<b>LDS</b>")

# times new roman
fig.update_layout(font=dict(family="Times New Roman", size=18))
# choose whit background thme
fig.update_layout(template="plotly_white")

# make ticks at 0.5, 0.7, 0.9, 0.95, 0.99
fig.update_xaxes(tickvals=alphas[:-1], ticktext=[str(a) for a in alphas][:-1])

# put legend in plot top right
fig.update_layout(legend=dict(x=0.7, y=0.9))
# update legend title
fig.update_layout(legend_title_text="Method")

inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7"]
# update lines with colors
for i, color in enumerate(inclusive_colors):
    if i >= len(fig.data):
        break
    fig.data[i].line.color = color
    fig.data[i].marker.color = color

# save imng
fig.write_image(results_path + "lds_plot.png", scale=3)
fig.write_image(results_path + "lds_plot.pdf", scale=3)
