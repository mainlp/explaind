import torch
from explaind.modulo_model.model import SingleLayerTransformerClassifier
from explaind.model_paths import ModelPath
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

experiment_path = "results/modulo_val_results/"

train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

config = json.load(open(experiment_path + "config.json", 'r'))

# load model
base_model = SingleLayerTransformerClassifier(config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])

# load model checkpoints (to be found in model_path.checkpoints which is a list of state_dicts)
model_path = ModelPath(base_model,                  
                       device=device, 
                       checkpoint_path=experiment_path + "model_checkpoint.pt",
                       features_to_cpu=False,
                       parallelize_over_targets=True)

def inject_previous_step(model_path, step, ignore_layers=["decoder.weight", "input_emb.weight", "linear2.weight", "linear2.bias"]):
    """
    Injects the previous step's parameters into the current step's parameters.
    """

    base_model = deepcopy(model_path.model)
    final_params = deepcopy(model_path.checkpoints[-1])

    # get step parameters
    step_params = deepcopy(model_path.checkpoints[step])

    injected_params = {}

    for k, v in final_params.items():
        # if k in step_params:
        #     injected_params[k] = step_params[k]
        # else:
        #     injected_params[k] = v
        if k in step_params and k not in ignore_layers:
            injected_params[k] = step_params[k]
        else:
            injected_params[k] = v

    # set the model's parameters to the injected parameters
    base_model.load_state_dict(injected_params)

    base_model.eval()

    step_model = deepcopy(model_path.model)
    step_model.load_state_dict(step_params)

    return base_model, step_model


def eval_injected_model(base_model, inj_model, step_model, val_loader):
    inj_logits = []
    pred_logits = [] 
    step_logits = []   
    pred_correct = 0
    inj_pred_acc = 0
    step_pred_acc = 0
    total = 0

    # compare predictions
    for i, (x, y) in tqdm(enumerate(val_loader)):
        x = x.to(device)
        y = y.to(device)

        # get predictions
        with torch.no_grad():
            pred_log = base_model(x)
            inj_pred_log = inj_model(x)
            step_pred_log = step_model(x)

        pred_logits.append(pred_log)
        inj_logits.append(inj_pred_log)
        step_logits.append(step_pred_log)

        # eval
        pred = torch.argmax(pred_log, dim=1)
        inj_pred = torch.argmax(inj_pred_log, dim=1)
        step_pred = torch.argmax(step_pred_log, dim=1)

        pred_correct += (pred == y).sum()
        inj_pred_acc += (inj_pred == y).sum()
        step_pred_acc += (step_pred == y).sum()
        total += y.size(0)

    pred_acc = pred_correct / total
    inj_acc = inj_pred_acc / total
    step_acc = step_pred_acc / total
    print(f"Pred acc: {pred_acc}")
    print(f"Inj acc: {inj_acc}")
    print(f"Step acc: {step_acc}")

    combined_logits = torch.cat(pred_logits, dim=0)
    inj_combined_logits = torch.cat(inj_logits, dim=0)
    step_combined_logits = torch.cat(step_logits, dim=0)

    print(f"Combined logits shape: {combined_logits.shape}")

    kl_div = torch.nn.KLDivLoss(reduction='mean', log_target=True)
    kl_div_inj = kl_div(torch.log_softmax(combined_logits, dim=1), torch.log_softmax(inj_combined_logits, dim=1))
    kl_div_step = kl_div(torch.log_softmax(combined_logits, dim=1), torch.log_softmax(step_combined_logits, dim=1))

    print(f"KL div inj: {kl_div_inj}")
    print(f"KL div step: {kl_div_step}")

    return pred_acc.cpu().item(), inj_acc.cpu().item(), step_acc.cpu().item(), kl_div_inj.item(), kl_div_step.item(), inj_combined_logits, step_combined_logits


base_model.load_state_dict(model_path.checkpoints[-1])

inj_logits = []
step_logits = []

pred_accs = []

emb_dec_inj_accs = []

step_accs = []

kl_div_injs = []
kl_div_steps = []

dec_inj_accs = []
kl_divs_dec_only = []

all_inj_accs = []
kl_div_all = []

emblin_inj = []
embdlin_kl = []

embdec_inj = []
kl_div_embdec = []

for i in tqdm(range(0, 1940, 10)):
    
    inj_model, step_model = inject_previous_step(model_path, i)
    result = eval_injected_model(base_model, inj_model, step_model, val_loader)

    inj_logits.append(result[5])
    step_logits.append(result[6])

    pred_accs.append(result[0])
    emb_dec_inj_accs.append(result[1])
    step_accs.append(result[2])
    kl_div_injs.append(result[3])
    kl_div_steps.append(result[4])

    inj_model, step_model = inject_previous_step(model_path, i, ignore_layers=["decoder.weight", "linear2.weight", "linear2.bias"])

    result_dec_only = eval_injected_model(base_model, inj_model, step_model, val_loader)
    dec_inj_accs.append(result_dec_only[1])
    kl_divs_dec_only.append(result_dec_only[3])

    inj_model, step_model = inject_previous_step(model_path, i, ignore_layers=["input_emb.weight", "input_emb.weight", "linear1.weight", "linea1.bias", "linear2.weight", "linear2.bias"])

    result_all = eval_injected_model(base_model, inj_model, step_model, val_loader)
    all_inj_accs.append(result_all[1])
    kl_div_all.append(result_all[3])

    inj_model, step_model = inject_previous_step(model_path, i, ignore_layers=["input_emb.weight", "input_emb.weight", "linear2.weight"])

    result_emb_lin = eval_injected_model(base_model, inj_model, step_model, val_loader)
    emblin_inj.append(result_emb_lin[1])
    embdlin_kl.append(result_emb_lin[3])

    inj_model, step_model = inject_previous_step(model_path, i, ignore_layers=["input_emb.weight", "input_emb.weight", "decoder.weight"])
    result_emb_dec = eval_injected_model(base_model, inj_model, step_model, val_loader)
    embdec_inj.append(result_emb_dec[1])
    kl_div_embdec.append(result_emb_dec[3])



diffs_inj = np.array(emb_dec_inj_accs) - np.array(step_accs)
diffs_dec_only = np.array(dec_inj_accs) - np.array(step_accs)
diffs_all = np.array(all_inj_accs) - np.array(step_accs)
diffs_emb_lin = np.array(emblin_inj) - np.array(step_accs)
diffs_emb_dec = np.array(embdec_inj) - np.array(step_accs)

inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# plot the results
fig = go.Figure()

# fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=step_accs, mode='lines', name='Actual Model', line=dict(color=inclusive_colors[0])))

# fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=emb_dec_inj_accs, mode='lines', name='Emb.+Lin2+Dec.', line=dict(color=inclusive_colors[6])))
fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=diffs_inj, mode='lines', name='Emb.+Lin2+Dec.', line=dict(color=inclusive_colors[6])))

# fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=dec_inj_accs, mode='lines', name='Decoder', line=dict(color=inclusive_colors[1])))
fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=diffs_dec_only, mode='lines', name='Decoder', line=dict(color=inclusive_colors[1])))

#fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=all_inj_accs, mode='lines', name='All but att.', line=dict(color=inclusive_colors[2])))
fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=diffs_all, mode='lines', name='All but att.', line=dict(color=inclusive_colors[2])))

# fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=emblin_inj, mode='lines', name='Emb.+Lin2', line=dict(color=inclusive_colors[3])))
fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=diffs_emb_lin, mode='lines', name='Emb.', line=dict(color=inclusive_colors[3])))

#fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=embdec_inj, mode='lines', name='Emb.+Dec.', line=dict(color=inclusive_colors[4])))
fig.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=diffs_emb_dec, mode='lines', name='Emb.+Dec.', line=dict(color=inclusive_colors[4])))

os.makedirs(experiment_path + "plots/injection_exp/", exist_ok=True)

# exclude dotted lines from legend
fig.for_each_trace(lambda t: t.update(showlegend=False) if t.name == "" else ())

fig.update_layout(
    xaxis_title="<b>Step",
    yaxis_title="<b>Diff. in Accuracy",
    legend_title="")

# increase font size of x and y axis labels
# fig.update_xaxes(title_font=dict(size=28))
# fig.update_yaxes(title_font=dict(size=28))
# fig.update_xaxes(tickfont=dict(size=24))
# fig.update_yaxes(tickfont=dict(size=24))
# # increase font size of legend
# fig.update_layout(legend_font=dict(size=22))
# # increase font size of title
# fig.update_layout(title_font=dict(size=26))

fig.update_traces(line=dict(width=1.2))
fig.update_layout(legend=dict(itemsizing='constant'))

# make font size bigger
fig.update_layout(font=dict(size=22))
fig.update_xaxes(title_font=dict(size=25))
fig.update_yaxes(title_font=dict(size=25))
fig.update_layout(legend_font=dict(size=16))
# smaller tick font
fig.update_xaxes(tickfont=dict(size=12))
fig.update_yaxes(tickfont=dict(size=12))

# move legend into graph
fig.update_layout(legend=dict(
    orientation="h",
    y=1.64,
    x=-0.0
))

# background white, grey lines
fig.update_layout(plot_bgcolor='white')
fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
fig.update_yaxes(showgrid=True, gridcolor='lightgrey')

# make axes black and show ticks
fig.update_xaxes(showline=False, linewidth=1, linecolor='black', tickwidth=2)
fig.update_yaxes(showline=False, linewidth=1, linecolor='black', tickwidth=2)
# show x axis at zero
fig.update_xaxes(zeroline=False, zerolinewidth=1, zerolinecolor='black')
fig.update_yaxes(zeroline=True, zerolinewidth=1, linecolor='black')

fig.update_layout(yaxis=dict(zerolinecolor='black'))

fig.write_image(experiment_path + "plots/injection_exp/inj_accs_low_res.png", width=500, height=300)
fig.write_image(experiment_path + "plots/injection_exp/inj_accs.png", width=500, height=300, scale=2)
fig.write_image(experiment_path + "plots/injection_exp/inj_accs2.pdf", width=500, height=300, scale=2)
 

 # same for KL div
fig_kl = go.Figure()
fig_kl.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=kl_div_injs, mode='lines', name='Emb.+Lin2+Dec.', line=dict(color=inclusive_colors[6])))
fig_kl.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=kl_divs_dec_only, mode='lines', name='Decoder', line=dict(color=inclusive_colors[1])))
fig_kl.add_trace(go.Scatter(x=list(range(0, 1950, 10)), y=kl_div_all, mode='lines', name='All but att.', line=dict(color=inclusive_colors[2])))

fig_kl.update_traces(line=dict(width=4))

# background white, grey lines
fig_kl.update_layout(plot_bgcolor='white')
fig_kl.update_xaxes(showgrid=True, gridcolor='lightgrey')
fig_kl.update_yaxes(showgrid=True, gridcolor='lightgrey')

fig_kl.write_image(experiment_path + "plots/injection_exp/inj_kl_divs_low_res.png", width=800, height=600)
fig_kl.write_image(experiment_path + "plots/injection_exp/inj_kl_divs.png", width=800, height=600, scale=2)
fig_kl.write_image(experiment_path + "plots/injection_exp/inj_kl_divs.pdf", width=800, height=600, scale=2)




# get a confusion matrix of step and inj model
def get_confusion_matrix(logits, targets):
    preds = torch.argmax(logits, dim=1)
    cm = torch.zeros(115, 115)
    for i in range(len(preds)):
        cm[targets[i], preds[i]] += 1
    return cm



def plot_conf_mat(inj_logs, fname="plots/injection_exp/inj_confusion_matrix.png"):
    conf_inj = get_confusion_matrix(inj_logs, val_loader.dataset.targets)

    # plot confusion matrix
    fig_cm = px.imshow(conf_inj, color_continuous_scale='ylorbr', aspect="auto")
    fig_cm.update_traces(showscale=False)
    fig_cm.update_xaxes(title_text="Predicted", title_font=dict(size=28))
    fig_cm.update_yaxes(title_text="True", title_font=dict(size=28))
    fig_cm.update_xaxes(tickfont=dict(size=24))
    fig_cm.update_yaxes(tickfont=dict(size=24))
    fig_cm.update_layout(title_font=dict(size=26))
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="True")
    
    # increase font size of x and y axis labels
    fig_cm.update_xaxes(title_font=dict(size=28))
    fig_cm.update_yaxes(title_font=dict(size=28))

    # don't show a legend
    fig_cm.update_layout(showlegend=False)

    # make color bar go from white to blue
    

    fig_cm.write_image(experiment_path + fname.replace(".png", "_low_res.png"), width=800, height=600)
    fig_cm.write_image(experiment_path + fname, width=800, height=600, scale=2)
    fig_cm.write_image(experiment_path + fname.replace(".png", ".pdf"), width=800, height=600)

# confusion matrix at step 1300
inj_logs = inj_logits[110]
step_logs = step_logits[110]
plot_conf_mat(inj_logs, fname="plots/injection_exp/inj_confusion_matrix1100.png")
plot_conf_mat(step_logs, fname="plots/injection_exp/step_confusion_matrix1100.png")

inj_logs = inj_logits[170]
step_logs = step_logits[170]
plot_conf_mat(inj_logs, fname="plots/injection_exp/inj_confusion_matrix_1700.png")
plot_conf_mat(step_logs, fname="plots/injection_exp/step_confusion_matrix_1700.png")