"""
Script to validate the predictions of the EPK model are correct for 
the CIFAR10 dataset. The script loads the model and the data, and then
compares the predictions of the model to the labels.
"""

import torch

from explaind.model_paths import ModelPath
from explaind.data_paths import DataPath
from explaind.cifar_model.data import get_dataloader
from explaind.cifar_model.model import construct_rn9
from explaind.optimizer_paths import SGDOptimizerPath
from explaind.epk_model import ExactPathKernelModel
from explaind.modulo_model.loss import RegularizedCrossEntropyLoss

import json
import plotly.express as px
import plotly.graph_objects as go


device = "cuda" if torch.cuda.is_available() else "cpu"

experiment_path = "results/cifar2_0.1/"
config_path = experiment_path + "config.json"
model_checkpoint_path = experiment_path + "model_epk_0.pt"
optimizer_checkpoint_path = experiment_path + "opt_epk_0.pt"
data_checkpoint_path = experiment_path + "data_epk_0.pt"

config = json.load(open(config_path, 'r'))

config["type"] = "cifar2"

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

optimizer = SGDOptimizerPath(model,
                            checkpoint_path=optimizer_checkpoint_path,
                            lr=config["lr"], 
                            momentum=config["momentum"], 
                            weight_decay=config["weight_decay"],
                            device=device,
                            buffer_float16=False,
                            # buffer_device_mapping={
                            #     "0.0.weight": "cuda:0",           #   172.8 MB
                            #     "9.weight": "cuda:0",             #   128.0 MB
                            #     "1.0.weight": "cuda:0",           #    20.48 GB
                            #     "2.module.0.0.weight": "cuda:0",  #    14.75 GB
                            #     "2.module.1.0.weight": "cuda:0",  #    14.75 GB
                            #     "3.0.weight": "cuda:2",           #    29.49 GB
                            #     "5.module.0.0.weight": "cuda:3",  #    58.98 GB
                            #     "5.module.1.0.weight": "cuda:1",  #    58.98 GB
                            #     "6.0.weight": "cuda:2",           #    29.49 GB
                            # },
                            buffer_device_mapping = {
                                "2.module.0.0.weight": "cuda",    # 27.47 GB
                                "3.0.weight": "cuda",             # 54.94 GB

                                "2.module.1.0.weight": "cuda",    # 27.47 GB
                                
                                "1.0.weight": "cuda",             # 38.15 GB
                                "6.0.weight": "cuda",             # 54.94 GB

                                "5.module.0.0.weight": "cuda",    # 109.89 GB

                                "5.module.1.0.weight": "cuda",    # 109.89 GB
                                "0.0.weight": "cuda",             # 0.32 GB
                                "9.weight": "cuda",               # 0.24 GB
                            },
                            buffer_on_cpu=False,
                            overwrite=False)

# optimizer = AdamWOptimizerPath(model,
#                                 checkpoint_path=optimizer_checkpoint_path,
#                                 lr=0.001,
#                                 betas=(0.9, 0.999),
#                                 weight_decay=0.1,
#                                 device=device, 
#                                 overwrite=False)

loader = get_dataloader(batch_size=256, num_workers=1, split='train', shuffle=False, augment=True, type=config["type"])[0]
datapath = DataPath(loader, 
                    checkpoint_path=data_checkpoint_path, 
                    full_batch=False, 
                    overwrite=False)

loss_fn = RegularizedCrossEntropyLoss(alpha=0.0)

epk = ExactPathKernelModel(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    data_path=datapath,
    integral_eps=0.1,
    features_to_cpu=False,
    device=device,
    train_set_size=config["train_size"],
    evaluate_predictions=True,
    early_integral_eps=0.1,
    early_steps=5,
    grad_scaling=1.0,
    kernel_store_interval=999999,
    keep_param_wise_kernel=True
)

torch.cuda.empty_cache()


# make batch size small enough so you don't run OOM
val_loader = get_dataloader(batch_size=10, num_workers=1, split='test', shuffle=False, augment=False, type="cifar2")[0]


# from torch.profiler import profile, record_function, ProfilerActivity

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
#     with record_function("model_inference"):
#         # Run the model on the validation set
#         preds = []
#         print(len(val_loader.dataset))
#         for i, (X, y) in enumerate(val_loader):
#             # we could put this loop into a function of the EPK class
#             # batch_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=200, shuffle=False)
#             pred = epk.predict(X, is_train=False, keep_kernel_matrices=True)
#             preds.append((i, pred, y))
# avgs = prof.key_averages()
# print(avgs.table(sort_by="cuda_memory_usage", row_limit=50))
# print(avgs.table(sort_by="cuda_time_total", row_limit=50))
# print(avgs.table(sort_by="cpu_time_total", row_limit=50))


preds = []
print(len(val_loader.dataset))
for i, (X, y) in enumerate(val_loader):
    torch.cuda.empty_cache()
    pred = epk.predict(X, is_train=False, keep_kernel_matrices=True)
    # send parameter kernels to cpu
    for k, v in pred[4]["param_kernel"].items():
        pred[4]["param_kernel"][k] = v.cpu()
    for k, v in pred[2].items():
        pred[2][k] = v.cpu()
    preds.append((i, pred, y))


epk_config = {
    "integral_eps": epk.integral_eps,
    "early_integral_eps": epk.early_integral_eps,
    "early_steps": epk.early_steps,
    "features_to_cpu": epk.features_to_cpu,
    "train_set_size": epk.train_set_size,
    "integral_steps": epk.integral_eps,
}

with open(experiment_path + "epk_config.json", 'w') as f:
    json.dump(epk_config, f, indent=4)


correct_ground_truth = 0
correct_model = 0
total = 0
avg_pred_delta = 0
max_pred_delta = 0

model_preds = []

for i, (X, y) in enumerate(val_loader):
    pred = preds[i][1][0]
    print(pred)
    apred = torch.argmax(pred, dim=1).cpu()
    correct_ground_truth += (apred == y.cpu()).sum().item()
    total += y.size(0)

    model_pred = model.forward(X, step=len(model.checkpoints)-1).cpu()
    model_preds.append(model_pred)
    model_apred = torch.argmax(model_pred, dim=1)
    correct_model += (model_apred == apred).sum().item()

    avg_pred_delta += (model_pred.cpu() - pred.cpu()).abs().sum().item() / 2
    max_pred_delta = max((model_pred.cpu() - pred.cpu()).abs().max().item(), max_pred_delta)

    if i == len(preds) - 1:
        break

print(f"Ground truth accuracy: {correct_ground_truth / total}")
print(f"Model accuracy: {correct_model / total}")
print(f"Avg pred delta: {avg_pred_delta / total}")
print(f"Max pred delta: {max_pred_delta}")

print(model_pred.shape)
print(pred.shape)

results = {}
results["ground_truth_accuracy"] = correct_ground_truth / total
results["model_accuracy"] = correct_model / total
results["avg_pred_delta"] = avg_pred_delta / total
results["max_pred_delta"] = max_pred_delta

with open(experiment_path + "validation_stats.json", 'w') as f:
    json.dump(results, f, indent=4)


# store combined matrix of kernel and regularization
preds_combined = None
for p in preds:
    if preds_combined is None:
        preds_combined = p[1][0]
    else:
        preds_combined = torch.cat((preds_combined, p[1][0]), dim=0)

print(preds_combined.shape)
torch.save(preds_combined, experiment_path + "epk_predictions.pt")
# store model predictions
model_preds_combined = torch.cat(model_preds, dim=0)
print(model_preds_combined.shape)
torch.save(model_preds_combined, experiment_path + "model_predictions.pt")

initial_preds_combined = None
for p in preds:
    if initial_preds_combined is None:
        initial_preds_combined = p[1][1]
    else:
        initial_preds_combined = torch.cat((initial_preds_combined, p[1][1]), dim=0)

print(initial_preds_combined.shape)
torch.save(initial_preds_combined, experiment_path + "initial_predictions.pt")



# kernel matrices
kernel_matrices = {}
for p in preds:
    for k, v in p[1][2].items():
        if k not in kernel_matrices:
            kernel_matrices[k] = v
        else:
            kernel_matrices[k] = torch.cat((kernel_matrices[k], v), dim=0)

# reg terms
reg_terms = None
for p in preds:
    if reg_terms is None:
        reg_terms = p[1][3]
    else:
        reg_terms = torch.cat((reg_terms, p[1][3]), dim=0)

# kernel_matrices_combined = torch.cat(kernel_matrices, dim=0)
# reg_terms_combined = torch.cat(reg_terms, dim=0)
print(kernel_matrices.keys())
print(reg_terms.shape)

torch.save(kernel_matrices, experiment_path + "kernel_matrices.pt")
torch.save(reg_terms, experiment_path + "reg_terms.pt")

# TODO: Concat / add prev_eval stuff and store (I know this is ugly but it does the job)
for_storing = {}
for k in preds[0][1][4].keys():
    for batch in range(len(preds)):
        print(k)
        print(k, type(preds[0][1][4][k]))

        if type(preds[0][1][4][k]) is dict:
            if k not in for_storing.keys():
                for_storing[k] = {}

            for k2 in preds[0][1][4][k].keys():
                if k2 not in for_storing[k].keys():
                    for_storing[k][k2] = preds[batch][1][4][k][k2]
                else:
                    for_storing[k][k2] = torch.cat((for_storing[k][k2], preds[batch][1][4][k][k2]), dim=0)

        elif type(preds[0][1][4][k][0]) is torch.Tensor:
            if k not in for_storing:
                for_storing[k] = preds[batch][1][4][k]
            else:
                for_storing[k] = [torch.cat((for_storing[k][i], preds[batch][1][4][k][i]), dim=0) for i in range(len(preds[0][1][4][k]))]

        elif type(preds[0][1][4][k][0]) is float:
            if k not in for_storing:
                for_storing[k] = preds[batch][1][4][k]
            else:
                for i in range(len(preds[0][1][4][k])):
                    for_storing[k][i] += preds[batch][1][4][k][i]

        elif type(preds[0][1][4][k][0]) is dict:
            print(k)
            if k not in for_storing.keys():
                print("We're here")
                for_storing[k] = {}
            for k2 in preds[0][1][4][k][0].keys():
                print("We're here")
                if k2 not in for_storing[k].keys():
                    print("We're here")
                    for_storing[k][k2] = torch.stack([preds[batch][1][4][k][step][k2] for step in range(len(preds[0][1][4][k]))], dim=1)
                else:
                    for_storing[k][k2] = torch.cat((for_storing[k][k2], torch.stack([preds[batch][1][4][k][step][k2] for step in range(len(preds[0][1][4][k]))], dim=1)), dim=0)

# divide the float values by the number of batches and stack dict tensors
for k in for_storing.keys():
    if type(for_storing[k]) is list and type(for_storing[k][0]) is float:
        for i in range(len(for_storing[k])):
            for_storing[k][i] /= len(preds)

# store each key in a separate file
for k in for_storing.keys():
    torch.save(for_storing[k], experiment_path + f"{k}.pt")


# make a scatter plot of all the float valued ones
for k in for_storing.keys():
    print(k)
    if type(for_storing[k]) is list and type(for_storing[k][0]) is float:
        fig = px.scatter(x=list(range(len(for_storing[k]))), y=for_storing[k])
        fig.update_layout(title=k)
        fig.update_xaxes(title_text="step")
        fig.update_yaxes(title_text="value")
        fig.write_image(experiment_path + f"{k}.png")

    elif type(for_storing[k]) is dict and type(for_storing[k][for_storing[k].keys()[0]]) is float:
        data = []
        for k2 in for_storing[k].keys():
            print(for_storing[k][k2].shape)
            all_vals = for_storing[k][k2].abs().sum(-1).sum(-1)
            plot_means = all_vals.mean(0)
            plot_stds = all_vals.std(0)
            data.append(go.Scatter(x=list(range(len(plot_means))), y=plot_means, name=k2))

        fig = go.Figure(data=data)
        fig.update_layout(title=k)
        fig.update_xaxes(title_text="step")
        fig.update_yaxes(title_text="value")
        fig.write_image(experiment_path + f"{k}.png")

        # also with logarithmic y axis
        fig.update_yaxes(type="log")
        fig.write_image(experiment_path + f"{k}_log.png")
   


