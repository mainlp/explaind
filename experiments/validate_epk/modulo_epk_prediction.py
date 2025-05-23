import torch
from explaind.modulo_model.model import SingleLayerTransformerClassifier
from explaind.modulo_model.loss import RegularizedCrossEntropyLoss
from explaind.epk_model import ExactPathKernelModel
from explaind.model_paths import ModelPath
from explaind.optimizer_paths import AdamWOptimizerPath
from explaind.data_paths import DataPath
import json
import os
from tqdm import tqdm
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Validate EPK on modulo model')
parser.add_argument('--path', type=str, default="results/modulo_val_results/", help='Path to the experiment folder')

experiment_path = parser.parse_args().path
os.makedirs(experiment_path, exist_ok=True)
model_checkpoint_path = experiment_path + "model_checkpoint.pt"
optimizer_checkpoint_path = experiment_path + "optimizer_checkpoint.pt"
data_checkpoint_path = experiment_path + "data_checkpoint.pt"

# load datasets
train_loader = torch.load(experiment_path + "train_loader_N=4000.pt", weights_only=False)
val_loader = torch.load(experiment_path + "val_loader_N=4000.pt", weights_only=False)

data_path = DataPath(train_loader,
                        checkpoint_path=data_checkpoint_path,
                        full_batch=False,
                        overwrite=False)

config = {}
with open(experiment_path + "config.json", "r") as f:
    config = json.load(f)

# load model and optimizer
model = SingleLayerTransformerClassifier(config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])
model = ModelPath(model=model,
                  device=device, 
                  checkpoint_path=model_checkpoint_path,
                  features_to_cpu=False,
                  parallelize_over_targets=True)

optimizer = AdamWOptimizerPath(model, 
                               checkpoint_path=optimizer_checkpoint_path,
                               betas=(0.9, 0.98),
                               lr=config["lr"], 
                               weight_decay=config["weight_decay"], 
                               device=device,
                               cast_to_float16=False
                               )


epk = ExactPathKernelModel(
    model=model,
    optimizer=optimizer,
    loss_fn=RegularizedCrossEntropyLoss(alpha=0.0),
    data_path=data_path,
    integral_eps=0.01,
    features_to_cpu=False,
    device=device,
    train_set_size=4000,
    evaluate_predictions=True,
    keep_param_wise_kernel=True,
    param_wise_kernel_keep_out_dims=True,
    early_integral_eps=0.01,
    early_steps=5,
    grad_scaling=1.0,
    kernel_store_interval=50,
)

# make batch size small enough so you don't run OOM
val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=10, shuffle=False)

# from torch.profiler import profile, record_function, ProfilerActivity

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
#     with record_function("model_inference"):
#         # Run the model on the validation set
#         preds = []
#         for i, (X, y) in enumerate(val_loader):
#             X = X.to(device)
#             y = y.to(device)
#             pred = epk.predict(X, is_train=False, keep_kernel_matrices=False)
#             preds.append((i, pred, y))
#             break

# print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=50))
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))


preds = []
for i, (X, y) in enumerate(val_loader):
    torch.cuda.empty_cache()
    print(f"Batch {i}")
    # torch.cuda.empty_cache()
    X = X.to(device)
    y = y.to(device)
    pred = epk.predict(X, y_test=y, is_train=False, keep_kernel_matrices=True)
    for k, v in pred[4]["param_kernel"].items():
        pred[4]["param_kernel"][k] = v.cpu()

    for k, v in pred[2].items():
        pred[2][k] = v.cpu()
    
    preds.append((i, pred, y))


os.makedirs(experiment_path, exist_ok=True)

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
avg_pred_delta_std = 0
max_pred_delta = 0
max_pred_delta_std = 0

model_preds = []
epk_preds = []

kl_divs = []

for i, (X, y) in enumerate(val_loader):
    pred = preds[i][1][0]
    epk_preds.append(pred)
    print(pred)
    apred = torch.argmax(pred, dim=1).cpu()
    correct_ground_truth += (apred == y.cpu()).sum().item()
    total += y.size(0)

    model_pred = model.forward(X, step=len(model.checkpoints)-1).cpu()
    model_preds.append(model_pred)
    model_apred = torch.argmax(model_pred, dim=1)
    correct_model += (model_apred == apred).sum().item()

    avg_pred_delta += (model_pred.cpu() - pred.cpu()).abs().sum().item() / 115
    max_pred_delta = max((model_pred.cpu() - pred.cpu()).abs().max().item(), max_pred_delta)

    avg_pred_delta_std += (((model_pred.cpu() - pred.cpu()).sum() / 115)**2).sum().item() 

    kl_divs.append(torch.nn.functional.kl_div(pred.log_softmax(dim=1).cpu(), model_pred.softmax(dim=1), reduction='sum').item())

    if i == len(preds) - 1:
        break

kl_div = sum(kl_divs) / total
print(f"Ground truth accuracy: {correct_ground_truth / total}")
print(f"Model accuracy: {correct_model / total}")
print(f"Avg pred delta: {avg_pred_delta / total}")
print(f"Max pred delta: {max_pred_delta}")
print(f"KL diveregnce: {kl_div}")

print("Std deviation of pred delta: ", (avg_pred_delta_std / total)**(0.5))

print(model_pred.shape)
print(pred.shape)

results = {}
results["ground_truth_accuracy"] = correct_ground_truth / total
results["model_accuracy"] = correct_model / total
results["avg_pred_delta"] = avg_pred_delta / total
results["max_pred_delta"] = max_pred_delta
results["std_pred_delta"] = (avg_pred_delta_std / total)**(0.5)
results["kl_div"] = kl_div

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

del preds_combined
del model_preds_combined
del initial_preds_combined
del kernel_matrices
del reg_terms

# TODO: Concat / add prev_eval stuff and store (I know this is ugly but it does the job)
for_storing = {}
for k in preds[0][1][4].keys():
    print(k, type(preds[0][1][4][k]))
    for batch in tqdm(range(len(preds))):
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
                for_storing[k] = {}
            for k2 in preds[0][1][4][k][0].keys():
                if k2 not in for_storing[k].keys():
                    for_storing[k][k2] = torch.stack([preds[batch][1][4][k][step][k2] for step in range(len(preds[0][1][4][k]))], dim=1)
                else:
                    for_storing[k][k2] = torch.cat((for_storing[k][k2], torch.stack([preds[batch][1][4][k][step][k2] for step in range(len(preds[0][1][4][k]))], dim=1)), dim=0)
    
    # store this part of the dict directly and del
    if type(for_storing[k]) is list and type(for_storing[k][0]) is float:
        for i in range(len(for_storing[k])):
            for_storing[k][i] /= len(preds)

    torch.save(for_storing[k], experiment_path + f"{k}.pt")

    del for_storing[k]