import torch
from torch import nn
import torch.nn.functional as F
import time
import json
import os
import numpy as np
import argparse

from experiments.train_models.mnist10_model import train_mnist, MNISTDataset

parser = argparse.ArgumentParser(description='Analyze ExPLAIND sensitivity.')
parser.add_argument('--experiment_path', type=str, default="results/mnist_sparse/", help='Path to the experiment folder')
args = parser.parse_args()

experiment_path = args.experiment_path

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloader(x, y, batch_size=5000, device=device):
    x, y = x.to(device), y.to(device)
    ds = MNISTDataset(x, y, device=device)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    return dl


# model, _, data, wrappers = train_mnist(checkpoint_path=experiment_path)  # this will just load the model and data if we have already run the training
# checkpoints = wrappers["model_path"].checkpoints
# model = model.model
# model.to(device)

# train_data, train_labels = data["train_data"], data["train_labels"]
# test_data, test_labels = data["test_data"], data["test_labels"]
# train_loader = get_dataloader(train_data, train_labels)
# test_loader = get_dataloader(test_data, test_labels)

explaind_results = torch.load(experiment_path + "mnist10_epk_preds.pt")


# # TOP-10
# frac = 10 / len(explaind_results[0][1][-1]["ground_truth_step"])
# print(f"Taking only TOP-10 steps ({frac} of total)")
# accs = []
# for j in range(len(explaind_results)):
#     step_gt = explaind_results[j][1][-1]["ground_truth_step"]
#     step_pred = explaind_results[j][1][-1]["predicted_step"]
#     init = explaind_results[j][1][1].cpu()
#     epk_pred = explaind_results[j][1][0].cpu()
#     # sanity check
#     summed = init + torch.stack(step_pred, dim=-1).sum(-1)
#     error = (epk_pred - summed).abs().sum()
#     sums = [sp.abs().sum().item() for sp in step_pred]
#     top10 = np.argsort(sums)[-num:]
#     steps_top10 = [step_pred[i] for i in top10]
#     summed_largest = init + torch.stack(steps_top10, dim=-1).sum(-1)
#     acc = (summed_largest.argmax(-1) == epk_pred.argmax(-1)).sum() / epk_pred.shape[0]
#     print("Sparse acc:", acc)
#     accs.append(acc)

# print("Overall:", sum(accs) / len(accs))
# print("Std. dev:", np.std(accs))

# frac = 0.2
# num = int(len(explaind_results[j][1][-1]["ground_truth_step"]) * frac)
# print(f"Taking only TOP-{num} steps ({frac} of total)")
# accs = []
# for j in range(len(explaind_results)):
#     step_gt = explaind_results[j][1][-1]["ground_truth_step"]
#     step_pred = explaind_results[j][1][-1]["predicted_step"]
#     init = explaind_results[j][1][1].cpu()
#     epk_pred = explaind_results[j][1][0].cpu()
#     # sanity check
#     summed = init + torch.stack(step_pred, dim=-1).sum(-1)
#     error = (epk_pred - summed).abs().sum()
#     sums = [sp.abs().sum().item() for sp in step_pred]
#     top10 = np.argsort(sums)[-num:]
#     steps_top10 = [step_pred[i] for i in top10]
#     summed_largest = init + torch.stack(steps_top10, dim=-1).sum(-1)
#     acc = (summed_largest.argmax(-1) == epk_pred.argmax(-1)).sum() / epk_pred.shape[0]
#     print("Sparse acc:", acc)
#     accs.append(acc)

# print("Overall:", sum(accs) / len(accs))
# print("Std. dev:", np.std(accs))

# frac = 0.3
# num = int(len(explaind_results[j][1][-1]["ground_truth_step"]) * frac)
# print(f"Taking only TOP-{num} steps ({frac} of total)")
# accs = []
# for j in range(len(explaind_results)):
#     step_gt = explaind_results[j][1][-1]["ground_truth_step"]
#     step_pred = explaind_results[j][1][-1]["predicted_step"]
#     init = explaind_results[j][1][1].cpu()
#     epk_pred = explaind_results[j][1][0].cpu()
#     # sanity check
#     summed = init + torch.stack(step_pred, dim=-1).sum(-1)
#     error = (epk_pred - summed).abs().sum()
#     sums = [sp.abs().sum().item() for sp in step_pred]
#     top10 = np.argsort(sums)[-num:]
#     steps_top10 = [step_pred[i] for i in top10]
#     summed_largest = init + torch.stack(steps_top10, dim=-1).sum(-1)
#     acc = (summed_largest.argmax(-1) == epk_pred.argmax(-1)).sum() / epk_pred.shape[0]
#     print("Sparse acc:", acc)
#     accs.append(acc)

# print("Overall:", sum(accs) / len(accs))
# print("Std. dev:", np.std(accs))

# frac = 0.4
# num = int(len(explaind_results[j][1][-1]["ground_truth_step"]) * frac)
# print(f"Taking only TOP-{num} steps ({frac} of total)")
# accs = []
# for j in range(len(explaind_results)):
#     step_gt = explaind_results[j][1][-1]["ground_truth_step"]
#     step_pred = explaind_results[j][1][-1]["predicted_step"]
#     init = explaind_results[j][1][1].cpu()
#     epk_pred = explaind_results[j][1][0].cpu()
#     # sanity check
#     summed = init + torch.stack(step_pred, dim=-1).sum(-1)
#     error = (epk_pred - summed).abs().sum()
#     sums = [sp.abs().sum().item() for sp in step_pred]
#     top10 = np.argsort(sums)[-num:]
#     steps_top10 = [step_pred[i] for i in top10]
#     summed_largest = init + torch.stack(steps_top10, dim=-1).sum(-1)
#     acc = (summed_largest.argmax(-1) == epk_pred.argmax(-1)).sum() / epk_pred.shape[0]
#     print("Sparse acc:", acc)
#     accs.append(acc)

# print("Overall:", sum(accs) / len(accs))
# print("Std. dev:", np.std(accs))


# frac = 0.5
# num = int(len(explaind_results[j][1][-1]["ground_truth_step"]) * frac)
# print(f"Taking only TOP-{num} steps ({frac} of total)")
# accs = []
# for j in range(len(explaind_results)):
#     step_gt = explaind_results[j][1][-1]["ground_truth_step"]
#     step_pred = explaind_results[j][1][-1]["predicted_step"]
#     init = explaind_results[j][1][1].cpu()
#     epk_pred = explaind_results[j][1][0].cpu()
#     # sanity check
#     summed = init + torch.stack(step_pred, dim=-1).sum(-1)
#     error = (epk_pred - summed).abs().sum()
#     sums = [sp.abs().sum().item() for sp in step_pred]
#     top10 = np.argsort(sums)[-num:]
#     steps_top10 = [step_pred[i] for i in top10]
#     summed_largest = init + torch.stack(steps_top10, dim=-1).sum(-1)
#     acc = (summed_largest.argmax(-1) == epk_pred.argmax(-1)).sum() / epk_pred.shape[0]
#     print("Sparse acc:", acc)
#     accs.append(acc)

# print("Overall:", sum(accs) / len(accs))
# print("Std. dev:", np.std(accs))

fracs = [f*0.1 for f in range(0, 11)]
all_nums = []
all_accs = []
all_stds = []
all_errs = []
all_errs_std = []

for frac in fracs:
    num = int(len(explaind_results[0][1][-1]["ground_truth_step"]) * frac)
    print(f"Taking only TOP-{num} steps ({frac} of total)")
    accs = []
    errors = []
    for j in range(len(explaind_results)):
        step_gt = explaind_results[j][1][-1]["ground_truth_step"]
        step_pred = explaind_results[j][1][-1]["predicted_step"]
        init = explaind_results[j][1][1].cpu()
        epk_pred = explaind_results[j][1][0].cpu()
        # summed prediction
        summed = init + torch.stack(step_pred, dim=-1).sum(-1)
        error = (epk_pred - summed).abs().sum()
        errors.append(error.item())
        sums = [sp.abs().sum().item() for sp in step_pred]
        top10 = np.argsort(sums)[-num:]
        steps_top10 = [step_pred[i] for i in top10]
        summed_largest = init + torch.stack(steps_top10, dim=-1).sum(-1)
        if frac == 0.0:
            summed_largest = init
        acc = ((summed_largest.argmax(-1) == summed.argmax(-1)).sum() / epk_pred.shape[0]).item()
        accs.append(acc)
        print("Sparse acc:", acc)
    all_accs.append(np.mean(accs))
    all_nums.append(num)
    all_stds.append(np.std(accs))
    all_errs.append(np.mean(errors))
    all_errs_std.append(np.std(errors))
    print("Numerical base error:", all_errs[-1])
    print("Numerical base error std:", all_errs_std[-1])
    print("Overall:", all_accs[-1])
    print("Std. dev:", all_stds[-1])

result = {
    "path": experiment_path,
    "frac": fracs,
    "num": all_nums,
    "acc": all_accs,
    "std": all_stds,
    "base_err": all_errs,
    "base_err_std": all_errs_std
}
print(experiment_path)
with open(experiment_path + "sensitivity_results_no_scal.json", "w") as f:
    json.dump(result, f)


fracs = [f*0.1 for f in range(0, 11)]
all_nums = []
all_accs = []
all_stds = []
all_errs = []
all_errs_std = []

for frac in fracs:
    num = int(len(explaind_results[0][1][-1]["ground_truth_step"]) * frac)
    print(f"Taking only TOP-{num} steps ({frac} of total)")
    accs = []
    errors = []
    for j in range(len(explaind_results)):
        step_gt = explaind_results[j][1][-1]["ground_truth_step"]
        step_pred = explaind_results[j][1][-1]["predicted_step"]
        init = explaind_results[j][1][1].cpu()
        epk_pred = explaind_results[j][1][0].cpu()
        # summed prediction
        summed = init + torch.stack(step_pred, dim=-1).sum(-1)
        error = (epk_pred - summed).abs().sum()
        errors.append(error.item())
        sums = [sp.abs().sum().item() for sp in step_pred]
        top10 = np.argsort(sums)[-num:]
        steps_top10 = [step_pred[i] for i in top10]
        scale_by = torch.stack(step_pred, dim=-1).sum() / torch.stack(steps_top10, dim=-1).sum()
        summed_largest = init + (scale_by * torch.stack(steps_top10, dim=-1).sum(-1))
        if frac == 0.0:
            summed_largest = init
        acc = ((summed_largest.argmax(-1) == summed.argmax(-1)).sum() / epk_pred.shape[0]).item()
        accs.append(acc)
        print("Sparse acc:", acc)
    all_accs.append(np.mean(accs))
    all_nums.append(num)
    all_stds.append(np.std(accs))
    all_errs.append(np.mean(errors))
    all_errs_std.append(np.std(errors))
    print("Numerical base error:", all_errs[-1])
    print("Numerical base error std:", all_errs_std[-1])
    print("Overall:", all_accs[-1])
    print("Std. dev:", all_stds[-1])

result = {
    "path": experiment_path,
    "frac": fracs,
    "num": all_nums,
    "acc": all_accs,
    "std": all_stds,
    "base_err": all_errs,
    "base_err_std": all_errs_std
}
print(experiment_path)
with open(experiment_path + "sensitivity_results_rescale.json", "w") as f:
    json.dump(result, f)