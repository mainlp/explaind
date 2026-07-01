# Load ExPLAIND scores to derive data attrib scores and also baselines and compute lds
from dattri.metric import lds
import torch
from experiments.train_models.mnist10_model import train_mnist, MNISTDataset
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F

def lds_explaind(
    all_y,
    ground_truth,
    initial_preds,
    scores,
    reg,
    init_pred=True,
    targ="loss",
):
    """Calculate the Linear Datamodeling Score (LDS) metric.

    The LDS is calculated as the Spearman rank correlation between the predicted scores
    and the ground truth values for each test sample across all retrained models.

    Args:
        score (torch.Tensor): The data attribution score tensor with the shape
            (num_train_samples, num_test_samples).
        ground_truth (Tuple[torch.Tensor, torch.Tensor]): A tuple of two tensors. The
            first one has the shape (num_subsets, num_test_samples), which is the
            ground-truth target values for all test samples under `num_subsets` models,
            each retrained on a subset of the training data. The second tensor has the
            shape (num_subsets, subset_size), where each row refers to the indices of
            the training samples used to retrain the model.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors. The first tensor
            contains the Spearman rank correlation between the predicted scores and the
            ground truth values for each test sample. The second tensor contains the
            p-values of the correlation. Both have the shape (num_test_samples,).
    """
    gt_values, indices = ground_truth
    num_subsets = indices.shape[0]
    num_test_samples = scores.shape[0]

    print(gt_values)

    # Sum scores over the training subset indices
    print(indices.device, scores.device)
    print("Scores shape:", scores.shape)
    with torch.no_grad():
        sum_scores = torch.stack(
            [scores[:, :, indices[i]].sum(-1) for i in range(num_subsets)],
            dim=-1,
        ) # shape: (num_subsets, num_test_samples)
        print(sum_scores.shape)
        if init_pred:
            preds = initial_preds.unsqueeze(-1) + sum_scores + reg.unsqueeze(-1) * (len(indices[0]) / 5000)
        else:
            preds = sum_scores * (5000 / len(indices[0])) + reg.unsqueeze(-1) * (len(indices[0]) / 5000)
        print("Preds shape:", preds.shape)
        print(preds.device, all_y.device)
        losses = []
        for i in range(preds.shape[-1]):
            loss = -F.nll_loss(F.log_softmax(preds[:, :, i].squeeze(), dim=1), all_y, reduction="none")
            if targ == "loss":
                losses.append(loss)
            else:
                losses.append(loss - torch.log(1 - torch.exp(loss)))

        loss = torch.stack(losses, dim=-1) # shape: (num_subsets, num_test_samples)
        print("Loss shape:", loss.shape)
        sum_scores = loss.detach().T
        print(sum_scores)

    

    # Calculate the Spearman rank correlation between the average scores and the
    # ground-truth values for each test sample
    print(sum_scores.shape, gt_values.shape)
    lds_corr = torch.stack(
        [
            torch.tensor(
                spearmanr(sum_scores[:, i], gt_values[:, i]).correlation,
                dtype=scores.dtype,
            )
            for i in range(num_test_samples)
        ],
        dim=0,
    )  # shape: (num_test_samples,)
    lds_pval = torch.stack(
        [
            torch.tensor(
                spearmanr(sum_scores[:, i], gt_values[:, i]).pvalue,
                dtype=scores.dtype,
            )
            for i in range(num_test_samples)
        ],
        dim=0,
    )  # shape: (num_test_samples,)
    return lds_corr, lds_pval


# load explaind scores
explaind_results = torch.load("results/mnist_cnn/mnist10_epk_preds.pt")

explaind_kernels = [pred[1][2] for pred in explaind_results]
explaind_reg = [pred[1][3] for pred in explaind_results]
initial_preds = [pred[1][1] for pred in explaind_results]

# sum up the dicts in explaind_kernels and explaind_reg
explaind_kernels = [torch.stack(list(kernel_dict.values())) for kernel_dict in explaind_kernels]
explaind_kernels = [torch.sum(kernel, dim=0) for kernel in explaind_kernels]
explaind_kernels = torch.cat(explaind_kernels, dim=0)

explaind_reg = torch.cat(explaind_reg, dim=0)
initial_preds = torch.cat(initial_preds, dim=0)

print("ExPLAIND kernels shape:", explaind_kernels.shape)
print("ExPLAIND reg shape:", explaind_reg.shape)
print("ExPLAIND initial preds shape:", initial_preds.shape)

# load model
model, _, data, wrappers = train_mnist(checkpoint_path="results/mnist_cnn/")

params = wrappers["model_path"].checkpoints[-1]
# fix statedict
fixed = {}
for k, v in params.items():
    fixed["model."+ k] = v
model.load_state_dict(fixed)

test_loader = torch.utils.data.DataLoader(MNISTDataset(data["test_data"], data["test_labels"]), batch_size=64, shuffle=False)
model_preds = []
model.eval()
all_y = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(explaind_kernels.device)
        pred = model(x)
        model_preds.append(pred)
        all_y.append(y)
model_preds = torch.cat(model_preds, dim=0)
all_y = torch.cat(all_y, dim=0)

# if you rerun epk, 256 should not be necessary anymore
epk_preds = initial_preds + (explaind_kernels.sum(-1).squeeze()) + explaind_reg

print("EPK Accuracy:", (epk_preds.argmax(dim=-1) == model_preds.argmax(dim=-1)).float().mean().item())
print("Avg. delta between model and epk preds:", (model_preds - epk_preds).abs().mean().item())

print(explaind_kernels.shape)


# loss is just negative of correct out probability
explaind_scores = [explaind_kernels[i, :, y, :].squeeze() for i, y in enumerate(all_y)]
explaind_scores = torch.stack(explaind_scores, dim=0)
explaind_scores = explaind_scores.T

lds_all = {"explaind_0": [], "explaind_1": [], "trak-1": [], "tracin": [], "explaind_no_init": [], "if-cg": [], "explaind_proba": []}

# # compute lds
# for alpha in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
#     print(f"Computing LDS for alpha={alpha}")
#     ground_truth = torch.load(f"results/mnist_lds_ground_truth/alpha_{alpha}/ground_truth.pt")
#     lds_scores = lds(explaind_scores, ground_truth)
#     lds_mean = lds_scores[0].mean().item()
#     print(f"LDS for alpha={alpha}: {lds_mean}")
#     lds_all["explaind_0"].append(lds_mean)


explaind_scores = explaind_kernels.squeeze() 
# put all scores to cpu to save memory
explaind_scores = explaind_scores.cpu()
initial_preds = initial_preds.cpu()
all_y = all_y.cpu()
explaind_reg = explaind_reg.cpu()

for alpha in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
    print(f"Computing LDS for alpha={alpha} with kernel scores")
    ground_truth = torch.load(f"results/mnist_lds_ground_truth/alpha_{alpha}/ground_truth.pt")
    ground_truth = (ground_truth[0].cpu(), ground_truth[1].cpu())
    lds_scores = lds_explaind(all_y, ground_truth, initial_preds, explaind_scores, explaind_reg)
    lds_mean = lds_scores[0].mean().item()
    print(f"LDS for alpha={alpha} with kernel scores: {lds_mean}")
    lds_all["explaind_1"].append(lds_mean)

# for alpha in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
#     print(f"Computing LDS for alpha={alpha} with kernel scores")
#     ground_truth = torch.load(f"results/mnist_lds_ground_truth/alpha_{alpha}/ground_truth.pt")
#     ground_truth = (ground_truth[0].cpu(), ground_truth[1].cpu())
#     lds_scores = lds_explaind(all_y, ground_truth, initial_preds, explaind_scores, explaind_reg, init_pred=False)
#     lds_mean = lds_scores[0].mean().item()
#     print(f"LDS for alpha={alpha} with kernel scores: {lds_mean}")
#     lds_all["explaind_no_init"].append(lds_mean)

# for alpha in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
#     print(f"Computing LDS for alpha={alpha} with kernel scores")
#     ground_truth = torch.load(f"results/mnist_lds_ground_truth/alpha_{alpha}/ground_truth.pt")
#     ground_truth = (ground_truth[0].cpu(), ground_truth[1].cpu())
#     lds_scores = lds_explaind(all_y, ground_truth, initial_preds, explaind_scores, explaind_reg, targ="proba")
#     lds_mean = lds_scores[0].mean().item()
#     print(f"LDS for alpha={alpha} with kernel scores: {lds_mean}")
#     lds_all["explaind_proba"].append(lds_mean)

trak_scores = torch.load(f"results/mnist_lds_ground_truth/trak_attribution.pt")
# compare against baselines
for alpha in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
    print(f"Computing LDS for alpha={alpha} with trak-1")
    ground_truth = torch.load(f"results/mnist_lds_ground_truth/alpha_{alpha}/ground_truth.pt")
    lds_scores = lds(trak_scores, ground_truth)
    lds_mean = lds_scores[0].mean().item()
    print(f"LDS for alpha={alpha} with trak-1: {lds_mean}")
    lds_all["trak-1"].append(lds_mean)

tracin_scores = torch.load("results/mnist_lds_ground_truth/tracin_attribution.pt")
for alpha in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
    print(f"Computing LDS for alpha={alpha} with tracin")
    ground_truth = torch.load(f"results/mnist_lds_ground_truth/alpha_{alpha}/ground_truth.pt")
    lds_scores = lds(tracin_scores, ground_truth)
    lds_mean = lds_scores[0].mean().item()
    print(f"LDS for alpha={alpha} with tracin: {lds_mean}")
    lds_all["tracin"].append(lds_mean)


# # if cg
# if_cg_scores = torch.load("results/mnist_lds_ground_truth/if_attribution.pt")
# for alpha in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
#     print(f"Computing LDS for alpha={alpha} with influence functions")
#     ground_truth = torch.load(f"results/mnist_lds_ground_truth/alpha_{alpha}/ground_truth.pt")
#     lds_scores = lds(if_cg_scores, ground_truth)
#     lds_mean = lds_scores[0].mean().item()
#     print(f"LDS for alpha={alpha} with influence functions: {lds_mean}")
#     lds_all["if-cg"].append(lds_mean)

# for method, scores in lds_all.items():
#     print(f"LDS scores for {method}: {scores}")

# save lds_all to disk
import json
with open("results/mnist_lds_ground_truth/lds_all.json", "w") as f:
    json.dump(lds_all, f)



