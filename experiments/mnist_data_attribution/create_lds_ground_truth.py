import torch
from torch import nn
import torch.nn.functional as F

from dattri.algorithm import IFAttributorCG
from dattri.task import AttributionTask
from dattri.benchmark.utils import SubsetSampler

from experiments.train_models.mnist10_model import *
from experiments.train_models.mnist10_model import train_mnist, MNISTDataset

from dattri.model_util.retrain import retrain_lds, retrain_loo
from dattri.metric.ground_truth import calculate_lds_ground_truth, calculate_loo_ground_truth

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloader(x, y, batch_size=64, device=device):
    x, y = x.to(device), y.to(device)
    ds = MNISTDataset(x, y, device=device)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    return dl

retrain_func = lambda dataloader, **kwargs: train_mnist(train_dataloader=dataloader, use_explaind=False, model_only=True, eval_acc=False, checkpoint_every=-1, checkpoint_path="temp/", **kwargs)

model, _, data, wrappers = train_mnist()  # this will just load the model and data if we have already run the training
dataloader = get_dataloader(data["train_data"], data["train_labels"])

# LDS for alpha = 0.5, 0,7, 0.9
alphas = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]  # 0.999 is LOO basically
lds_path = "results/mnist_lds_ground_truth"
for alpha in alphas:
    print(f"Creating LDS ground truth for alpha={alpha}")
    alpha_path = f"{lds_path}/alpha_{alpha}"
    if os.path.exists(alpha_path):
        print(f"Path {alpha_path} already exists, skipping...")
        continue
    retrain_lds(train_func=retrain_func, dataloader=dataloader, seed=42, 
                path=alpha_path, subset_ratio=alpha, num_subsets=1000)
    

# LOO
# loo_path = f"{lds_path}/loo"
# if os.path.exists(loo_path):
#     print(f"Path {loo_path} already exists, skipping...")
# else:
#     # sample 1000 random samples from the training set to do LOO retraining, otherwise it will be too expensive to do LOO on the entire training set
#     ids = torch.randperm(len(data["train_data"]))[:1000]
#     retrain_loo(train_func=retrain_func, dataloader=dataloader, seed=42, path=loo_path, indices=ids)
#     # also store ids
#     torch.save(ids, f"{loo_path}/ids.pt")


# todo: create ground truths as in
# https://github.com/TRAIS-Lab/dattri/blob/ef50a35e6cdfa358ca104b7655ef15dcb9650afe/examples/customized_retraining/mnist.py#L159
# use above imports to get ground truth

def target_mnist(ckpt_path, dataloader, device=device):
    model = Net().to(device)
    checkpoint = torch.load(ckpt_path, weights_only=True)
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    model.eval()
    loss_list = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = F.nll_loss(yhat, y, reduction="none")
            loss_list.append(loss.clone().detach().cpu())
    return torch.cat(loss_list)

test_ds = MNISTDataset(data["test_data"], data["test_labels"], device=device)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

for alpha in alphas:
    print(f"Calculating LDS ground truth for alpha={alpha}")
    alpha_path = f"{lds_path}/alpha_{alpha}"
    if os.path.exists(f"{alpha_path}/ground_truth.pt"):
        print(f"Ground truth for alpha={alpha} already exists, skipping...")
        continue
    ground_truth = calculate_lds_ground_truth(target_mnist, alpha_path, test_loader)
    torch.save(ground_truth, f"{alpha_path}/ground_truth.pt")

# def loo_target_func(model, dataloader):
#     model.eval()
#     loss_list = []
#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             yhat = model(x)
#             loss = F.nll_loss(yhat, y, reduction="none")
#             loss_list.append(loss.clone().detach().cpu())
#     return torch.cat(loss_list)


# if not os.path.exists(f"{loo_path}/ground_truth.pt"):
#     loo_ground_truth = calculate_loo_ground_truth(target_mnist, loo_path, test_loader)
#     torch.save(loo_ground_truth, f"{loo_path}/ground_truth.pt")


# THEN: Load ground truth and benchmark TRAK, TracIN and ours
# TODO