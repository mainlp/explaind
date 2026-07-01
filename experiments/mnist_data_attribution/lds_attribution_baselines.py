import torch
from torch import nn
import torch.nn.functional as F
import time
import json
import os

from dattri.algorithm import TracInAttributor, TRAKAttributor, IFAttributorCG
from dattri.task import AttributionTask
from dattri.benchmark.utils import SubsetSampler

from experiments.train_models.mnist10_model import train_mnist, MNISTDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloader(x, y, batch_size=256, device=device):
    x, y = x.to(device), y.to(device)
    ds = MNISTDataset(x, y, device=device)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    return dl


model, _, data, wrappers = train_mnist()  # this will just load the model and data if we have already run the training
checkpoints = wrappers["model_path"].checkpoints[1:-1:4] + [wrappers["model_path"].checkpoints[-1]] 

print(len(wrappers["model_path"].checkpoints))
print(len(checkpoints))

model = model.model
model.to(device)

train_data, train_labels = data["train_data"], data["train_labels"]
test_data, test_labels = data["test_data"], data["test_labels"]
train_loader = get_dataloader(train_data, train_labels)
test_loader = get_dataloader(test_data, test_labels)

def f(params, data_target_pair):
    x, y = data_target_pair
    x, y = x.to(device), y.to(device)
    loss = F.nll_loss
    yhat = torch.func.functional_call(model, params, x)
    l = loss(yhat, y)
    return l

def proba_target(params, data_target_pair):
    x, y = data_target_pair
    x, y = x.to(device).unsqueeze(0), y.to(device).unsqueeze(0)
    print(x.shape, y.shape)
    yhat = torch.func.functional_call(model, params, x)
    l = - F.nll_loss(yhat, y)
    prob = l - torch.log(1 - torch.exp(l))
    return prob

def proba(params, data_target_pair):
    x, y = data_target_pair
    x, y = x.to(device).unsqueeze(0), y.to(device).unsqueeze(0)
    print(x.shape, y.shape)
    yhat = torch.func.functional_call(model, params, x)
    l = torch.exp(-F.nll_loss(yhat, y))
    return l



task = AttributionTask(
    loss_func=proba_target,
    model=model,
    checkpoints=checkpoints,
)

if not os.path.exists("results/mnist_lds_ground_truth/trak_attribution.pt"):
    attributor = TRAKAttributor(task=task,
                                correct_probability_func=proba,
                                device=device)

    t0 = time.time()
    attributor.cache(train_loader)
    with torch.no_grad():
        score = attributor.attribute(test_loader)
    t1 = time.time()

    print(f"TRAK attribution took {t1 - t0:.2f} seconds")
    stats = {
        "time": t1 - t0,
    }
    if not os.path.exists("results/mnist_lds_ground_truth/trak_stats.json"):
        os.makedirs("results/mnist_lds_ground_truth", exist_ok=True)
    with open("results/mnist_lds_ground_truth/trak_stats.json", "w") as f:
        json.dump(stats, f)

    # checkpoint scores
    trak_attrib_path = f"results/mnist_lds_ground_truth/trak_attribution.pt"
    torch.save(score, trak_attrib_path)

if not os.path.exists("results/mnist_lds_ground_truth/tracin_attribution.pt"):
    attributor = TracInAttributor(task=task, device=device, weight_list=torch.ones(len(checkpoints)), normalized_grad=False)

    t0 = time.time()
    #attributor.cache(train_loader)
    with torch.no_grad():
        score = attributor.attribute(train_loader, test_loader)
    t1 = time.time()

    print(f"TracIn attribution took {t1 - t0:.2f} seconds")
    stats = {
        "time": t1 - t0,
    }
    if not os.path.exists("results/mnist_lds_ground_truth/tracin_stats.json"):
        os.makedirs("results/mnist_lds_ground_truth", exist_ok=True)
    with open("results/mnist_lds_ground_truth/tracin_stats.json", "w") as f:
        json.dump(stats, f)

    # checkpoint scores
    tracin_attrib_path = f"results/mnist_lds_ground_truth/tracin_attribution.pt"
    torch.save(score, tracin_attrib_path)


# IF
if not os.path.exists("results/mnist_lds_ground_truth/if_attribution.pt"):
    def proba_target(params, data_target_pair):
        x, y = data_target_pair
        x, y = x.to(device), y.to(device)
        #print(x.shape, y.shape)
        yhat = torch.func.functional_call(model, params, x)
        l = - F.nll_loss(yhat, y)
        prob = l - torch.log(1 - torch.exp(l))
        return prob

    train_loader = get_dataloader(train_data, train_labels, batch_size=128)
    test_loader = get_dataloader(test_data, test_labels, batch_size=128)


    task = AttributionTask(
        loss_func=proba_target,
        model=model,
        checkpoints=checkpoints[-1],
    )
    attributor = IFAttributorCG(
        task=task,
        max_iter=10,
        regularization=1e-2,
        device=device,
    )


    t0 = time.time()
    attributor.cache(train_loader)
    with torch.no_grad():
        score = attributor.attribute(train_loader, test_loader)
    t1 = time.time()

    print(f"IF attribution took {t1 - t0:.2f} seconds")
    stats = {
        "time": t1 - t0,
    }
    if not os.path.exists("results/mnist_lds_ground_truth/if_stats.json"):
        os.makedirs("results/mnist_lds_ground_truth", exist_ok=True)
    with open("results/mnist_lds_ground_truth/if_stats.json", "w") as f:
        json.dump(stats, f)

    if_attrib_path = f"results/mnist_lds_ground_truth/if_attribution.pt"
    torch.save(score, if_attrib_path)
