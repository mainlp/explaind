from explaind.cifar_model.model import construct_rn9
from explaind.cifar_model.data import get_dataloader
import os
from tqdm import tqdm
from torch.amp import autocast #, GradScaler
import numpy as np
# from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from explaind.optimizer_paths import SGDOptimizerPath #, AdamWOptimizerPath
from explaind.model_paths import ModelPath
from explaind.data_paths import DataPath
from explaind.modulo_model.loss import RegularizedCrossEntropyLoss

import json

import torch
import warnings

import time
import random

import argparse

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

def sample_subset_and_store_mask(loader, mask_path, random_subset_alpha=0.1):
    """
    Sample a random subset of the dataset and store the mask in the specified path.
    
    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        mask_path (str): Path to store the mask.
        random_subset_alpha (float): Fraction of the dataset to sample.
    """
    # Create a mask for the random subset
    num_samples = int(len(loader.dataset) * random_subset_alpha)
    indices = np.random.choice(len(loader.dataset), size=num_samples, replace=False)
    mask = np.zeros(len(loader.dataset), dtype=bool)
    mask[indices] = True

    # Save the mask
    np.save(mask_path, mask)

    return apply_mask_to_loader(loader, mask_path)

def apply_mask_to_loader(loader, mask_path):
    """
    Apply the mask to the DataLoader to filter the dataset.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        mask_path (str): Path to the mask.
    
    Returns:
        torch.utils.data.DataLoader: Filtered DataLoader.
    """
    mask = np.load(mask_path)
    indices = np.where(mask)[0]
    
    # Create a new DataLoader with the filtered dataset
    filtered_dataset = torch.utils.data.Subset(loader.dataset, indices)
    filtered_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size=loader.batch_size, shuffle=True)
    
    return filtered_loader


def train_epk_cifar(epk_history=True, checkpoint_path="results/", loader=None, store_other_checkpoints=True,
                    mask_path=None,
                     seed=42, lr=0.1, epochs=24, momentum=0.9, batch_size=512, cifar_type="cifar10",
          weight_decay=5e-4, lr_peak_epoch=5, model_id=0, device='cuda', random_subset_alpha=-1):
    """
    Train a model on CIFAR-10 or CIFAR-2 using a setup similar to the TRAK paper. Automatically stores relevant
    checkpoints and config at "{checkpoint_path}{cifar_type}/checkpoints/".

    Args:
        epk_history (bool): If True, set up the training procedure such that it stores checkpoints to apply
            the exact path kernel. If False, train a standard model without path checkpoints.
        seed (int): Random seed for reproducibility (might not work when switching systems or versions).
        lr (float): Learning rate.
        epochs (int): Number of epochs to train for.
        momentum (float): Momentum for SGD.
        batch_size (int): Batch size for training.
        cifar_type (str): Type of CIFAR dataset to use. Can be "cifar10" or "cifar2" (cats and dogs).
        weight_decay (float): Weight decay for SGD.
        lr_peak_epoch (int): Epoch at which the learning rate peaks for the learning rate schedule.
        model_id (int): ID of the model to save checkpoints for.
        device (str): Device to use for training. Can be "cuda" or "cpu".
        random_subset_alpha (float): Fraction of the dataset to sample. If -1, no sampling is done.

    Returns:
        tuple of model, optimizer, and dataloader.
    """
    
    os.makedirs(f'{checkpoint_path}{cifar_type}/checkpoints/epk/model_{model_id}', exist_ok=True)
    os.makedirs(f'{checkpoint_path}{cifar_type}/checkpoints/standard/model_{model_id}', exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load the model
    if cifar_type == "cifar10":
        model = construct_rn9(num_classes=10).cuda()
    elif cifar_type == "cifar2":
        model = construct_rn9(num_classes=2).cuda()

    # Load the dataset
    if loader == None:
        loader, ds = get_dataloader(batch_size=batch_size, num_workers=8, split='train', shuffle=True, type=cifar_type)

        if random_subset_alpha > 0:
            mask_path = f'{checkpoint_path}{cifar_type}/checkpoints/standard/mask_{model_id}.npy'
            loader = sample_subset_and_store_mask(loader, mask_path, random_subset_alpha=random_subset_alpha)
            print(f"Random subset of {random_subset_alpha*100:.1f}% of the dataset sampled and stored at {mask_path}.")
            print(f"Subset size: {len(loader.dataset)} samples.")

    val_loader, val_ds = get_dataloader(batch_size=100, num_workers=8, split='test', shuffle=False, type=cifar_type)

    # pin memory
    val_loader.pin_memory = True
    loader.pin_memory = True

    print(f"Train size: {len(loader.dataset)} samples.")
    print(f"Val size: {len(val_loader.dataset)} samples.")

    t0 = time.time()
    
    if epk_history:

        opt = SGDOptimizerPath(model, lr=lr, momentum=momentum, weight_decay=weight_decay,
                            checkpoint_path=f'{checkpoint_path}{cifar_type}/checkpoints/epk/model_{model_id}/opt_{model_id}.pt',
                            device=device, 
                            overwrite=True)
        model = ModelPath(model=model, device=device, 
                        checkpoint_path=f'{checkpoint_path}{cifar_type}/checkpoints/epk/model_{model_id}/model_{model_id}.pt',
                        overwrite=True)
        datapath = DataPath(loader, checkpoint_path=f'{checkpoint_path}{cifar_type}/checkpoints/epk/model_{model_id}/data_{model_id}.pt',
                           full_batch=False, overwrite=True)

    else:
        opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    iters_per_epoch = len(loader)
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    if epk_history:
        scheduler = lr_scheduler.LambdaLR(opt.optimizer, lr_schedule.__getitem__)
    else:
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    #scaler = GradScaler()
    loss_fn = RegularizedCrossEntropyLoss(alpha=0.0)

    val_accs = []

    for ep in range(epochs):
        it = 0

        for it, batch in enumerate(loader):

            # print(batch)
            # print(batch[0].shape, batch[1].shape)
            if epk_history:
                ims, labs = datapath.get_batch(batch)  # removes additional indices and logs batch
            else:
                ims, labs = batch
            # print(ims.shape, labs.shape)

            ims = ims.cuda()
            labs = labs.cuda()
            opt.zero_grad(set_to_none=True)

            #with autocast(device_type=device):
            out = model.forward(ims)
            loss = loss_fn(out, labs)

            loss.backward()
            opt.step()

            #scaler.scale(loss).backward()
            #scaler.step(opt)
            #scaler.update()
            
            print(f"Epoch {ep+1}/{epochs}, Iteration {it+1}/{iters_per_epoch}, Loss: {loss.item():.4f}", end='\r')

            if epk_history:
                model.log_checkpoint()
                opt.log_checkpoint()

            # print(opt.optimizer.state_dict()["state"][0]["momentum_buffer"].sum())

            scheduler.step()

        val_acc = evaluate(model, val_loader, model_id=model_id, device=device)
        val_accs.append(val_acc)

        print(f"Validation Accuracy: {val_acc:.4f}")

        if store_other_checkpoints:
            torch.save(model.state_dict(), f'{checkpoint_path}{cifar_type}/checkpoints/standard/model_{model_id}/sd_{model_id}_epoch_{ep}.pt')

    # Save the model and optimizer checkpoints
    if epk_history:
        model.save_checkpoints()
        opt.save_checkpoints()
        datapath.save_checkpoints()
    if store_other_checkpoints:
        torch.save(model.state_dict(), f'{checkpoint_path}{cifar_type}/checkpoints/standard/model_{model_id}/sd_{model_id}_final.pt')

    if epk_history:
        train_acc = evaluate(model, loader, datapath, model_id=model_id, device=device)
    else:
        train_acc = evaluate(model, loader, model_id=model_id, device=device)
    val_acc = evaluate(model, val_loader, model_id=model_id, device=device)

    t1 = time.time()

    # save config
    config = {
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "val_accs": str(val_accs),
        "batch_size": batch_size,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "lr_peak_epoch": lr_peak_epoch,
        "model_id": model_id,
        "device": device,
        "loader.batch_size": loader.batch_size,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_size": len(loader.dataset),
        "val_size": len(val_loader.dataset),
        "total_time": t1 - t0,
        "cifar_type": cifar_type,
        "random_subset_alpha": random_subset_alpha,
        "epk_history": epk_history,
        "optimizer": "SGD",
        "model": "ResNet9",
        "loss_fn": "RegularizedCrossEntropyLoss",
        "mask_path": mask_path if random_subset_alpha > 0 else None,
        "datapath": f'{checkpoint_path}{cifar_type}/checkpoints/epk/model_{model_id}/data_{model_id}.pt' if epk_history else None,
        "model_path": f'{checkpoint_path}{cifar_type}/checkpoints/epk/model_{model_id}/model_{model_id}.pt' if epk_history else None,
        "opt_path": f'{checkpoint_path}{cifar_type}/checkpoints/epk/model_{model_id}/opt_{model_id}.pt' if epk_history else None,
        "scheduler": "LambdaLR"
    }

    if epk_history:
        with open(f'{checkpoint_path}{cifar_type}/checkpoints/epk/model_{model_id}/config.json', 'w') as f:
            json.dump(config, f, indent=4)
    else:
        with open(f'{checkpoint_path}{cifar_type}/checkpoints/standard/model_{model_id}/config.json', 'w') as f:
            json.dump(config, f, indent=4)

    if epk_history:
        return model, opt, datapath
    
    # store predictions of val set
    val_preds = []

    for batch in val_loader:
        ims, labs = batch
        ims = ims.cuda()
        labs = labs.cuda()
        out = model(ims)
        val_preds.append(out.cpu().detach().numpy())

    val_preds = np.concatenate(val_preds)
    print(f"Val preds shape: {val_preds.shape}")

    np.save(f'{checkpoint_path}{cifar_type}/checkpoints/standard/model_{model_id}/val_preds.npy', val_preds)

    return model, opt, loader



def evaluate(model, loader, datapath=None, model_id=0, device=device):
    model.eval()

    with torch.no_grad():
        total_correct, total_num = 0., 0.
        for batch in tqdm(loader):
            if datapath is not None:
                ims, labs = datapath.get_batch(batch, log=False)
            else:
                ims, labs = batch

            ims = ims.cuda()
            labs = labs.cuda()
            with autocast(device_type=device):
                out = model(ims)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

        print(f'Accuracy of model {model_id}: {total_correct / total_num * 100:.1f}%')

    return total_correct / total_num


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10 or CIFAR-2.')

    parser.add_argument("--path", type=str, default="results/", help="Path to store the model and config.")

    model, opt, datapath = train_epk_cifar(epk_history=True,
                                           store_other_checkpoints=False,
                                           device=device, 
                                           checkpoint_path=parser.parse_args().path,
                                           cifar_type="cifar2",
                                           batch_size=512,
                                           epochs=12, 
                                           lr=0.1, 
                                           weight_decay=0.001,
                                           lr_peak_epoch=5,
                                           momentum=0.9,
                                           model_id="epk_0",
                                           random_subset_alpha=-1)