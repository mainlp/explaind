import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from explaind.modulo_model.data import ModulusDataGenerator, ClassificationDataset
from explaind.modulo_model.model import SingleLayerTransformerClassifier
from  explaind.modulo_model.loss import RegularizedCrossEntropyLoss
from explaind.data_paths import DataPath
from explaind.model_paths import ModelPath
from explaind.optimizer_paths import AdamWOptimizerPath

import pandas as pd
import os

import random
import json
import argparse


# Set the random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Set the device
if not torch.cuda.is_available():
    print('No GPU available, using the CPU instead.')
else:
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def generate_data(num_samples, val_num_samples, p=113, power=1, batch_size=None):


    data_gen = ModulusDataGenerator(num_samples=num_samples, 
                                    val_num_samples=val_num_samples, 
                                    test_num_samples=0,
                                    P=p, 
                                    power=power)
    train_data, val_data, test_data = data_gen.generate_data()

    train_dataset = ClassificationDataset(train_data[0], train_data[1])
    val_dataset = ClassificationDataset(val_data[0], val_data[1])

    batch_size = num_samples if batch_size is None else batch_size

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, (train_data, val_data, test_data)

def store_weights(checkpoints, epoch, checkpoint_path):
    torch.save(checkpoints, checkpoint_path + f'epoch_{epoch}.pt')

def save_data(data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    train_data, val_data, test_data = data
    train_df = pd.DataFrame(train_data[0], columns=['a', 'add', 'b', 'eq'])
    train_df['target'] = train_data[1]
    train_df['add'] = train_df['add'].astype(int)
    train_df['eq'] = train_df['eq'].astype(int)
    train_df.to_csv(path + 'train.csv', index=False)
    val_df = pd.DataFrame(val_data[0], columns=['a', 'add', 'b', 'eq'])
    val_df['target'] = val_data[1]
    val_df['add'] = val_df['add'].astype(int)
    val_df['eq'] = val_df['eq'].astype(int)
    val_df.to_csv(path + 'val.csv', index=False)
    test_df = pd.DataFrame(test_data[0], columns=['a', 'add', 'b', 'eq'])
    test_df['target'] = test_data[1]
    test_df['add'] = test_df['add'].astype(int)
    test_df['eq'] = test_df['eq'].astype(int)
    test_df.to_csv(path + 'test.csv', index=False)

def output_fn(model, weights, input_ids, target_mask):

    # if len(input_ids.shape) == 1:
    input_ids = input_ids.unsqueeze(0)
    
    logits = torch.func.functional_call(model, (weights, dict()), args=(input_ids,))

    target_logit = torch.sum(logits * target_mask)

    return target_logit

def train_model(
    train_loader,
    val_loader,
    device,
    subset=-1,
    epk_history=True,
    model_id="0_epk",
    log_path="results/modulo_model/",
    epochs=5000,
    num_samples=4000,
    lr=0.001,
    hidden_size=64,
    dim_mlp=512,
    dropout=0.1,
    weight_decay=1.0,
    n_token=115,
    scale_weights=0.1,
    seed=42,
    use_writer=False,
    checkpoint_epochs=list(range(0, 5000, 300)),
):
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "/" + model_id, exist_ok=True)
    os.makedirs(log_path + "/" + model_id + "/tb_logs/", exist_ok=True)
    os.makedirs(log_path + "/" + model_id + "/checkpoints/", exist_ok=True)
    os.makedirs(log_path + "/data/", exist_ok=True)

    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # make subset if necessary
    print(subset)
    if subset > 0:
        # sample ids
        ids = np.random.choice(len(train_loader.dataset), subset, replace=False)
        # create new dataset
        train_loader.dataset = torch.utils.data.Subset(train_loader.dataset, ids)
        # create new dataloader
        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True)
        print("Subset size", len(train_loader.dataset))

        # store subset
        torch.save(train_loader, log_path + f"/data/train_loader_N={subset}.pt")

    # init model
    model = SingleLayerTransformerClassifier(n_token,
                                d_model=hidden_size,
                                nhead=4,
                                dim_mlp=dim_mlp,
                                dropout=dropout).to(device)

    # wrap into path wrapper
    if epk_history:
        model = ModelPath(model, device=device, checkpoint_path=checkpoint_path + "model_checkpoint.pt", overwrite=True)
    
    # scale model init weights 
    for p in model.parameters():
        p.data = scale_weights * p.data

    print(model)
    # setup cross entropy loss and optimizer
    loss_fct = RegularizedCrossEntropyLoss(alpha=0.0, p=2, device=device)

    if not epk_history:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))  # weight decay needed for grokking acording to Huang et al.
    else:
        optimizer = AdamWOptimizerPath(model, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98), checkpoint_path=checkpoint_path + "optimizer_checkpoint.pt", overwrite=True, device=device)
    
    if epk_history:
        data_path = DataPath(train_loader, checkpoint_path=checkpoint_path + "data_checkpoint.pt", overwrite=True, full_batch=False)

    print(optimizer)

    # setup tensorboard
    if use_writer:
        writer = SummaryWriter(log_dir=log_path + "/" + model_id + "/tb_logs/")

    # train model
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_reg = 0
        n = 0
        m = 0

        for batch in train_loader:

            if epk_history:
                x, y = data_path.get_batch(batch)

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model.forward(x)
            l, reg = loss_fct(output, y, params=model.parameters(), output_reg=True)

            l.backward()
            optimizer.step()

            # log checkpoint values we need for epk prediction
            model.log_checkpoint() 
            optimizer.log_checkpoint()

            train_loss += l.item()
            _, predicted = torch.max(output, 1)
            train_accuracy += (predicted == y).sum().item()
            train_reg += reg.item()
            n += len(y)
            m += 1

        train_loss /= m
        train_accuracy /= n
        train_reg /= m

        if use_writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Reg/train', train_reg, epoch)
        
        # validation
        model.eval()
        with torch.no_grad():
            val_accuracy = 0
            val_loss = 0
            val_reg = 0
            m, n = 0, 0
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                output = model.forward(x)

                l, reg = loss_fct(output, y, params=model.parameters(), output_reg=True)
                val_loss += l.item()
                val_reg += reg.item()

                _, predicted = torch.max(output, 1)
                val_accuracy += (predicted == y).sum().item()
                m += 1
                n += len(y)

            val_loss /= m
            val_accuracy /= n
            val_reg /= m


        print(f"Epoch {epoch},\
              Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f},\
              Train Acc: {train_accuracy:.6f}, Val Acc: {val_accuracy:.6f},\
              Train Reg: {train_reg:.4f}, Val Reg: {val_reg:.4f}")
        
        if epoch in checkpoint_epochs:
            torch.save(model.state_dict(), log_path + "/" + model_id + f"/checkpoint_epoch_{epoch}.pt")

        if use_writer:
            writer.add_scalar('Loss/val', val_loss, epoch) 
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Reg/val', val_reg, epoch)

            
        if train_accuracy == 1.0 and val_accuracy == 1.0:
            torch.save(model.state_dict(), log_path + "/" + model_id + f"/checkpoint_epoch_{epoch}.pt")
            break


    if use_writer:
        writer.close()

    optimizer.save_checkpoints()
    model.save_checkpoints()
    data_path.save_checkpoints()

    # return model, checkpoints, optimizer, loss_fct
    model_parts = {}
    model_parts["model"] = model
    # model_parts["weights"] = checkpoints
    model_parts["optimizer"] = optimizer
    model_parts["loss_fct"] = loss_fct

    # save a config with hyperparams etc
    config = {
        "epochs": epochs,
        "num_samples": num_samples,
        "lr": lr,
        "hidden_size": hidden_size,
        "dim_mlp": dim_mlp,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "n_token": n_token,
        "scale_weights": scale_weights,
        "seed": seed,
        "tensorboard_log_path": log_path,
        "checkpoint_path": checkpoint_path,
        "subset size": len(train_loader.dataset) if subset > 0 else -1,
        "epk_history": epk_history,
        "model_id": model_id,
        "checkpoint_epochs": checkpoint_epochs,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    with open(checkpoint_path + 'config.json', 'w') as f:
        json.dump(config, f)

    return model_parts, train_accuracy, val_accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a modulo model.')
    parser.add_argument('--n_samples', type=int, default=4000, help='Number of samples to train on')
    parser.add_argument('--p', type=int, default=113, help='Modulo value')
    parser.add_argument('--new_data', type=bool, default=True, help='Generate new data or use existing data')
    parser.add_argument('--path', type=str, default='results/modulo_val_results/', help='Path to save checkpoints')

    # checkpoint_path = 'results/modulo_val_results/'
    checkpoint_path = parser.parse_args().path
    n_samples = parser.parse_args().n_samples
    p = parser.parse_args().p
    new_data = parser.parse_args().new_data

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if parser.parse_args().new_data:
        train_loader, val_loader, data = generate_data(n_samples, 2000, p=p, power=1, batch_size=None)
        # store data as .csv
        save_data(data, checkpoint_path + "dataset/")
        
        # save dataloaders
        torch.save(train_loader, checkpoint_path + f"train_loader_N={n_samples}.pt")
        torch.save(val_loader, checkpoint_path + f"val_loader_N={n_samples}.pt")

    else:
        train_loader = torch.load(checkpoint_path + f"train_loader_N={n_samples}.pt")
        val_loader = torch.load(checkpoint_path + f"val_loader_N={n_samples}.pt")


    model, tacc, vacc = train_model(train_loader, val_loader, device, 
                                    log_path=checkpoint_path, 
                                    epochs=4000, lr=0.001, hidden_size=64, dim_mlp=512, 
                                    dropout=0.0, weight_decay=4.0,
                                    n_token=p+2, num_samples=n_samples, scale_weights=1.0, 
                                    seed=43, use_writer=True,)