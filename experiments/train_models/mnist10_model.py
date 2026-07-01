"""
MNIST-10 model training. Code forked from https://github.com/tuomaso/train_mnist_fast

This code trains a simple convolutional neural network on the MNIST dataset. The model is trained for 1 epoch with a learning rate of 0.01 and a batch size of 256. 
The training and test data are saved as PyTorch tensors in the 'data' directory.
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
import random
import os
import json
import numpy as np
import argparse

# import sys
# sys.path.append("explaind")  # add the root directory to the path

# export PYTHONPATH
# os.environ["PYTHONPATH"] = "../../"

from explaind.data_paths import DataPath
from explaind.model_paths import ModelPath
from explaind.optimizer_paths import AdamWOptimizerPath

def save_data():
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=10000, shuffle=True)

    train_data = []
    train_labels = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        train_data.append(data)
        train_labels.append(target)

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    torch.save(train_data, 'data/train_data.pt')
    torch.save(train_labels, 'data/train_labels.pt')
    
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=10000, shuffle=True)

    test_data = []
    test_labels = []

    for batch_idx, (data, target) in enumerate(test_loader):
        test_data.append(data)
        test_labels.append(target)

    test_data = torch.cat(test_data, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    torch.save(test_data, 'data/test_data.pt')
    torch.save(test_labels, 'data/test_labels.pt')

class Argument():
    def __init__(self, 
                batch_size=256, 
                test_batch_size=2000, 
                epochs=1, 
                lr=1e-2, 
                no_cuda=False, 
                max_samples=None,
                save_model=False):
        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.no_cuda = no_cuda
        self.save_model = save_model
        self.max_samples = max_samples

def set_seeds(seed):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed for both CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 5, 1)
        self.conv2 = nn.Conv2d(24, 32, 3, 1)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, *args, **kwargs):
        if len(x.shape) == 3:
            # add batch dim
            x = x.unsqueeze(0)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, 
          model,
          dataloader,
          data_path,
          device, 
          optimizer, 
          scheduler, 
          epochs,
          use_explaind=True,
          checkpoint_every=1, 
          checkpoint_path="results/mnist_cnn/checkpoints/"):
    model.train()

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    for j in range(epochs):
        # this loop needs only a single epoch
        for i, batch in enumerate(dataloader):
            if checkpoint_every > 0 and (i) % checkpoint_every == 0:
                # for the other attribution methods
                print(f"Checkpointing model at step {i}...")
                torch.save(model.state_dict(), checkpoint_path + f"model_step_{i}.pt")

            #i = i%batches_per_epoch
            # data = train_data[batch_size*i:batch_size*(i+1)]
            # target = train_labels[batch_size*i:batch_size*(i+1)]
            if use_explaind:
                data, target = data_path.get_batch(batch)
            else:
                data, target = batch

            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model.forward(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if use_explaind:
                print("log")
                model.log_checkpoint()
                optimizer.log_checkpoint()

            if scheduler is not None:
                scheduler.step()

    if use_explaind:
        model.save_checkpoints()
        optimizer.save_checkpoints()
        data_path.save_checkpoints()
    return model


def test(args, model, test_data, test_labels):
    model.eval()
    batch_size = 1024
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for i in range(len(test_data)//batch_size):
            data = test_data[batch_size*i:batch_size*(i+1)]
            target = test_labels[batch_size*i:batch_size*(i+1)]
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_labels)

    #print('Epoch {} Test set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n'.format(
    #    epoch, test_loss, 100. * correct / len(test_data)))
    return 100.*correct / len(test_labels)


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, device=None):
        self.data = data
        self.targets = targets
    
        if device is not None:
            self.data = self.data.to(device)
            self.targets = self.targets.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def train_mnist(use_explaind=True, 
                checkpoint_path="results/mnist_cnn/", 
                seed=42, 
                checkpoint_every=1,
                eval_acc=True,
                model_only=False,
                train_dataloader=None,
                long=False):
    args = Argument(max_samples=5120)
    if long:
        args = Argument(max_samples=30000, epochs=1, lr=1e-2)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    set_seeds(seed)

    if train_dataloader is None:
        train_data = torch.load('data/train_data.pt', map_location=device)
        train_labels = torch.load('data/train_labels.pt', map_location=device)

        if args.max_samples is not None:
            train_data = train_data[:min(args.max_samples, len(train_data))]
            train_labels = train_labels[:min(args.max_samples, len(train_labels))]

    if not model_only:
        test_data = torch.load('data/test_data.pt', map_location=device)
        test_labels = torch.load('data/test_labels.pt', map_location=device)

        # subsample test set to 1024
        rand_indices = torch.randperm(len(test_data))[:1024]
        test_data = test_data[rand_indices]
        test_labels = test_labels[rand_indices]

        if not os.path.exists(checkpoint_path) or not (os.path.isfile(checkpoint_path + "mnist_cnn_final.pt")):
            first_run = True
        else:
            first_run = False

        # store
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        if not os.path.isfile(checkpoint_path + "train_data_sampled.pt"):
            torch.save(train_data, checkpoint_path + "train_data_sampled.pt")
            torch.save(train_labels, checkpoint_path + "train_labels_sampled.pt")
            torch.save(test_data, checkpoint_path + "test_data_sampled.pt")
            torch.save(test_labels, checkpoint_path + "test_labels_sampled.pt")
        else:
            print(f"Sampled data already exists at {checkpoint_path}, loading it...")
            train_data = torch.load(checkpoint_path + "train_data_sampled.pt", map_location=device)
            train_labels = torch.load(checkpoint_path + "train_labels_sampled.pt", map_location=device)
            test_data = torch.load(checkpoint_path + "test_data_sampled.pt", map_location=device)
            test_labels = torch.load(checkpoint_path + "test_labels_sampled.pt", map_location=device)

    model = Net().to(device)
    if long:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.7, 0.9), eps=1e-8, weight_decay=0.1)

    # create a dataloader with targets
    # train_labels = [l for l in train_labels]
    if train_dataloader is None:
        train_dataset = MNISTDataset(train_data, train_labels, device=device)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    if not model_only:
        test_dataset = MNISTDataset(test_data, test_labels, device=device)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    train_data_path = None
    if use_explaind:
        # wrap the model with explaind
        train_data_path = DataPath(train_dataloader, checkpoint_path=checkpoint_path + "data_path.pt", overwrite=first_run, full_batch=False)
        model = ModelPath(model, checkpoint_path=checkpoint_path + "model_path.pt", overwrite=first_run, device=device, modelmodel=False, output_dim=10)
        optimizer = AdamWOptimizerPath(model, lr=args.lr, betas=(0.7, 0.9), eps=1e-8, weight_decay=0.1, checkpoint_path=checkpoint_path + "optimizer_path.pt", overwrite=first_run, device=device, track_lr=True)
    
    if long:
        scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                                total_steps=int(len(train_dataloader)*args.epochs), 
                                cycle_momentum=False)
    else:
        scheduler = None

    # check if we already have this run saved (in that case just load checkpoints and model and return)
    if not os.path.exists(checkpoint_path) or not (os.path.isfile(checkpoint_path + "mnist_cnn_final.pt")) or model_only:
        model = train(args, model, train_dataloader, train_data_path,
                        device, optimizer, scheduler, args.epochs, checkpoint_every=checkpoint_every,
                        use_explaind=use_explaind, checkpoint_path=checkpoint_path + "checkpoints/")

        if not model_only: 
            test_acc = test(args, model, test_data, test_labels)

            # print(f"Final test accuracy: {test_acc:.2f}%")

            # write to a results json
            results = {
                "test_acc": test_acc,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_samples": args.max_samples,
            }
            with open(checkpoint_path + "results.json", "w") as f:
                json.dump(results, f)

            torch.save(model.state_dict(), checkpoint_path + "mnist_cnn_final.pt")
    
        if model_only:
            return model

    model_orig = Net().to(device)
    state_dict = torch.load(checkpoint_path + "mnist_cnn_final.pt", map_location=device)
    # remove model. prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model_orig.load_state_dict(new_state_dict)

    if eval_acc:
        test_acc = test(args, model_orig, test_data, test_labels)
        print(f"Final test accuracy: {test_acc:.2f}%")
        print(f"Num test samples: {len(test_labels)}")

    checkpoints = []
    if checkpoint_every > 0:
        for i in range(0, int(((len(train_data)-1)//args.batch_size + 1)*args.epochs), checkpoint_every):
            checkpoint_path_i = checkpoint_path + f"checkpoints/model_step_{i}.pt"
            if os.path.isfile(checkpoint_path_i):
                checkpoints.append(torch.load(checkpoint_path_i, map_location=device))
            else:
                print(f"Checkpoint {checkpoint_path_i} not found.")
                break
    data = {
        "train_data": train_data.cpu(),
        "train_labels": train_labels.cpu(),
        "test_data": test_data.cpu(),
        "test_labels": test_labels.cpu(),
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
    }

    explaind_wrappers = {
        "model_path": model,
        "optimizer_path": optimizer,
        "data_path": train_data_path,
    }

    return model, checkpoints, data, explaind_wrappers


def main(path, long):
    print("Loading MNIST...")
    save_data()
    print("Data loaded. Starting training...")
    train_mnist(use_explaind=True, checkpoint_path=path, seed=42, long=long)
    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST-10 model.')
    parser.add_argument('--path', type=str, default="results/mnist_cnn/", help='Path to the experiment folder')
    parser.add_argument('--long', action="store_true", help="Whether to use longer train loop with more epochs and data, or fast&short version.")
    args = parser.parse_args()
    main(path=args.path, long=args.long)