"""
A wrapper to keep track of the data during training. In the EPK model, this is used to
replicate the batches used during training. This implementation is somewhat naive
as we're just storing the batches in a list. This is not memory efficient, but it works
well with many different dataloader implementations. In the future, we might switch to
just storing dataset masks/indices, i.e. out own DataLoader implementation.
"""

import torch
from torch.utils.data import DataLoader
import os
import numpy as np


class DataPath(DataLoader):
    """
    A wrapper to keep track of the data during training. In the EPK model, this is used to
    replicate the batches used during training.
    """

    def __init__(self, dataloader, checkpoint_path=None, full_batch=True, overwrite=False):
        
        self.dataloader = dataloader
        self.full_batch = full_batch
        self.checkpoint_path = checkpoint_path
        self.checkpoints = []

        self.index_dataloader()

        if checkpoint_path is not None and not overwrite:
            if not os.path.exists(checkpoint_path):
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            self.load_checkpoints()

    def index_dataloader(self):
        """
        Append indices to the targets of the dataloader to keep track of the each sample.
        """
        new_targets = []
        for i in range(len(self.dataloader.dataset.targets)):
            if isinstance(self.dataloader.dataset.targets[i], torch.Tensor) and len(self.dataloader.dataset.targets[i].shape) == 2 and i == 0:
                return
            if isinstance(self.dataloader.dataset.targets[i], torch.Tensor) and len(self.dataloader.dataset.targets[i].shape) < 2:
                t = torch.LongTensor([self.dataloader.dataset.targets[i].item(), i])
                self.dataloader.dataset.targets[i] = t
                new_targets.append(t)
            elif isinstance(self.dataloader.dataset.targets[i], int) or isinstance(self.dataloader.dataset.targets[i], np.int64):
                # numbers are put into a new tensor with the index
                new_targets.append(torch.Tensor([int(self.dataloader.dataset.targets[i]), i]))
            elif isinstance(self.dataloader.dataset.targets[i], float) or isinstance(self.dataloader.dataset.targets[i], np.float64):
                # numbers are put into a new tensor with the index
                new_targets.append(torch.Tensor([float(self.dataloader.dataset.targets[i]), i]))
            else:
                print("Unknown type of targets in the dataset. Falling back to tuples (which might lead to problems later).")
                new_targets.append((self.dataloader.dataset.targets[i], i))

        self.dataloader.dataset.targets = new_targets
        

    def __next__(self):
        """
        Get the next batch of data.
        """
        batch = next(self.dataloader)
        if not self.full_batch or not self.checkpoints:
            self.log_batch(batch)

        batch = self.get_batch(batch)  # remove the indices from the batch

        return batch
    
    def get_batch(self, batch, target_type="int", log=True):
        """
        Remove the indices from the batch and log to checkpoints.
        """
        X, target = batch
        indices = target[:, -1].long()
        y = target[:, :-1].squeeze(1)

        if target_type == "int":
            y = y.long()
        elif target_type == "float":
            y = y.float()

        if log:
            self.log_batch(X, y, indices)

        return X, y
    
    def log_batch(self, X, y, indices):
        """
        Store the batch in the checkpoints.
        """
        X = X.detach().clone().cpu()
        y = y.detach().clone().cpu()
        indices = indices.detach().clone().cpu()
        self.checkpoints.append({
            'X': X,
            'y': y,
            'indices': indices
        })

    def get_checkpoint(self, index):
        """
        Get the checkpoint at the given index.
        """
        if index >= len(self.checkpoints):
            raise IndexError("Index out of range")

        return self.checkpoints[index]

    def save_checkpoints(self):
        """
        Save the checkpoints to a file.
        """
        torch.save(self.checkpoints, self.checkpoint_path)

    def load_checkpoints(self):
        """
        Load the checkpoints from a file.
        """
        if os.path.exists(self.checkpoint_path):
            self.checkpoints = torch.load(self.checkpoint_path)
        else:
            self.checkpoints = []

    def __len__(self):
        """
        Return the number of batches checkpointed.
        """
        return len(self.checkpoints)