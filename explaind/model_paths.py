"""
Here, we define a wrapper for standard torch models to include the parameter history and the ability to interpolate between two sets of weights.
"""

import torch
import torch.nn as nn
import os
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelPath(nn.Module):

    def __init__(self, model, 
                 device=device, 
                 checkpoints=None, 
                 checkpoint_path=None,
                 features_to_cpu=False,
                 parallelize_over_samples=True,
                 parallelize_over_targets=True,
                 overwrite=False,
                 output_dim=115):
        """
        Initialize the PathModel class.

        Args:

        model: torch.nn.Module
            The model to be wrapped.

        checkpoints: list of dict, default=None
            A list of dictionaries containing the model's parameters at different points in training.
            If None, try to load from checkpoint_path.

        checkpoint_path: str, default=None
            The path to load the checkpoints from. 
        """

        super(ModelPath, self).__init__()

        if checkpoint_path is not None and not os.path.exists(checkpoint_path):
            # make dirs before the file
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        

        self.model = model
        self.checkpoints = checkpoints
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.features_to_cpu = features_to_cpu
        self.parallelize_over_samples = parallelize_over_samples
        self.parallelize_over_targets = parallelize_over_targets
        self.overwrite = overwrite
        self.output_dim = output_dim

        if checkpoints is None and checkpoint_path is not None and not overwrite:
            self.load_checkpoints()
            self.model.load_state_dict(self.checkpoints[0])

        if self.checkpoints is None:
            self.checkpoints = []
            self.log_checkpoint()
        
        self.model.to(self.device)



    def forward(self, x, step=None):
        """
        Forward pass through the model.

        Args:

        x: torch.Tensor
            The input to the model.

        step: int, default=None
            The parameter step to use for interpolation. If None, use the latest checkpoint.
            In other words, it will compute model(x, params_step) where params_step is the step-th checkpoint.
        """

        self.model.eval()

        if step is None:
            return self.model(x.to(self.device))

        self.model.to(self.device)
        
        self.model.load_state_dict(self.checkpoints[step])

        return self.model(x.to(self.device))
    
    def to(self, device):
        self.device = device
        self.model.to(device)

    def __call__(self, *args, **kwds):
        return self.model.__call__(*args, **kwds)
    
    def train(self):    
        self.model.train()

    def eval(self):
        self.model.eval()

    def log_checkpoint(self):
        """
        Log the current model parameters.
        """
        checkpoint = deepcopy(self.model.state_dict())
        # put tensors on cpu
        for key in checkpoint.keys():
            checkpoint[key] = checkpoint[key].detach().clone().cpu()
        self.checkpoints.append(checkpoint)

    def parameters(self, *args, **kwargs):
        return super().parameters(*args, **kwargs)
    
    def forward_dataloader(self, data_loader, step=None):
        """
        Forward pass through the model.

        Args:

        x: torch.Tensor
            The input to the model.

        step: int, default=None
            The parameter step to use for interpolation. If None, use the latest checkpoint.
            In other words, it will compute model(x, params_step) where params_step is the step-th checkpoint.
        """
        self.model.eval()

        if step is not None:
            self.model.load_state_dict(self.checkpoints[step])
        
        self.model.to(self.device)
        
        predictions = []

        for X, y in data_loader:
            predictions.append(self.model(X.to(self.device)))

        # a matrix of shape (len(data_loader), #classes)
        predictions = torch.cat(predictions, dim=0)

        return predictions
    
    def interpolate_checkpoints(self, step0, step1, alpha):
        """
        Interpolate between two checkpoints.
        """

        checkpoint1 = self.checkpoints[step0]
        checkpoint2 = self.checkpoints[step1]

        interpolated_checkpoint = dict()

        for key in checkpoint1.keys():
            interpolated_checkpoint[key] = (1 - alpha) * checkpoint1[key] + alpha * checkpoint2[key]

        return interpolated_checkpoint
    
    def gradient_step_integral(self, X, y,  
                               step0, step1, 
                               pred_target, 
                               pred_steps,
                               optimizer=None):
        value = None

        for pred_step in pred_steps:

            step_checkpoint = self.interpolate_checkpoints(step0, step1, pred_step)
            pred_features = self.gradient_feature_map(X, y, 
                                                      step_checkpoint, 
                                                      pred_target)
            
            if optimizer is not None:
                pred_features = optimizer.test_feature_map(pred_features, pred_step)

            if value is None:
                value = pred_features
            else:
                # value = torch.stack([value, pred_features], dim=0).sum(dim=0)
                # we've got a dict now
                value = {name: torch.stack([value[name], pred_features[name]], dim=0).sum(dim=0) for name in value.keys()}
        
        # return 1 / len(pred_steps) * value
        return {name: v / len(pred_steps) for name, v in value.items()}
    
    def step_gradient_feature_map(self, X, y, step, pred_target_mask=None):
        """
        Compute the gradient feature map for a given step.
        """
        weights = self.checkpoints[step]
        return self.gradient_feature_map(X, y, weights, pred_target_mask=pred_target_mask)
    
    def gradient_feature_map_batched(self, 
                             data_loader, 
                             weight, 
                             pred_target_mask=None,
                             ):

        self.model.eval()
        self.model.load_state_dict(weight)
        self.model.to(self.device)

        grads = None

        for X, y in data_loader:
            
            if pred_target_mask is None:
                target_mask = self.get_target_mask(y, device=self.device)
            else:
                target_mask = torch.stack([pred_target_mask for _ in range(X.shape[0])], dim=0)

            fm = self._grad_feature_map(X, target_mask)

            if grads is None:
                grads = fm
            else:
                # concatenate over samples
                grads = {name: torch.cat([grads[name], fm[name]], dim=1) for name in fm.keys()}

        # 1 is the data dimension
        # grads = {name: torch.cat([g[name] for g in grads], dim=1) for name in grads[0].keys()}

        # reshape to (#target_features, #samples, #features)
        grads = {name: g.reshape(g.shape[0], g.shape[1], -1) for name, g in grads.items()}

        return grads
    
    def gradient_feature_map(self, X, y, weight, pred_target_mask=None):
        
        self.model.eval()
        self.model.load_state_dict(weight)
        self.model.to(self.device)

        grads = None
            
        if pred_target_mask is None:
            target_mask = self.get_target_mask(y, device=self.device)
        else:
            target_mask = torch.stack([pred_target_mask for _ in range(X.shape[0])], dim=0)

        fm = self._grad_feature_map(X, target_mask)

        grads = {name: g.reshape(g.shape[0], g.shape[1], -1) for name, g in fm.items()}

        return grads

    
    def _grad_feature_map(self, X, target_mask):

        # reset grads
        self.model.zero_grad()

        # only keep float valued params
        params = {}
        for k, v in self.model.state_dict().items():
            if v.dtype == torch.float and not k.endswith(r"running_mean") and not k.endswith(r"running_var"):
                params[k] = v

        if len(target_mask.shape) == 2:
            # do not parallelize over targets
            grads = torch.func.vmap(self._grads_out, in_dims=(None, None, 0, 0), out_dims=0)(self.model, params, X.to(self.device), target_mask)

            # first dim is target feature dim 
            grads = {name: g.unsqueeze(0) for name, g in grads.items()}

        elif len(target_mask.shape) == 3:
            # parallelize over targets

            if X.shape[0] == target_mask.shape[0] and len(X.shape) > 1:
                # repeat X
                X = torch.stack([X for _ in range(target_mask.shape[1])], dim=1)
                grads = torch.vmap(torch.func.vmap(self._grads_out, 
                                    in_dims=(None, None, 0, 0),
                                    out_dims=0),
                                in_dims=(None, None, 0, 0), 
                                out_dims=0
                                )(self.model, params, X.to(self.device), target_mask)
                
        else:
            raise ValueError("target_mask must be 2D or 3D tensor.")
                

                
            #     else:
            #         grads = torch.vmap(torch.func.vmap(self._grads_out, 
            #                                         in_dims=(None, None, 0, 0), 
            #                                         out_dims=0), 
            #                         in_dims=(None, None, None, 1),
            #                         out_dims=0
            #                         )(self.model, self.model.state_dict(), X.to(self.device), target_mask)
                    

            # else:
            #     target_grads = []
            #     for i in range(target_mask.shape[1]):
            #         grads = torch.func.vmap(self._grads_out, in_dims=(None, None, 0, 0), out_dims=0)(self.model, self.model.state_dict(), X.to(self.device), target_mask[:, i, :])
            #         grads = {name: g.unsqueeze(0) for name, g in grads.items()}
            #         target_grads.append(grads)
            #     grads = {name: torch.cat([g[name] for g in target_grads], dim=0) for name in target_grads[0].keys()}

        # detach the gradients
        grads = {name: g.detach() for name, g in grads.items()}

        if self.features_to_cpu:
            grads = {name: g.cpu() for name, g in grads.items()}

        #rint(grads.keys())
        return grads
    
    def _output_fn(self, model, checkpoints, input_ids, target_mask):
        input_ids = input_ids.unsqueeze(0)

        logits = torch.func.functional_call(model, (checkpoints, dict()), args=(input_ids,))

        if target_mask is not None:
            logits = torch.sum(logits * target_mask)

        return logits
    
    def _grads_out(self, *args, **kwargs):
        """
        Compute the gradients of the model's output with respect to the input.
        """
        return torch.func.grad(self._output_fn, has_aux=False, argnums=1)(*args, **kwargs)
    
    def get_target_mask(self, target, device):
        """
        Get the target mask for the model's output.
        """
        target_mask = self.get_one_hot_mask(target, device=device)

        return target_mask
    
    def get_regularization_term(self, data_features, step):
        """
        Calculate the regularization term at the specified step.
        """
        params = self.checkpoints[step]

        reg_term = None

        for k, v in data_features.items():
            
            # if len(v.shape) > 3:
            #     v = v.reshape(v.shape[0], v.shape[1], -1)  # should already be flattened

            params_flat = params[k].flatten()

            if self.features_to_cpu:
                params_flat = params_flat.cpu()

            prod = torch.einsum("abi,i->ab", v, params_flat)

            if reg_term is None:
                reg_term = prod
            else:
                reg_term += prod

        return reg_term
    
    def update_regularization_features(self, last_features, step, weight_decay, momentum=0.9):
        """
        Update the regularization features with the last features.
        """
        if last_features is None:
            last_features = {}
        
        next = {}

        for k, v in self.checkpoints[step].items():
            if k in last_features and momentum != 0:
                next[k] = weight_decay * v.clone().detach().flatten() + momentum * last_features[k]
            else:
                next[k] = weight_decay * v.clone().detach().flatten()

        return next

    
    def get_one_hot_mask(self, target, device):
        target_mask = torch.zeros(target.shape[0], self.output_dim).to(device)
        target_mask[torch.arange(target.shape[0]), target] = 1
        return target_mask
    
    def save_checkpoints(self):
        """
        Save the checkpoints to the checkpoint_path.
        """

        if self.checkpoint_path is None:
            print("No checkpoint path provided.")
            return

        torch.save(self.checkpoints, self.checkpoint_path)
    
    def load_checkpoints(self):
        """
        Load the checkpoints from the checkpoint_path.
        """

        if self.checkpoint_path is None:
            print("No checkpoint path provided (and likely also no checkpoints when instantiating).")
            return
        
        if not os.path.exists(self.checkpoint_path):
            print("Checkpoint path does not exist.")
            return

        self.checkpoints = torch.load(self.checkpoint_path, weights_only=True, map_location=self.device)
    
    def __str__(self):
        return f"ModelPath with {len(self.checkpoints)} checkpoints.\n{self.model}"




