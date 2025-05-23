"""
A wrapper for standard torch optimizers to include the history over different values that are needed
to calculate the EPK representation of the model.
"""

import torch
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OptimizerPath(optim.Optimizer):

    def __init__(self, 
                 optimizer, 
                 checkpoints=[], 
                 checkpoint_path=None,
                 device=device,
                 overwrite=False):
        """
        Initialize the OptimizerPath class.
        """

        super(OptimizerPath, self).__init__(optimizer.param_groups, optimizer.defaults)

        self.optimizer = optimizer
        self.checkpoints = checkpoints
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.overwrite = overwrite

        if checkpoint_path is not None and not os.path.exists(checkpoint_path):
            # make dirs before the file
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        if self.checkpoints == [] and self.checkpoint_path is not None and not self.overwrite:
            print("Loading optimizer checkpoints.")
            self.load_checkpoints()

    def test_feature_map(self, grads, step):
        """
        Test the feature map at the specified step.
        """
        # this changes for other optimizers
        return grads
    
    def train_feature_map(self, grads, step):
        """
        Train the feature map at the specified step.
        """
        # this changes for other optimizers
        return grads
    
    def get_step_kernel(self, data_features, train_features, step, next_avg, ids):
        """
        Calculate the kernel (dot product of sample features) for the specified step.
        """
        step_kernel = {name: torch.einsum("abc,dec->adbe", data_features[name], train_features[name]) for name in data_features.keys()}
    
        return step_kernel, next_avg
    
    def get_learning_rate(self, step):
        """
        Get the learning rate at the specified step.
        """
        return self.optimizer.param_groups[0]["lr"]
    
    def get_weight_decay(self, step):
        """
        Get the weight decay at the specified step.
        """
        return self.optimizer.param_groups[0]["weight_decay"]

    def regularization_term(self, grads, step):
        """
        Calculate the regularization term at the specified step.
        """
        return None
    
    def add_to_history(self, state_dict):
        """
        Add the optimizer state dictionary to the history.

        Needs to be reimplented usually.
        """
        self.checkpoints.append(state_dict)

    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        self.optimizer.step(closure)

    def log_checkpoint(self):
        """
        Log the optimizer checkpoint.
        """
        state_dict = self.optimizer.state_dict()
        self.add_to_history(state_dict)

    def load_checkpoints(self):
        """
        Load the optimizer checkpoints from the specified path.
        """
        if self.checkpoint_path is None or not os.path.exists(self.checkpoint_path):
            print("No optimizer checkpoints found at the specified path. Starting from scratch.")
            self.checkpoints = []
            return
        
        self.checkpoints = torch.load(self.checkpoint_path, weights_only=False)

    def save_checkpoints(self):
        """
        Save the optimizer checkpoints to the specified path.
        """
        if self.checkpoint_path is None:
            print("save_checkpoints: No optimizer checkpoints path specified.")
            return
        
        torch.save(self.checkpoints, self.checkpoint_path) 

    def get_value(self, key, step):
        """
        Get the value of the optimizer parameter at the specified step.
        """
        return self.get_checkpointed_value(key, step)

    def zero_grad(self, *args, **kwargs):
        """
        Zero the gradients of the model.
        """
        return self.optimizer.zero_grad(*args, **kwargs)

    def get_checkpointed_value(self, key, step):
        """
        Get the parameter at the specified step.
        """
        if step >= len(self.checkpoints):
            return None
        if key not in self.checkpoints[step]:
            return None
        return self.checkpoints[step][key]
    
    def get_checkpoint(self, step):
        """
        Get the checkpoint at the specified step.
        """
        return self.checkpoints[step]
    
    def __str__(self):
        return "OptimizerPath with {} checkpoints. \nCheckpoint path: {}\n{}".format(len(self.checkpoints), self.checkpoint_path, self.optimizer)

class AdamWOptimizerPath(OptimizerPath):

    def __init__(self, 
                 model, 
                 lr=0.001, 
                 betas=(0.9, 0.999), 
                 eps=1e-8, 
                 weight_decay=0.1, 
                 amsgrad=False,  # not implemented yet
                 track_lr=False,
                 track_weight_decay=False,
                 checkpoints=[], 
                 checkpoint_path=None,
                 overwrite=False,
                 cast_to_float16=False,
                 device=device):
        """
        Initialize the AdamWOptimizerPath class.
        """
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.current_step = 0
        self.lr = lr
        self.amsgrad = amsgrad
        self.device = device
        self.track_lr = track_lr
        self.track_weight_decay = track_weight_decay
        self.cast_to_float16 = cast_to_float16
        self.model = model

        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        super(AdamWOptimizerPath, self).__init__(optimizer, checkpoints, checkpoint_path, device, overwrite)

        # optimizer stores params as ids not by name, .parameters() is an OrderedDict
        self.id2param = {id: param for id, param in enumerate(model.state_dict())}
        self.param2id = {param: id for id, param in enumerate(model.state_dict())}

        # initialize the first checkpoint
        # initialize the second moment average (of not loaded before)
        if self.checkpoints == []:
            exp_avg_sq = {k: torch.zeros_like(v).cpu() for k, v in model.state_dict().items()}
            exp_avg = {k: torch.zeros_like(v).cpu() for k, v in model.state_dict().items()}
            self.checkpoints.append({"exp_avg_sq": exp_avg_sq, "exp_avg": exp_avg})

    def add_to_history(self, state_dict):
        """
        Add the optimizer state dictionary to the history.
        """
        # we only need the second moment terms
        checkpoint = {"exp_avg_sq": dict((self.id2param[k], v["exp_avg_sq"].detach().clone().cpu()) for k, v in state_dict["state"].items()),
                      "exp_avg": dict((self.id2param[k], v["exp_avg"].detach().clone().cpu()) for k, v in state_dict["state"].items())}
        
        if self.track_lr:
            checkpoint["lr"] = state_dict["param_groups"][0]["lr"]

        if self.track_weight_decay:
            checkpoint["weight_decay"] = state_dict["param_groups"][0]["weight_decay"]

        self.checkpoints.append(checkpoint)

    def get_learning_rate(self, step):
        """"
        Get the learning rate at the specified step.
        """
        if "lr" in self.checkpoints[step] and self.checkpoints[step]["lr"] is not None:
            return self.checkpoints[step]["lr"]

        return super().get_learning_rate(step)

    def get_weight_decay(self, step):
        """
        Get the weight decay at the specified step.
        """
        if "weight_decay" in self.checkpoints[step] and self.checkpoints[step]["weight_decay"] is not None:
            return self.checkpoints[step]["weight_decay"]

        return super().get_weight_decay(step)
    
    def second_moment_average(self, step, new_grads=None, flatten=False):
        """
        Calculate the second moment average at the specified step.
        """
        if self.get_value("exp_avg_sq", step+1) is not None and self.get_value("exp_avg_sq", step+1):
            second_moment_avg = self.get_value("exp_avg_sq", step+1)

        elif new_grads is not None and self.get_checkpointed_value("exp_avg_sq", step) is not None:
            second_moment_avg = self.update_second_moment_avg(new_grads, step)

        else:
            print("Second moment average could not be calculated.")
            return None, None
        
        # put to device
        second_moment_avg = {k: v.to(self.device) for k, v in second_moment_avg.items()}

        if self.cast_to_float16:
            second_moment_avg = {k: v.to(torch.bfloat16) for k, v in second_moment_avg.items()}

        # stored as dict of params
        scaled = dict((k, v / (1 - self.beta2 ** (step + 1))) for k, v in second_moment_avg.items())

        if flatten:
            scaled = torch.cat([v.flatten() for k, v in scaled.items()], dim=0)
            second_moment_avg = torch.cat([v.flatten() for k, v in second_moment_avg.items()], dim=0)

        return scaled, second_moment_avg

    def update_second_moment_avg(self, new_grads, step):
        """
        Update the second moment average.
        """
        last_val = self.get_checkpointed_value("exp_avg_sq", step)

        second_moment_avg = {}
        for k, v in last_val.items():
            if self.cast_to_float16:
                v = v.to(torch.bfloat16)
                new_grads[k] = new_grads[k].to(torch.bfloat16)
            second_moment_avg[k] = (self.beta2 * v.detach() + (1 - self.beta2) * new_grads[k].detach() ** 2)

        if len(self.checkpoints) > step + 1:
            self.checkpoints[step+1]["exp_avg_sq"] = second_moment_avg
        elif len(self.checkpoints) == step + 1:
            self.checkpoints.append({"exp_avg_sq": second_moment_avg, "exp_avg": {}})
        else:
            raise ValueError("Checkpoints are not in order. You need to compute previous steps first. (Current step: {}, requested step: {})".format(len(self.checkpoints), step))
        
        return second_moment_avg
    
    def test_feature_map(self, grads, step):
        """
        Test the feature map at the specified step.
        """
        return grads
    
    def get_reg_sign(self, step=None):
        """
        Get the sign of the regularization term.
        """
        return -1.0
    
    def apply_reg(self, pred, reg_term, last_term=None, step=0):
        """
        Apply the regularization term at the specified step.

        For AdamW, this is applied post-gradient update.
        """
        if reg_term is not None:
            lr = self.get_learning_rate(step + 1)
            weight_decay = self.get_weight_decay(step + 1)
            sign = self.get_reg_sign()
            pred = pred - lr * reg_term
        else:
            print("No regularization term provided. Skipping regularization.")
        
        return pred, None
    
    def train_feature_map(self, grads, previous, step, ids=None, dataset_size=-1):
        """
        Train the feature map at the specified step.
        """
        # this changes for other optimizers

        update = {}
        for k, v in  grads.items():

            if previous is None:
                previous = {}

            if self.cast_to_float16:
                grads[k] = grads[k].to(torch.bfloat16)

            if k not in previous:
                if step != 0:
                    raise ValueError("Previous is missing keys. This should only happen after the first step.")
                previous[k] = torch.zeros((grads[k].shape[0], dataset_size, grads[k].shape[2]), dtype=grads[k].dtype, device=grads[k].device)
            
            # assign previous value for the samples in this batch
            # exp_avg[k] = self.beta1 * previous[k] + (1 - self.beta1) * grads[k]
            previous[k] = self.beta1 * previous[k]

            update[k] = previous[k][:, ids].to(grads[k].device) + (1 - self.beta1) * grads[k]

            previous[k][:, ids] = update[k]

        # apply second moment average scaling
        second_moment_avg_scaled = self.second_moment_average(step, new_grads=grads)[0]

        norm_grads = {}
        for k, v in previous.items():
            if "model." + k in second_moment_avg_scaled:
                norm_grads[k] = v / (torch.sqrt(second_moment_avg_scaled["model." + k]).flatten() + self.eps) / (1 - self.beta1 ** (step + 1))
            else:
                # this can happen for parameters that are not updated 
                # (e.g. because their not p[art of the computational graph but still part of the model)
                norm_grads[k] = v.to(self.device)

        return norm_grads, previous
    
    def get_step_kernel(self, data_features, train_features, step, next_avg, ids):
        """
        Calculate the kernel (dot product of sample features) for the specified step.
        """
        # put data features on device
        if self.cast_to_float16:
            data_features = {name: v.to(torch.bfloat16) for name, v in data_features.items()}

        step_kernel = {name: torch.einsum("abc,dec->adbe", data_features[name], train_features[name]) for name in data_features.keys()}
        
        if self.cast_to_float16:
            # cast kernel back to float32
            step_kernel = {name: v.to(torch.float32) for name, v in step_kernel.items()}

        return step_kernel, next_avg
    
    def get_param_wise_kernel(self, data_features, train_features, to_cpu=False, keep_out_dims=False):
        """
        Calculate the kernel (dot product of sample features) for the specified step.
        """
        # put data features on device
        if self.cast_to_float16:
            data_features = {name: v.to(torch.bfloat16) for name, v in data_features.items()}

        # step_kernel = {name: torch.einsum("abc,dec->dbec", data_features[name], train_features[name]) for name in data_features.keys()}
        step_kernel = {}
        for name in data_features.keys():

            step_kernel[name] = torch.einsum("abc,dec->abdc", data_features[name], train_features[name])

            if not keep_out_dims:
                step_kernel[name] = step_kernel[name].abs().sum(1).sum(1)
        

        if self.cast_to_float16:
            # cast kernel back to float32
            step_kernel = {name: v for name, v in step_kernel.items()}

        if to_cpu: 
            step_kernel = {name: v.detach().cpu() for name, v in step_kernel.items()}
        else:
            step_kernel = {name: v.to(self.device).detach() for name, v in step_kernel.items()}

        return step_kernel
    
    def get_reg_term(self, data_features, previous_step_reg_features, step, prev_eval=None, param_kernel_store_interval=100):

        step_reg_features = self.model.update_regularization_features(previous_step_reg_features, 
                                                                      step, 
                                                                      self.get_weight_decay(step + 1), momentum=0.0)
        step_reg = {name: torch.einsum("abi,i->ab", v, step_reg_features[name]) for name, v in data_features.items()}

        step_reg_term = None
        for k, v in step_reg.items():

            if step_reg_term is None:
                step_reg_term = v
            else:
                step_reg_term = step_reg_term + v

        if prev_eval is not None:
            param_reg_term = {name: torch.einsum("abi,i->ai", v, step_reg_features[name]) for name, v in data_features.items()}
            if "param_reg_term" not in prev_eval or prev_eval["param_reg_term"] is None:
                prev_eval["param_reg_term"] = param_reg_term
            else:
                for name, v in param_reg_term.items():
                    prev_eval["param_reg_term"][name] = prev_eval["param_reg_term"][name] + v
            if step % param_kernel_store_interval == 0:
                prev_eval[f"param_reg_kernel_step_{step}"] = prev_eval["param_reg_term"]
                # reset
                prev_eval["param_reg_term"] = None

        return - self.get_learning_rate(step + 1) * step_reg_term, step_reg_features
    
class SGDOptimizerPath(OptimizerPath):
    """
    A wrapper for SGD optimizer to include the history over different values that are needed
    to calculate the EPK representation of the model.
    """

    def __init__(self, 
                 model, 
                 lr=0.01,
                 weight_decay=0.0001,
                 momentum=0.9,
                 dampening=0.0,
                 nesterov=False,
                 track_lr=True,
                 track_weight_decay=True,
                 checkpoints=[], 
                 checkpoint_path=None,
                 overwrite=False,
                 buffer_on_cpu=True,
                 buffer_float16=True,
                 buffer_device_mapping=None,
                 device=device):
        """
        Initialize the SGDOptimizerPath class. 
        
        See algorithm at:
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        """
        
        self.weight_decay = weight_decay
        self.lr = lr
        self.nesterov = nesterov
        self.momentum = momentum
        self.dampening = dampening
        self.device = device
        self.track_lr = track_lr
        self.track_weight_decay = track_weight_decay
        self.model = model
        self.buffer_on_cpu = buffer_on_cpu
        self.buffer_float16 = buffer_float16
        self.buffer_device_mapping = buffer_device_mapping

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, 
                              momentum=momentum, dampening=dampening)

        super(SGDOptimizerPath, self).__init__(optimizer, checkpoints, checkpoint_path, device, overwrite)

        # initialize the first checkpoint
        if self.checkpoints == []: 
            self.log_checkpoint()

    def get_learning_rate(self, step):
        """
        Get the learning rate at the specified step.
        """
        if "lr" in self.checkpoints[step] and self.checkpoints[step]["lr"] is not None:
            return self.checkpoints[step]["lr"]
        return super().get_learning_rate(step)
    
    def get_weight_decay(self, step):
        """
        Get the weight decay at the specified step.
        """
        if "weight_decay" in self.checkpoints[step] and self.checkpoints[step]["weight_decay"] is not None:
            return self.checkpoints[step]["weight_decay"]
        return super().get_weight_decay(step)
    
    def add_to_history(self, state_dict):
        """
        Add the optimizer state dictionary to the history.
        """
        checkpoint = {}

        if self.track_lr:
            checkpoint["lr"] = state_dict["param_groups"][0]["lr"]

        if self.track_weight_decay:
            checkpoint["weight_decay"] = state_dict["param_groups"][0]["weight_decay"]

        self.checkpoints.append(checkpoint)

    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        self.optimizer.step(closure)
    
    def log_checkpoint(self):
        """
        Log the optimizer checkpoint.
        """
        state_dict = self.optimizer.state_dict()
        self.add_to_history(state_dict)

    def test_feature_map(self, grads, step):
        return super().test_feature_map(grads, step)
    
    def get_reg_sign(self, step=None):
        """
        Get the sign of the regularization term.
        """
        return 1.0
    
    def get_reg_term(self, data_features, previous_step_reg_features, step, prev_eval=None, param_kernel_store_interval=100):

        step_reg_features = self.model.update_regularization_features(previous_step_reg_features, 
                                                                      step, 
                                                                      self.get_weight_decay(step + 1), momentum=self.momentum)
        step_reg = {name: torch.einsum("abi,i->ab", v, step_reg_features[name]) for name, v in data_features.items()}

        step_reg_term = None
        for k, v in step_reg.items():
            if step_reg_term is None:
                step_reg_term = v
            else:
                step_reg_term = step_reg_term + v

        return - self.get_reg_sign() * self.get_learning_rate(step + 1) * step_reg_term, step_reg_features

    def train_feature_map(self, grads, previous, step, ids=None, dataset_size=None, reg=None):
        """
        Train the feature map at the specified step.
        """
        # this changes for other optimizers

        if previous is None and self.momentum != 0:
            previous = {}
            for k, v in grads.items():

                if self.buffer_device_mapping is not None and self.buffer_float16:
                    previous[k] = torch.zeros((v.shape[0], dataset_size, v.shape[2]), dtype=torch.float16, device=self.buffer_device_mapping[k])
                elif self.buffer_device_mapping is not None:
                    previous[k] = torch.zeros((v.shape[0], dataset_size, v.shape[2]), device=self.buffer_device_mapping[k])
                
                elif self.buffer_float16 and self.buffer_on_cpu:
                    previous[k] = torch.zeros((v.shape[0], dataset_size, v.shape[2]), dtype=torch.float16).cpu()
                elif self.buffer_float16:
                    previous[k] = torch.zeros((v.shape[0], dataset_size, v.shape[2]), dtype=torch.float16).to(self.device)

                elif self.buffer_on_cpu:
                    previous[k] = torch.zeros((v.shape[0], dataset_size, v.shape[2])).cpu()
                else:
                    previous[k] = torch.zeros((v.shape[0], dataset_size, v.shape[2])).to(self.device)


        # apply weight decay
        for k, v in grads.items():
            
            if self.buffer_float16 and self.momentum != 0:
                grads[k] = grads[k].to(torch.float16)

            if self.buffer_device_mapping is not None and self.momentum != 0:
                # map to the device
                if k in self.buffer_device_mapping:
                    grads[k] = grads[k].to(self.buffer_device_mapping[k])
                else:
                    print("Key not in buffer device mapping: ", k)
                    print("Buffer device mapping: ", self.buffer_device_mapping)

            elif self.buffer_on_cpu and self.momentum != 0:
                grads[k] = grads[k].cpu()


        # add momentum features
        if previous is not None and self.momentum != 0:
            for k, v in previous.items():
                previous[k] *= self.momentum
                previous[k][:, ids] = previous[k][:, ids] + (1 - self.dampening) * grads[k].to(grads[k].device)

        else:
            previous = grads

        return previous, previous
    
    def get_step_kernel(self, data_features, train_features, step, next_avg, ids):
        """
        Calculate the kernel (dot product of sample features) for the specified step.
        """
        if self.momentum != 0:
            if self.buffer_float16:
                data_features = {name: v.to(torch.float16) for name, v in data_features.items()}

            if self.buffer_device_mapping is not None:
                data_features = {name: v.to(self.buffer_device_mapping[name]) for name, v in data_features.items()}

            elif self.buffer_on_cpu:
                data_features = {name: v.cpu() for name, v in data_features.items()}

            step_kernel = {name: torch.einsum("abc,dec->adbe", data_features[name], train_features[name]) for name in data_features.keys()}

            # put kernel back on device
            step_kernel = {name: v.to(self.device).to(torch.float32) for name, v in step_kernel.items()}
        else:
            step_kernel = {name: torch.einsum("abc,dec->adbe", data_features[name], train_features[name]) for name in data_features.keys()}

        return step_kernel, next_avg
    
    def get_param_wise_kernel(self, data_features, train_features, to_cpu=False, keep_out_dims=True):
        """
        Calculate the kernel (dot product of sample features) for the specified step.
        """
        if self.momentum != 0:
            if self.buffer_float16:
                data_features = {name: v.to(torch.float16) for name, v in data_features.items()}

            if self.buffer_device_mapping is not None:
                data_features = {name: v.to(self.buffer_device_mapping[name]) for name, v in data_features.items()}

            elif self.buffer_on_cpu:
                data_features = {name: v.cpu() for name, v in data_features.items()}

            step_kernel = {name: torch.einsum("abc,dec->abdc", data_features[name], train_features[name]) for name in data_features.keys()}

            # put kernel back on device
            if to_cpu:
                step_kernel = {name: v.detach().cpu().to(torch.float32) for name, v in step_kernel.items()}
            else:
                step_kernel = {name: v.to(self.device).detach().to(torch.float32) for name, v in step_kernel.items()}

        else:
            step_kernel = {name: torch.einsum("abc,dec->adbec", data_features[name], train_features[name]) for name in data_features.keys()}
    
        return step_kernel

    
