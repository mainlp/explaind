import torch
from torch import nn
from copy import deepcopy
import os

import numpy as np
from tensordict import TensorDict
import re
import json

# from torchdata.stateful_dataloader import StatefulDataLoader as DataLoader
from tqdm import tqdm
from explaind.accelerated.layer_utils import DropoutWHistory, LayerNormWHistory


class ReplicatedModel(nn.Module):
    """
    A wrapper for a layer that replicates it and creates a tie between both copies,
    which are in turn kept on the same device in the automated device mapping we use
    with accelerate.
    """
    def __init__(self, 
                 name, 
                 model0, 
                 model1,
                 opt_state,
                 gradient_scaling,
                 current_step,
                 train_grad_scaling=1.0,
                 test_grad_scaling=1.0,
                 loss_grad=None,
                 num_steps=10,
                 device='cpu',
                 debug=False):
        super().__init__()
        self.model0 = model0
        self.interpolated_model = deepcopy(model0)
        self.model1 = model1
        self.name = name
        self.scores = None  # will be set in compute_influence_score
        self.accumulated = None  # will be set in compute_influence_score
        self.current_step = current_step
        self.gradient_scaling = gradient_scaling
        self.device = device
        self.train_grad_scaling = train_grad_scaling
        self.test_grad_scaling = test_grad_scaling

        self.batch_len = -1

        self.loss_grad = loss_grad

        self.debug = debug

        self.overall = 0

        # map opt_state to the parameters of the layer
        opt_state = {
            pname.lstrip(r"\."): opt_state[pname] 
            for pname in opt_state.keys()
        }
        if type(self.gradient_scaling) is not TensorDict:
            if type(self.gradient_scaling) is dict:
                self.gradient_scaling = TensorDict(self.gradient_scaling).to(self.device)
            else:
                self.gradient_scaling = TensorDict({pname: gradient_scaling for pname in opt_state.keys()}).to(self.device)

        self.learning_rate = TensorDict({pname: opt_state[pname]["lr"] for pname in opt_state.keys()}).to(self.device)
        self.weight_decay = TensorDict({pname: opt_state[pname]["weight_decay"] for pname in opt_state.keys()}).to(self.device)
        self.first_moment = TensorDict({pname: opt_state[pname]["exp_avg"] for pname in opt_state.keys()})
        # self.first_moment_new = {pname: opt_state[pname]["exp_avg_new"] for pname in opt_state.keys()}  # we will not need this later
        # self.second_moment_old = {pname: opt_state[pname]["exp_avg_sq_old"] for pname in opt_state.keys()}  # we will not need this later
        self.second_moment = TensorDict({pname: opt_state[pname]["exp_avg_sq"] for pname in opt_state.keys()})
        self.beta1 = TensorDict({pname: opt_state[pname]["betas"][0] for pname in opt_state.keys()}).to(self.device)
        self.beta2 = TensorDict({pname: opt_state[pname]["betas"][1] for pname in opt_state.keys()}).to(self.device)
     
        self.test_gradients = TensorDict({param[0]: torch.zeros_like(param[1].data) 
                               for param in self.model1.named_parameters() if param[1].requires_grad})

        # Number of steps for interpolation
        self.num_steps = num_steps
        self.completed_integral_steps = 0
        self.current_forward_alpha = 0.5 / self.num_steps
        self.reset()
        self.update_interpolated(self.current_forward_alpha)

    def get_tensordict_from_generator(self, generator):
        return TensorDict({k: v for k, v in generator})

    def get_grads(self, generator):
        return TensorDict({k: v.grad for k, v in generator})

    def forward(self, alpha=-1, **kwargs):
        for param in self.model0.parameters():
            assert param.grad is None, "Gradients should be None in forward pass."
        for param in self.model1.parameters():
            assert param.grad is None, "Gradients should be None in forward pass."
        for param in self.interpolated_model.parameters():
            assert param.grad is None, "Gradients should be None in forward pass."
        if self.current_forward_alpha == 0 or alpha == 0:
            out = self.model0.forward(**kwargs)
        elif self.current_forward_alpha == 1.0 or alpha == 1:
            out = self.model1.forward(**kwargs)
        elif self.current_forward_alpha < 1 and self.current_forward_alpha > 0:
            # Interpolated layer
            out = self.interpolated_model.forward(**kwargs)
        else:
            print(self.current_forward_alpha, alpha)
            raise ValueError("Alpha must be between 0 and 1.")
            
        return out
        
    # backward
    def backward(self, grad_output):
        if self.current_forward_alpha == 0:
            return self.model0.backward(grad_output)
        elif self.current_forward_alpha == 1:
            return self.model1.backward(grad_output)
        else:
            return self.interpolated_model.backward(grad_output)
        
    def update_interpolated(self, alpha):
        """
        Update the interpolated layer based on the feature maps of the two layers.
        The feature maps are used to compute the interpolation.
        """
        # Update the interpolated layer based on the feature maps
        for param0, param1, param_interpolated in zip(self.model0.parameters(), 
                                                      self.model1.parameters(), 
                                                      self.interpolated_model.parameters()):
            if param0.requires_grad:
                # Linear interpolation of parameters
                param_interpolated.data = (1 - alpha) * param0.data + alpha * param1.data

    def update_test_gradient_integral(self):
        """
        Update the test gradient integral for the layer.
        This method should be implemented to compute the integral using the test gradients.
        """
        # width of the current integral step (always the same for std. midpoint rule)
        width = 1.0 / self.num_steps

        # just for debugging purposes
        self.overall += width

        # extract the test gradients from the model parameters and add to the integral
        # self.test_gradients = {
        #     i: param.grad * width + self.test_gradients[i] 
        #     for i, param in self.interpolated_model.named_parameters() if param.requires_grad
        # }
        self.test_gradients += (self.get_grads(self.interpolated_model.named_parameters()) * width)

        self.completed_integral_steps += 1
        # midppoint rule (simple, but effective)
        self.current_forward_alpha = (0.5 + self.completed_integral_steps) / self.num_steps 

        # this updates the interpolation of the parameters to the next integral step
        self.update_interpolated(self.current_forward_alpha)

        # sets the gradients to None for the next forward pass
        self.reset_gradients()

    def compute_influence_score(self):
        """
        Compute the influence score for a test instance using the train gradients and test gradients.
        This method should be implemented to compute the influence score.
        """
        if self.completed_integral_steps < self.num_steps:
            raise ValueError("Integral not yet completed. Cannot compute influence score.")

        # Compute the influence scores as the elem-wise products of the train and test gradients
        # overwrite train_grads to save memory
        assert self.second_moment is not None, "Second moment must be computed before computing influence scores."

        self.scores = self.get_grads(self.model0.named_parameters()) * self.train_grad_scaling
        self.scores *= self.gradient_scaling * (1 - self.beta1) * self.learning_rate
        self.scores /= (1 - self.beta1 ** self.current_step)
        self.scores /= (self.second_moment.sqrt() / ((1 - (self.beta2 ** self.current_step))**0.5) + 1e-8)

        # dot product of feature maps
        self.scores *= -1
        self.scores *= self.test_gradients * self.test_grad_scaling

        self.accumulated = {
            i: {"sum": torch.sum(param).item(),
                "mean": torch.mean(param).item(), 
                "std": torch.std(param).item()}
                for i, param in self.scores.items()
        }
        return self.scores, self.accumulated
    
    def compute_regularization_scores(self):
        # Compute the regularization score for the layer. This is just the test map times the parameters.
        # overwrite train_grads to save memory
        self.scores = -self.get_tensordict_from_generator(self.model0.named_parameters())
        self.scores *= self.test_gradients * self.test_grad_scaling * self.weight_decay * self.learning_rate

        self.accumulated = {
            i: {
                "sum": torch.sum(param).item(),
                "mean": torch.mean(param).item(),
                "std": torch.std(param).item()
            }
            for i, param in self.scores.items()
            }
        return self.scores, self.accumulated

    def compute_first_moment_scores(self):
        # Compute the first moment score for the layer. This is just the test map times the parameters.
        self.scores = self.first_moment * self.beta1 * self.learning_rate
        self.scores /= (1 - self.beta1 ** self.current_step)
        self.scores /= ((self.second_moment.sqrt() / ((1 - (self.beta2 ** self.current_step))**0.5) + 1e-8))
        self.scores = -self.scores * self.test_gradients * self.test_grad_scaling
  
        self.accumulated = {
            i: {
                "sum": torch.sum(param).item(),
                "mean": torch.mean(param).item(),
                "std": torch.std(param).item()
            }
            for i, param in self.scores.items()
        }
        return self.scores, self.accumulated
    
    def reset(self):
        """
        Reset the layer to its initial state.
        This method should be implemented to reset the layer's parameters and gradients.
        """
        # Reset the layer's parameters and gradients
        for param in self.model0.parameters():
            param.grad = None
        for param in self.model1.parameters():
            param.grad = None
        for param in self.interpolated_model.parameters():
            param.grad = None
        # self.train_gradients = {param[0]: torch.zeros_like(param[1].data) for param in self.model0.named_parameters() if param[1].requires_grad}
        self.test_gradients = TensorDict({param[0]: torch.zeros_like(param[1].data) for param in self.model1.named_parameters() if param[1].requires_grad})
        self.scores = None

        self.current_forward_alpha = 0.5 / self.num_steps
        self.completed_integral_steps = 0
        self.eval()
        self.reset_gradients()
    
    def reset_alpha(self):
        """
        Reset the alpha value for the layer.
        This method should be implemented to reset the alpha value.
        """
        self.current_forward_alpha = 0.5 / self.num_steps
        self.overall = 0

    def reset_gradients(self):
        """
        Reset the gradients for the layer.
        This method should be implemented to reset the gradients.
        """
        # Reset the gradients for the layer
        for param in self.model0.parameters():
            if param.grad is not None:
                param.grad = None
        for param in self.model1.parameters():
            if param.grad is not None:
                param.grad = None
        for param in self.interpolated_model.parameters():
            if param.grad is not None:
                param.grad = None
        self.scores = None


class StepExplainer:
    def __init__(self, 
                 checkpoint_path,
                 test_instances=None,
                 save_all_scores=False,
                 ignore_train_types=True,
                 expl_type="outputs",  # alternatively, 'loss' will be implemented
                 score_path=None,
                 num_steps=10,
                 log_bf16=True,
                 compute_sim_scores=False,
                 test_batch_limit=-1,
                 debug=False,):
        
        # ask accelerator to load checkpoint
        self.accelerator = None
        # self.accelerator.load_state(checkpoint_path)

        self.model = None  # maybe we need a wrapper for this to get accelerate to parallelize it correctly
        self.optimizer = None
        self.scheduler = None
        self.train_instances = None
        self.wrapped_model = None  # will be set when computing the step model
        self.config = None
        self.lr = None  # learning rate, will be set when loading the checkpoint

        self.train_grad_scaling = 1.0
        self.log_bf16 = log_bf16
        self.compute_sim_scores = compute_sim_scores

        self.debug=debug
        self.ignore_train_types = ignore_train_types
        self.expl_type = expl_type

        self.checkpoint_path = checkpoint_path
        self.num_steps = num_steps
        self.step_id = -1  # will be set when loading the checkpoint
        self.test_instances = test_instances

        self.tied_layer_map = None  # will be set when loading the checkpoint if tied weights are used
        self.optimizer_state_mapping = None
        self.optimizer_pre = None
        self.sim_scores = None  # will be set when computing the influence scores

        self.all_scores = {}
        self.acc_scores = {}
        self.acc_layer_scores = {}

        self.ignore_index = -100  # for loss computations

        self.test_batch_limit = test_batch_limit

        self.save_all_scores = save_all_scores

        self.score_path = score_path if score_path is not None else os.path.join(self.checkpoint_path, "scores")

        self.load_checkpoint()

        if self.optimizer_state_mapping is None:
            self.optimizer_state_mapping = {}
            for param_group in self.optimizer.state_dict()["param_groups"]:
                for id, name in zip(param_group["params"], param_group["param_names"]):
                    self.optimizer_state_mapping[name] = id
        
        if self.optimizer_pre is None:
            self.optimizer_pre = deepcopy(self.optimizer)

        self.compute_step_model()

         # add ids for train sample types
        self.train_types = list(self.train_instances.keys())
        self.test_types = list(self.test_instances.keys())

    def load_checkpoint(self, path=None):
        raise NotImplementedError("This method should be implemented in a subclass.")
    
    def compute_step_model(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def _replace_history_layers(self, prefix, model):
        """
        Recursively replace Dropout and LayerNorm layers with their counterparts that keep track of history.

        More layers types to be supported in the future.
        """
        for name, layer in zip(model.named_children(), model.children()):
            # recursively wrap submodules
            
            if isinstance(layer, nn.Module):
                subprefix = prefix + "." + name[0] if prefix != "" else name[0]
                if len(list(layer.children())) > 0:
                    # If the layer has children, recursively wrap them
                    # print(f"Entering submodule: {name[0]} ({subprefix})")
                    new = self._replace_history_layers(subprefix, layer)
                
                else:
                    if isinstance(layer, nn.Dropout):
                        new = DropoutWHistory(p=layer.p, inplace=layer.inplace)
                    elif isinstance(layer, nn.LayerNorm):
                        new = LayerNormWHistory(layer.normalized_shape, 
                                                layer.eps, 
                                                layer.elementwise_affine,
                                                weight_param=layer.weight,
                                                bias_param=layer.bias)
                    else:
                        new = layer
        
                setattr(model, name[0], new)
            
        return model

    def _wrap_model(self, prefix, model0, model1):
        """
        Wrap only the overall model. Don't wrap submodules.
        """
        return ReplicatedModel(prefix, 
                               model0, 
                               model1, 
                               self._get_optimizer_state(model0, prefix), 
                               self.gradient_scaling, 
                               self.step_id + 1, # see: https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py#L347
                               num_steps=self.num_steps,
                               train_grad_scaling=self.train_grad_scaling,
                               test_grad_scaling=self.test_grad_scaling,)
        
    
    def _get_optimizer_state(self, layer, name):
        """
        Get the optimizer state for the given layer. Only supports AdamW optimizer for now.
        """
        # print("Getting optimizer state for layer:", name)
        param_names = [name + "."+ n[0] for n in layer.named_parameters() if n[1].requires_grad]
        # print("Parameter names:", param_names)
        # check if we have tied weights
        if self.tied_layer_map is not None:
            for i, n in enumerate(param_names):
                if n in self.tied_layer_map.keys():
                    # replace with the tied weight
                    tied_name = self.tied_layer_map[n]
                    # print(f"Replacing {n} with tied weight {tied_name}")
                    param_names[i] = tied_name

        param_name_ids = [self.optimizer_state_mapping[n.lstrip(r"\.")] for n in param_names]
        # print("Parameter name ids:", param_name_ids)
        # print()
        param_optimizer_states = {n : dict() for n in param_names}

        # add learning rate, weight decay, ... if available
        # iterate over param groups
        for k, n in zip(param_name_ids, param_names):
            for j, param_group in enumerate(self.optimizer_pre.state_dict()["param_groups"]):
                if k in param_group["params"]:
                    # second moment from post-optimization step
                    # param_optimizer_states[n]["exp_avg_new"] = self.optimizer.state_dict()["state"][k]["exp_avg"]
                    # param_optimizer_states[n]["exp_avg_sq_old"] = self.optimizer_pre.state_dict()["state"][k]["exp_avg_sq"]
                    param_optimizer_states[n]["exp_avg_sq"] = self.optimizer.state_dict()["state"][k]["exp_avg_sq"]
                    param_optimizer_states[n]["exp_avg"] = self.optimizer_pre.state_dict()["state"][k]["exp_avg"]
                    param_optimizer_states[n]["weight_decay"] = param_group["weight_decay"]
                    param_optimizer_states[n]["lr"] = param_group["lr"]
                    param_optimizer_states[n]["betas"] = param_group["betas"] # , (0.9, 0.999))
                    param_optimizer_states[n]["eps"] = param_group["eps"] # , 1e-8)
                    # param_optimizer_states[n]["initial_lr"] = param_group.get("initial_lr", 0.0)
                    
        return param_optimizer_states

    def _get_masks(self, outputs, lm_labels, first_only=True):
        """
        Get the masks for correct and incorrect predictions based on the model outputs and labels.
        """
        # Assuming lm_labels is a tensor of shape (batch_size, sequence_length)
        # containing correct token indices at each token position
        # outputs is a tensor of shape (batch_size, sequence_length, vocab_size)
        # where each element is an integer representing the correct prediction
        assert outputs.shape[0] == lm_labels.shape[0], "Outputs and labels must have the same batch size"
        assert outputs.shape[1] == lm_labels.shape[1], "Outputs and labels must have the same sequence length"

        # this implementation is garbage but works for now
        pred_pos_mask = torch.zeros_like(outputs, dtype=torch.bool)
        pred_neg_mask = torch.ones_like(outputs, dtype=torch.bool)
        for batch_idx in range(outputs.shape[0]):
            for seq_idx in range(outputs.shape[1]):
                if lm_labels[batch_idx, seq_idx] != -100:
                    # mark the position of the correct prediction as True in pred_pos_mask (pred happens one pos before token pos)
                    pred_pos_mask[batch_idx, seq_idx-1, lm_labels[batch_idx, seq_idx]] = True
                    pred_neg_mask[batch_idx, seq_idx-1, lm_labels[batch_idx, seq_idx]] = False
                    if first_only:
                        break

        return pred_pos_mask, pred_neg_mask

    def compute_regularization_scores(self):
        """
        Compute regularization scores for the model.
        This method should be implemented to compute the regularization scores using the model.
        """
        # Implement regularization score computation logic here
        scores, accumulated = self.wrapped_model.compute_regularization_scores()
        # Extract scores from the model
        # scores = self.wrapped_model.scores
        # accumulated = self.wrapped_model.accumulated
        return scores, accumulated

    def compute_first_moment_scores(self):
        """
        Compute first moment scores for the model.
        This method should be implemented to compute the first moment scores using the model.
        """
        # Implement first moment score computation logic here
        scores, accumulated = self.wrapped_model.compute_first_moment_scores()
        # Extract scores from the model
        # scores = self.wrapped_model.scores
        # accumulated = self.wrapped_model.accumulated
        return scores, accumulated
    
    def reset_gradients(self):
        """
        Reset the gradients for the model.
        This method should be implemented to reset the gradients using the model.
        """
        # Implement gradient reset logic here
        self.wrapped_model.reset_gradients()

    def reset_layers(self):
        """
        Reset the layers of the model.
        This method should be implemented to reset the layers using the model.
        """
        # Implement layer reset logic here
        self.trigger_function_across_layers(self.wrapped_model, "reset")

    def compute_test_integral(self, test_instances, mask):
        for i in range(self.num_steps):
            # compute test grads
            self.wrapped_model.eval()
            out = self.wrapped_model.forward(**test_instances,
                                                all_outs=True, 
                                            )
            
            if self.expl_type == "loss":
                # apply loss
                out = [out[0]] 
            elif self.expl_type == "outputs":
                out = out[1]
            
            sum_corr = torch.sum(out[mask], dim=None)

            # sum_corr.backward()
            self.accelerator.backward(sum_corr)
            
            # update feature map (call this for all replicated layers)
            self.wrapped_model.update_test_gradient_integral()
            self.wrapped_model.reset_gradients()  # reset gradients after each step

    def compute_influence_scores(self):
        """
        Compute influence scores for the training instances.
        This method should be implemented to compute the influence scores using the model and train_batches.
        """
        # Implement influence score computation logic here
        if not os.path.exists(self.score_path):
            os.makedirs(self.score_path)

        # we loop over all combinations of train and test batches, as well as outputs (for the test feature map)
        # avoid re-computing test integrals, recompute train grads?

        test_counter = -1

        for test_type, test_instances in tqdm(self.test_instances.items()):

            test_counter += 1
            if self.test_batch_limit > 0 and test_counter >= self.test_batch_limit:
                print(f"Reached test batch limit of {test_counter}, stopping further computations.")
                break

            if self.expl_type == "loss":
                # we are decomposing the loss, so out should be the loss
                # apply loss to out
                out_classes = [("loss", 0)] # fix later

            elif self.expl_type == "outputs":
                raise NotImplementedError("Output decomposition is currently not implemented.")
                # we're decomposing outputs (using addition of correct+incorrect trick)
                # iter over correct and incorrect outputs (for test map)
                # this is the most inefficient way to get the out dims but hey
                # self.wrapped_model.eval()
                # out = self.wrapped_model.forward(**test_instances, all_outs=True)[1]  # outputs
                # correct_mask, incorrect_mask = self._get_masks(out, test_instances["lm_labels"])
                # out_classes = [("correct", correct_mask), ("incorrect", incorrect_mask)]

            for corr, mask in out_classes:
                ################ TEST FEATURE MAP INTEGRAL (EXPENSIVE) ################################
                # compute the test integral 
                self.compute_test_integral(test_instances, mask)
                
                ################ END OF TEST FEATURE MAP INTEGRAL (EXPENSIVE) ################################
  
                self.wrapped_model.reset_alpha()  # train grads are computed with alpha=0 (i.e. model0), so reset alpha to 0
                # for each train type, compute influence scores

                for train_type, train_batch in tqdm(self.train_instances.items()):
                    # compute train grads
                    # self.wrapped_model.eval()
                    self.wrapped_model.reset_gradients()
                    self.wrapped_model.train()

                    out_train = self.wrapped_model.forward(**train_batch, all_outs=True, 
                                                           #num_items_in_batch=512,
                                                           alpha=0) # todo: get actual batch size

                    if self.expl_type == "loss":
                        # apply loss
                        out_train = [out_train[0]]
                        train_mask = 0
                    elif self.expl_type == "outputs":
                        out_train = out_train[1]
                        train_mask, _ = self._get_masks(out_train, train_batch["lm_labels"])

                    # print non-zero elements in train_mask
                    # print indices of non zero
                    sum_corr = torch.sum(out_train[train_mask], dim=None)
                    # sum_corr.backward()
                    self.accelerator.backward(sum_corr)
                
                    # self.wrapped_model.log_train_gradient()

                    # recursively compute and extract scores
                    scores, accumulated = self.compute_and_extract_scores(self.wrapped_model, "", scores={})

                    # save scores to disk
                    self.log_scores(scores, accumulated, train_type, test_type, corr)
                    # print(test_type, train_type, corr, "done.")

                self.wrapped_model.reset_gradients()
                # compute regularization scores
                scores, accumulated = self.compute_regularization_scores()
                self.log_scores(scores, accumulated, "regularization", test_type, corr)

                scores, accumulated = self.compute_first_moment_scores()
                self.log_scores(scores, accumulated, "first_moment", test_type, corr)

                self.wrapped_model.reset()

        print("Computing influence scores completed.")
        if self.compute_sim_scores:
            print("Computing similarities between influence scores...")
            self.compute_similarities()
            print("Similarity computation completed.")
    
    def _get_instance_length(self, instances):
        """
        Get the length of the instance. E.g., number of tokens in the batch.

        Might be scenario specific depending on how loss is normalized.
        """
        return instances["input_ids"].shape[1]

    def log_scores(self, scores, accumulated, train_type, test_type, corr, bf16=True):
        """
        Save the influence scores to disk.
        This method should be implemented to save the influence scores using the model and train_batches.
        """
        # remove number suffixes from train type
        train_type_normalized = re.sub(r"_\d+$", "", train_type)
        key = (train_type_normalized, test_type, corr)

        if key not in self.all_scores.keys():
            if self.save_all_scores or self.compute_sim_scores:
                self.all_scores[key] = deepcopy(scores.detach().cpu().to_dict())

        elif self.save_all_scores:
                for par_name, par_scores in scores.items():
                    self.all_scores[key][par_name] += par_scores.detach().cpu()

        
        if key not in self.acc_scores.keys():
            self.acc_scores[key] = deepcopy(accumulated)
            self.acc_scores[key]["num_vals"] = 1

        else:
            # sum sum scores
            for par_name, par_acc in accumulated.items():
                self.acc_scores[key][par_name]["sum"] += par_acc["sum"]
                self.acc_scores[key][par_name]["mean"] = self.acc_scores[key][par_name]["mean"] + par_acc["mean"]
                self.acc_scores[key][par_name]["std"] = self.acc_scores[key][par_name]["std"] + par_acc["std"]
                self.acc_scores[key]["num_vals"] += 1

    def compute_similarities(self):
        """
        Compute similarities between the different influence scores by layer across groups.
        """
        sim_scores = {}

        torch.cuda.empty_cache()

        for i, ((train_type, test_type, corr), scores) in tqdm(enumerate(self.all_scores.items())):
            for j, ((train_type2, test_type2, corr2), scores2) in enumerate(self.all_scores.items()):
                if i >= j:
                    continue
                sims = {}
                # compute similarity between scores and scores2
                for par_name, par_scores1 in scores.items():
                    par_scores2 = scores2[par_name]
                    # put both on gpu
                    par_scores1["scores"] = par_scores1["scores"]#.to(self.accelerator.device)
                    par_scores2["scores"] = par_scores2["scores"]#.to(self.accelerator.device)

                    sums = torch.sum(par_scores1["scores"] * par_scores2["scores"], dim=None)
                    norms = torch.norm(par_scores1["scores"]) * torch.norm(par_scores2["scores"])

                    sims[par_name] = ((sums / (norms + 0.000001)).detach().clone().item(), norms.detach().clone().item())
                    
                    if self.layer_mapping is not None:
                        for k, v in self.layer_mapping.items():
                            # sum up all scores in the layer
                            if par_name in v:
                                if k not in sims:
                                    sims[k] = [0.0, 0.0]
                                sims[k][0] += sums.detach().clone().item()
                                sims[k][1] *= norms.detach().clone().item()

                if self.layer_mapping is not None:
                    for k, v in self.layer_mapping.items():
                        if k in sims:
                            sims[k] = sims[k][0] / (sims[k][1] + 0.000001)

                sim_scores[((train_type, test_type, corr), (train_type2, test_type2, corr2))] = deepcopy(sims)
        self.sim_scores = sim_scores

        torch.cuda.empty_cache()

            
    def save_scores_to_disk(self):
        """
        Save all computed scores to disk.
        This method should be implemented to save the influence scores using the model and train_batches.
        """
        if not os.path.exists(self.score_path):
            os.mkdir(self.score_path)

        if self.save_all_scores and self.save_all_scores:
            torch.save(self.all_scores, os.path.join(self.score_path, "scores.pt"))
        
        if self.acc_scores is not None:
            torch.save(self.acc_scores, os.path.join(self.score_path, "accumulated_scores.pt"))

        if self.sim_scores is not None:
            torch.save(self.sim_scores, os.path.join(self.score_path, "similarity_scores.pt"))
        else:
            print("No similarity scores computed. Skipping saving similarity scores.")

        # store param shapes
        shapes = {}
        for par_name, par in self.wrapped_model.model0.named_parameters():
            shapes[par_name] = list(par.shape)
            
        with open(os.path.join(self.score_path, "parameter_shapes.json"), "w") as f:
            json.dump(shapes, f, indent=4)
        

        if self.acc_layer_scores is not None:
            torch.save(self.acc_layer_scores, os.path.join(self.score_path, "accumulated_layer_scores.pt"))
        
    def compute_and_extract_scores(self, wrapped_model, rec_name, scores):
        """
        Compute and extract influence scores from the model.
        This method should be implemented to compute the influence scores using the model.
        """
        wrapped_model.compute_influence_score()
        scores = wrapped_model.scores
        accumulated = wrapped_model.accumulated
        return scores, accumulated


    def verify_loss_decomposition(self):
        # sum up scores and check accuracy of decomposition vs actual output

        if self.expl_type != "loss":
            print("Explanation type is not loss. This function is likely not implemented for the current configuration.")

        results = {}

        regs = []
        kers = []
        fms = []
        actuals = []

        counter = -1 

        for test_type, test_instances in self.test_instances.items():

            counter += 1
            if self.test_batch_limit > 0 and counter >= self.test_batch_limit:
                print(f"Reached test batch limit of {counter}, stopping further computations.")
                break

            torch.cuda.empty_cache()

            print(test_type)

            indices = [key for key in self.acc_scores.keys()
                        if key[1] == test_type] 
            # print(indices)
            # print(self.all_scores.keys())
            # type_scores = [self.all_scores[key] for key in indices] 

            reg_scores = []
            ker_scores = []
            fm_scores = []

            for idx in indices:
                for k, v in self.acc_scores[idx].items():
                    if k in {"num_vals"}:
                        continue
                    if idx[0] == "regularization":
                        reg_scores.append(v["sum"])
                    elif idx[0] == "first_moment":
                        fm_scores.append(v["sum"])
                    else:
                        ker_scores.append(v["sum"])

            with torch.no_grad():
                torch.cuda.empty_cache()
                self.wrapped_model.eval()
                loss_pre = self.wrapped_model.model0(**test_instances).loss * self.test_loss_scaling
                loss_post = self.wrapped_model.model1(**test_instances).loss * self.test_loss_scaling

            # print('num test samples: ', test_instances['target_ids'].shape[0])    
            print(loss_pre, loss_post)
            torch.cuda.empty_cache()
            actual = loss_post - loss_pre
            print('actual change', (loss_post - loss_pre).item())

            reg_sums = []
            for ts in reg_scores:
                # print(ts)
               reg_sums.append(ts)

            s_reg = sum(reg_sums)

            ker_sums = []
            for ts in ker_scores:
                # print(ts)
                ker_sums.append(ts)

            s_ker = sum(ker_sums)

            fm_sums = []
            for ts in fm_scores:
                # print(ts)
                fm_sums.append(ts)

            s_fm = sum(fm_sums)

            print("reg", s_reg)
            print("ker", s_ker)
            print("fm", s_fm)

            # s = s_reg + s_ker + s_fm

            s = s_reg + s_fm + s_ker

            print('estimated', s)

            delta = loss_post - loss_pre - s

            # print("r/reg", (actual - s_fm) / s_reg )

            print('delta', delta.item())

            print('actual/estimated', (actual / s).item())

            print('estimated/actual', (s / actual).item())

            results[test_type] = {"loss_pre": loss_pre.item(), 
                                  "loss_post": loss_post.item(), 
                                  "actual_change": (loss_post - loss_pre).item(),
                                  "reg_sum": s_reg,
                                  "ker_sum": s_ker,
                                  "fm_sum": s_fm,
                                  "estimated_change": s, 
                                  "delta": delta.item(),
                                  "actual/estimated": (actual / s).item(),
                                  "estimated/actual": (s / actual).item(),
                                  "len(ker)": len(ker_scores),
                                  "len(fm)": len(fm_scores),
                                  "len(reg)": len(reg_scores),
                                  }

            regs.append(s_reg)
            kers.append(s_ker)
            fms.append(s_fm)
            actuals.append(actual.item())

            tkers = torch.tensor(kers)
            tfms = torch.tensor(fms)
            tactuals = torch.tensor(actuals)
            tregs = torch.tensor(regs)

        print("Deltas:", [results[k]['delta'] for k in results.keys()])

        X = torch.stack([tregs, tkers, tfms], dim=1)
        y = tactuals

        w = np.linalg.lstsq(X, y, rcond=None)[0]

        print("Weights:", w)
        print("Predictions:", X @ w)
        print("Actuals:", y)
        print()
        results["reg_ker_fm_weights(reg,ker,fm)"] = w.tolist()

        X2 = X = torch.stack([tregs, tkers], dim=1)
        y2 = tactuals - tfms 
        w2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        print("Weights without first moment:", w2)
        print("Predictions without first moment:", X2 @ w2)
        print("Actuals:", y2)
        print()
        results["reg_ker_weights_no_fm(reg,ker)"] = w2.tolist()

        return results, regs, kers, fms, actuals
    


class Explainer(StepExplainer):
    """
    This is a sample class inheriting from StepExplainer and describing
    the methods and fields that need to be implemented in inheriting classes.

    See gpt2_explainer.py for a working example.
    """
    def __init__(self, 
                 checkpoint_path,
                 ## your additional arguments here
                 test_instances=None,
                 save_all_scores=False,
                 ignore_train_types=True,
                 expl_type="outputs",  # alternatively, 'loss' will be implemented
                 num_steps=10,
                 debug=False,):
        super().__init__(checkpoint_path,
                         test_instances,
                         save_all_scores,
                         ignore_train_types,
                         expl_type,
                         num_steps,
                         debug)
        ## your additional initialization code here
                 

    def load_checkpoint(self, path=None):
        """
        Load checkpoints.

        This method should be implemented to load the model, optimizer, scheduler,
        training instances, and dataloader from the checkpoint.
        """
        # Implement checkpoint loading logic here
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # a dict of 'train_type' -> batch of instances
        self.train_instances = None

        # current learning rate for each parameter group
        # get from scheduler
        self.lr = None

        # number of steps performed before the current one that is to be explained
        # i.e. 's' in the formulas
        self.step_id = None

        # if you need to tie layers together, you can use this map
        self.tied_layer_map = None

        # optional mapping of layer names to groups of layer names for similarity computation
        self.layer_mapping = None

        # clipping factor for gradients (from the current update step)  
        # self.clip_norm_factor = 1.0

    def compute_step_model(self):
        """
        Compute the step model.

        This method should be implemented to create the model before and after the update step
        and wrap them in a ReplicatedModel (which is the backbone of the param interpolations
        and scorecomputations).
        """
        # Implement step model computation logic here
        
        ### compute update step ###
        updated_model = None  # model after the update step

        # if you have implemented lod_checkpoint correctly, this helps you get the correct wrapped model
        # i.e. an instance of ReplicatedModel
        self.wrapped_model = self._wrap_model("", self.model, updated_model)

        