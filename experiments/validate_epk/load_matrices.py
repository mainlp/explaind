import torch
from explaind.modulo_model.model import SingleLayerTransformerClassifier
import json
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

experiment_path = "results/modulo_val_results/"

train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

config = json.load(open(experiment_path + "config.json", 'r'))

# load model
base_model = SingleLayerTransformerClassifier(config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])

# load model checkpoints (to be found in model_path.checkpoints which is a list of state_dicts)
# model_path = ModelPath(config["model_path"], config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])

# dict of layer_name -> torch tensors of shape (num_val, 1, out_dim, num_train) = (2000, 1, 115, 4000)
# i.e. per module/layer the accumulated kernel values over all params and steps
# kernel_matrices = torch.load(experiment_path + "kernel_matrices.pt", map_location=device)

# 76 GB, make sure you have enough memory, takes 77.36 tgt with torch
# this is layer_name -> torch tensors of shape (num_val, 1, out_dim, num_params) = (2000, 1, 115, num_params)
# param_kernels = torch.load(experiment_path + "param_kernel.pt", map_location=device)

kernel_path = 'results/modulo_val_results_0.01_w/step_matrices/'

step_kernels_pos = []  # will be list of dicts of layer_name -> torch tensors of shape (num_val, num_params) = (2000, num_params)
step_kernels_neg = []

# takes about 3 minutes (loading 35 GB worth of tensors)
for i in tqdm(range(0, 1939, 20)):
    step_kernels_pos.append(torch.load(kernel_path + f"param_kernel_step_{i}_correct.pt", map_location=device, weights_only=True))
    step_kernels_neg.append(torch.load(kernel_path + f"param_kernel_step_{i}_incorrect.pt", map_location=device, weights_only=True))

print(f"Loaded {len(step_kernels_pos)} pos step kernels.")
print(f"Loaded {len(step_kernels_neg)} neg step kernels.")

# note: If you want to recover the original param shapes, you can use the following code:
print(f"Recovered original param shapes for pos step kernels:")
reshaped_step_kernels_pos = []
v = True
for kern in step_kernels_pos:
    for layer_name, kernel in kern.items():
        before_shape = kernel.shape
        param_shape = base_model.state_dict()[layer_name].shape
        kern[layer_name] = kernel.reshape(kernel.shape[0], *param_shape)
        if v:
            print(f"Layer {layer_name}: {before_shape} -> {kern[layer_name].shape}")
    v = False
    reshaped_step_kernels_pos.append(kern)
