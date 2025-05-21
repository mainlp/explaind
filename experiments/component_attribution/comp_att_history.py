import torch
import os
from tqdm import tqdm

from experiments.component_attribution.component_attribution import get_embedded_samples, get_high_and_low_sim_samples,plot_sim_matrices_with_different_sorting, concat_embs

device = "cpu"

inclusive_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

def make_all_plots(kernel, plot_dir, loader):
    combinations = {
        "embedding": ("input_emb.weight",),
        "att. encoders": ("head1_encoder.weight", "head2_encoder.weight", "head3_encoder.weight", "head4_encoder.weight"),
        "att. decoders": ("head1_decoder.weight", "head2_decoder.weight", "head3_decoder.weight", "head4_decoder.weight"),
        "linear1": ("linear1.weight", "linear1.bias"),
        "linear2": ("linear2.weight", "linear2.bias"),
        "decoder": ("decoder.weight",),
        "model": ('input_emb.weight', 
                  'head1_encoder.weight', 'head2_encoder.weight', 'head3_encoder.weight', 'head4_encoder.weight', 
                  'head1_decoder.weight', 'head2_decoder.weight', 'head3_decoder.weight', 'head4_decoder.weight', 
                  'linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias', 'decoder.weight')
    }

    for k, v in tqdm(combinations.items()):
        print(k)
        layer_dir = plot_dir + f"{k}/"
        os.makedirs(layer_dir, exist_ok=True)

        # concatenate the embeddings
        emb = concat_embs(kernel, v)
        data, targets, input_emb = get_embedded_samples(emb, loader.dataset, labels="just don't")
        get_high_and_low_sim_samples(data=data, targets=targets, input_emb=input_emb, num_samples=10, num_highest=10, num_lowest=10, result_path=layer_dir + f"{k}_high_low.txt", verbose=False)
        
        for sorting in ["input_sum"]:
            plot_sim_matrices_with_different_sorting(
                input_emb, data, targets, layer_dir, sort_by=sorting, name=k
            )


experiment_path = "results/modulo_val_results/"

train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

# load step matrices
kernel_path = 'results/modulo_val_results/kernel_matrices/'

step_kernels = []  # will be list of dicts of layer_name -> torch tensors of shape (num_val, num_params) = (2000, num_params)
step_regs = []

# takes about 3 minutes (loading 35 GB worth of tensors)
for i in tqdm(range(0, 1939, 50)):
    step_kernels.append(torch.load(kernel_path + f"param_kernel_step_{i}.pt", map_location=device, weights_only=True))
    step_regs.append(torch.load(kernel_path + f"param_reg_kernel_step_{i}.pt", map_location=device, weights_only=True))

print(f"Loaded {len(step_kernels)} step kernels")
print(f"Loaded {len(step_regs)} step regs")

current_kernel = None

# plot the step kernels like un component_attribution.py

for i, step_kernel in tqdm(enumerate(step_kernels)):

    if i*50 not in [200, 250, 500, 1100, 1900]:
        continue
    print(f"Plotting step kernel {i}")
    plot_dir = f"results/modulo_val_results/plots/step_kernels/step_kernels_{i*50}/"

    # need to combine both to have equivalent kernel as in other plots
    combined_step_kernel = {}
    for layer_name, kernel in step_kernel.items():
        combined_step_kernel[layer_name] = kernel 

    # if current_kernel is None:
    #     current_kernel = combined_step_kernel
    # else:
    #     for layer_name, kernel in combined_step_kernel.items():
    #         current_kernel[layer_name] += kernel

    os.makedirs(plot_dir, exist_ok=True)
    make_all_plots(combined_step_kernel, plot_dir, val_loader)