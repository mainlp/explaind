import torch
from explaind.modulo_model.data import ClassificationDataset
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
        print(data, targets, input_emb.shape)
        get_high_and_low_sim_samples(data=data, targets=targets, input_emb=input_emb, num_samples=10, num_highest=10, num_lowest=10, result_path=layer_dir + f"{k}_high_low.txt", verbose=False)
        
        for sorting in ["input_sum"]:
            plot_sim_matrices_with_different_sorting(
                input_emb, data, targets, layer_dir, sort_by=sorting, name=k
            )


experiment_path = "results/modulo_train_epk/"

train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
train_loader_s = torch.load(experiment_path + "train_loader_sub.pt")

# load step matrices
kernel_path = 'results/modulo_train_epk/step_matrices/'

step_kernels = []  # will be list of dicts of layer_name -> torch tensors of shape (num_val, num_params) = (2000, num_params)

# takes about 3 minutes (loading 35 GB worth of tensors)
for i in tqdm(range(0, 1939, 20)):
    step_kernels.append(torch.load(kernel_path + f"param_kernel_step_{i}.pt", map_location="cpu", weights_only=True))


print(f"Loaded {len(step_kernels)} step kernels")

current_kernel = None

# plot the step kernels like un component_attribution.py

# convert subset dataloader into dataset
X = []
y = []
for sample in train_loader_s.dataset:
    print(sample)
    X.append(sample[0])
    y.append(sample[1][0])

X = torch.stack(X)
y = torch.stack(y)

# make dataset with data and target fields
dataset = ClassificationDataset(X, y)

loader = torch.utils.data.DataLoader(dataset, batch_size=400, shuffle=False)


plot_every = 20
for i, step_kernel in tqdm(enumerate(step_kernels)):
    print(f"Plotting step kernel {i}")
    plot_dir = f"results/modulo_train_epk/plots/step_kernels/step_kernels_pos_{i*20}/"

    # if current_kernel is None:
    #     current_kernel = step_kernel
    # else:
    #     for layer_name, kernel in step_kernel.items():
    #         current_kernel[layer_name] += kernel

    current_kernel = step_kernel

    dir_exists = os.path.exists(plot_dir)

    if (i * 20) % plot_every == 0 and i > 0 and not dir_exists:
        os.makedirs(plot_dir, exist_ok=True)
        make_all_plots(current_kernel, plot_dir, loader)



# plot the final kernel by loading param_kernel
final_kernel = torch.load(experiment_path + f"param_kernel.pt", map_location="cpu", weights_only=True)
plot_dir = "results/modulo_train_epk/plots/final_param_kernel/"
os.makedirs(plot_dir, exist_ok=True)

# sum up the output dims
for k, v in final_kernel.items():
    final_kernel[k] = v.sum(dim=1).sum(dim=1)

# plot the final kernel
make_all_plots(final_kernel, plot_dir, loader)