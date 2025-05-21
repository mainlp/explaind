import torch

from explaind.modulo_model.model import SingleLayerTransformerClassifier
from explaind.model_paths import ModelPath
from explaind.modulo_model.loss import RegularizedCrossEntropyLoss

import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import os
from tqdm import tqdm
import random


device = "cuda" if torch.cuda.is_available() else "cpu"

def inject_previous_step(model_path, step_from, step_to, ignore_layers=["decoder.weight", "input_emb.weight", "linear2.weight", "linear2.bias"]):
    """
    Injects the previous step's parameters into the current step's parameters.
    """

    base_model = deepcopy(model_path.model)
    to_params = deepcopy(model_path.checkpoints[step_to])

    # get step parameters
    if step_from == "random":
        step_params = {}
        for k, v in to_params.items():
            step_params[k] = torch.randn_like(v) * 0.1
        print("Randomly initialized step params")
    else:
        step_params = deepcopy(model_path.checkpoints[step_from])

    injected_params = {}

    for k, v in step_params.items():
        if k not in ignore_layers:
            print("Taking", k, "from step params at", step_from)
            injected_params[k] = step_params[k]
        else:
            print("Taking", k, "from to params at", step_to)
            injected_params[k] = to_params[k]

    # set the model's parameters to the injected parameters
    base_model.load_state_dict(injected_params)

    base_model.eval()

    step_model = deepcopy(model_path.model)
    step_model.load_state_dict(step_params)

    return base_model, step_model


def eval_injected_model(base_model, inj_model, step_model, val_loader):
    inj_logits = []
    pred_logits = [] 
    step_logits = []   
    pred_correct = 0
    inj_pred_acc = 0
    step_pred_acc = 0
    total = 0

    # compare predictions
    for i, (x, y) in tqdm(enumerate(val_loader)):
        x = x.to(device)
        y = y.to(device)

        # get predictions
        with torch.no_grad():
            pred_log = base_model(x)
            inj_pred_log = inj_model(x)
            step_pred_log = step_model(x)

        pred_logits.append(pred_log)
        inj_logits.append(inj_pred_log)
        step_logits.append(step_pred_log)

        # eval
        pred = torch.argmax(pred_log, dim=1)
        inj_pred = torch.argmax(inj_pred_log, dim=1)
        step_pred = torch.argmax(step_pred_log, dim=1)

        pred_correct += (pred == y).sum()
        inj_pred_acc += (inj_pred == y).sum()
        step_pred_acc += (step_pred == y).sum()
        total += y.size(0)

    pred_acc = pred_correct / total
    inj_acc = inj_pred_acc / total
    step_acc = step_pred_acc / total
    print(f"Pred acc: {pred_acc}")
    print(f"Inj acc: {inj_acc}")
    print(f"Step acc: {step_acc}")

    combined_logits = torch.cat(pred_logits, dim=0)
    inj_combined_logits = torch.cat(inj_logits, dim=0)
    step_combined_logits = torch.cat(step_logits, dim=0)

    print(f"Combined logits shape: {combined_logits.shape}")

    kl_div = torch.nn.KLDivLoss(reduction='mean', log_target=True)
    kl_div_inj = kl_div(torch.log_softmax(combined_logits, dim=1), torch.log_softmax(inj_combined_logits, dim=1))
    kl_div_step = kl_div(torch.log_softmax(combined_logits, dim=1), torch.log_softmax(step_combined_logits, dim=1))

    print(f"KL div inj: {kl_div_inj}")
    print(f"KL div step: {kl_div_step}")

    return pred_acc.cpu().item(), inj_acc.cpu().item(), step_acc.cpu().item(), kl_div_inj.item(), kl_div_step.item(), inj_combined_logits, step_combined_logits


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=200,
    lr=0.001,
    weight_decay=1.0,
    log_path="results/modulo_val_results_wreg/pipeline_train/",
    seed=42,
):
    os.makedirs(log_path, exist_ok=True)

    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # setup cross entropy loss and optimizer
    loss_fct = RegularizedCrossEntropyLoss(alpha=0.0, p=2, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))  # weight decay needed for grokking acording to Huang et al.
    
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    # train model
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_reg = 0
        n = 0
        m = 0

        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model.forward(x)
            l, reg = loss_fct(output, y, params=model.parameters(), output_reg=True)

            l.backward()
            optimizer.step()

            # log checkpoint values we need for epk prediction

            train_loss += l.item()
            _, predicted = torch.max(output, 1)
            train_accuracy += (predicted == y).sum().item()
            train_reg += reg.item()
            n += len(y)
            m += 1

        train_loss /= m
        train_accuracy /= n
        train_reg /= m
        
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
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
            
        if train_accuracy == 1.0 and val_accuracy == 1.0:
            break

    # return model, checkpoints, optimizer, loss_fct
    model_parts = {}
    model_parts["model"] = model
    # model_parts["weights"] = checkpoints
    model_parts["optimizer"] = optimizer
    model_parts["loss_fct"] = loss_fct

    # save a config with hyperparams etc
    config = {
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "tensorboard_log_path": log_path,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_reg": train_reg,
        "val_reg": val_reg,
        "final_epoch": epoch,
        "epochs_trained": len(train_accuracies),
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    with open(log_path + f'config_{seed}.json', 'w') as f:
        json.dump(config, f)

    return train_accuracies, val_accuracies, train_losses, val_losses, model_parts, num_epochs


if __name__ == "__main__":
    experiment_path = "results/modulo_val_results_wreg/"
    plot_path = "results/modulo_val_results_wreg/plots/inject_pipeline_train/"

    os.makedirs(plot_path, exist_ok=True)

    train_loader = torch.load(experiment_path + "train_loader_N=4000.pt")
    val_loader = torch.load(experiment_path + "val_loader_N=4000.pt")

    config = json.load(open(experiment_path + "config.json", 'r'))

    # load model
    base_model = SingleLayerTransformerClassifier(config["n_token"], d_model=config["hidden_size"], nhead=4, dim_mlp=config["dim_mlp"], dropout=config["dropout"])

    # load model checkpoints (to be found in model_path.checkpoints which is a list of state_dicts)
    model_path = ModelPath(base_model,                  
                        device=device, 
                        checkpoint_path=experiment_path + "model_checkpoint.pt",
                        features_to_cpu=False,
                        parallelize_over_targets=True)
    
    # swap pipeline into model at ep 700
    # step_from = 1100
    # step_to = 1700


    num_epochs = []

    outer = ["decoder.weight", "input_emb.weight", "linear2.weight", "linear2.bias"]
    not_outer = [k for k in model_path.checkpoints[0].keys() if k not in outer]

    # for step_from in range(0, 1900, 50):
    for seed in [0, 1, 2, 3, 4]:
        for step_to in range(1939, 250, -100):
            
            if not os.path.exists(plot_path + f"pipeline_train_rand_{step_to}/config_{seed}.json"):
                base_model, step_model = inject_previous_step(model_path, "random", step_to, ignore_layers=["decoder.weight", "input_emb.weight", "linear2.weight", "linear2.bias"])

                res = train_model(
                    base_model,
                    train_loader,
                    val_loader,
                    device,
                    epochs=501,
                    lr=0.001,
                    seed=seed,
                    weight_decay=4.0,
                    log_path=plot_path + f"pipeline_train_rand_{step_to}/",
                )

                num_epochs.append(res)
            
            if not os.path.exists(plot_path + f"outer_train_rand_{step_to}/config_{seed}.json"):
                base_model, step_model = inject_previous_step(model_path, "random", step_to, ignore_layers=not_outer)

                res = train_model(
                    base_model,
                    train_loader,
                    val_loader,
                    device,
                    epochs=501,
                    lr=0.001,
                    seed=seed,
                    weight_decay=4.0,
                    log_path=plot_path + f"outer_train_rand_{step_to}/",
                )

                num_epochs.append(res)


            if True: # not os.path.exists(plot_path + f"att_train_rand_{step_to}/config_{seed}.json"):
                
                base_model, step_model = inject_previous_step(model_path, "random", step_to, ignore_layers=["decoder.weight", "input_emb.weight", "linear2.weight", "linear2.bias", "linear1.weight", "linear1.bias"])

                res = train_model(
                    base_model,
                    train_loader,
                    val_loader,
                    device,
                    epochs=501,
                    lr=0.001,
                    seed=seed,
                    weight_decay=4.0,
                    log_path=plot_path + f"att_train_rand_{step_to}/",
                )

                num_epochs.append(res)