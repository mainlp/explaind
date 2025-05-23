
[![PyPI - Python](https://img.shields.io/badge/python-v3.12+-blue.svg)](https://pypi.org/project/explaind/0.0.1/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/mainlp/explaind/blob/main/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/explaind)](https://pypi.org/project/explaind/0.0.1/)
[![arXiv](https://img.shields.io/badge/TODO-<COLOR>.svg)](https://github.com/mainlp/explaind)

# ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior

![Screenshot 2025-05-21 at 08 44 04](https://github.com/user-attachments/assets/077ef1a3-2e37-4e3a-8434-cf6673369122)

This repository is the official implementation of [Grokking ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior](https://github.com/mainlp/path_kernels). To jump directly to the experiments, go to [`experiments/`](https://github.com/mainlp/explaind/tree/main/experiments).


## Requirements

### Manual installation

We ran all our experiments in python version `3.12.7`. You can use `conda` to create a fresh environment first, clone our repository, and install the necessary packages using `pip`.

```
conda create -n explaind python=3.12
conda activate explaind

git clone git@github.com:mainlp/explaind.git

pip install torch torchvision numpy tqdm tensorboard pandas
```

If you also want to recreate the plots shown in the paper, you additionally need the following packages:

```setup
pip install plotly umap_learn
```

Alternatively, you can also install from the requirements file:

```setup
pip install -r requirements.txt
```

### PyPi installation

We also provide a PyPi package, which you can directly install with pip:

```setup
pip install explaind
```

Note, that this will only install the code contained in `explaind/`. To replicate our experiments, you will still need to clone this repository.

## Training models with history

If you want to apply ExPLAIND to your own model, you need to retrain it tracking relevant parts of the training process by using the wrappers provided in this repository. Note that depending on model size full history tracking can become very expensive. We're currently working on a solution for allowing cheaper partial tracking. For example, the training process of the modulo addition model includes the following additions:

```python
model = SingleLayerTransformerClassifier().to(device)
# wrap into path wrapper
model = ModelPath(model, device=device, checkpoint_path="model_checkpoint.pt")
loss_fct = RegularizedCrossEntropyLoss(alpha=alpha, p=reg_pow, device=device)
optimizer = AdamWOptimizerPath(model, checkpoint_path="optimizer_checkpoint.pt")
data_path = DataPath(train_loader, checkpoint_path=checkpoint_path + "data_checkpoint.pt", overwrite=True, full_batch=False)

for epoch in range(epochs):
    for batch in data_path.dataloader:
        x, y = data_path.get_batch(batch)
        optimizer.zero_grad()
        output = model.forward(x)
        l, reg = loss_fct(output, y, params=model.parameters(), output_reg=True)
        l.backward()
        optimizer.step()

        # log checkpoint values we need for epk prediction
        model.log_checkpoint() 
        optimizer.log_checkpoint()

# save the checkpoints to disk at the locations defined above
# loading these later, we can compute the EPK reformulation of the model
optimizer.save_checkpoints()
model.save_checkpoints()
data_path.save_checkpoints()
```

For actual, executable training scripts, you can have a look at `experiments/train_models/modulo_model.py` and `experiments/train_models/cifar2_model.py`.

## Getting EPK predictions and kernel accumulations

Once you have the history of the training run you want to explain, you can load them into the EPK module and compute the prediction of the reformulated model as follows:

```python
epk = ExactPathKernelModel(
    model=model,  # wrapper from before
    optimizer=optimizer,  # wrapper from before
    loss_fn=RegularizedCrossEntropyLoss(alpha=0.0),
    data_path=data_path,  # wrapper from before
    integral_eps=0.01,  # 1/eps = 1/0.01 = 100 integral steps
    evaluate_predictions=True,
    keep_param_wise_kernel=True,
    param_wise_kernel_keep_out_dims=True,
)

# make batch size small enough so you don't run OOM
val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=100, shuffle=False)

preds = []
for i, (X, y) in enumerate(val_loader):
    torch.cuda.empty_cache()
    X = X.to(device)
    y = y.to(device)
    pred = epk.predict(X, y_test=y, keep_kernel_matrices=True)
    preds.append((i, pred, y))
```

Note that there are different settings for which (accumulated) slices of the kernel to store during the prediction. Depending on your choices there, runtimes can vary greatly because of GPU I/O and extra matrix computations involved. For the complete respective valiadtion scripts, consider giving `experiments/validate_epk/` a look.

## Experiments, ablations, and plots

Besides further instructions on how to reproduce the experiments in our paper, the `experiments/` folder contains all the scripts to run additional experiments, ablations, and generate plots. Any checkpoints, plots or other artifacts will be stored in `results/` by default.


## Contributing

We publish this repository under MIT license and welcome anybody who wants to contribute. If you have a question or an idea, feel free to reach out to Florian (feichin[at]cis[dot]lmu[dot]de) or simply start a pull request/issue.

If you want to use our code for your own projects, please consider citing our work:

```
TODO
```

