[![PyPI - Python](https://img.shields.io/badge/python-v3.12+-blue.svg)](https://pypi.org/project/explaind/0.0.2/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/mainlp/explaind/blob/main/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/explaind)](https://pypi.org/project/explaind/0.0.2/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.20076-b31b1b.svg)](https://arxiv.org/abs/2505.20076)
[![ICML](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/virtual/2026/poster/66073)

![Screenshot 2026-06-30 at 15 35 19](https://github.com/user-attachments/assets/6d900f3b-4ee2-49d3-bded-063806bd3519)

# ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior

This repository is the official implementation of [ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior](https://icml.cc/virtual/2026/poster/66073). ExPLAIND reformulates trained models through the exact path kernel (EPK) to attribute model behavior to training data, model components, and training dynamics.

The core package lives in [`explaind/`](explaind/). Reproducibility scripts for the ICML paper live in [`experiments/`](experiments/), and LLM-scale experiments live in [`scaled_experiments/`](scaled_experiments/).

This repository continues to be worked on. If you encounter any issues please feel free to report.

## Requirements

We ran the experiments with Python `3.12.7`.

### Manual installation

```bash
conda create -n explaind python=3.12
conda activate explaind

git clone git@github.com:mainlp/explaind.git
cd explaind
pip install -r requirements.txt
```

If you only need the package code from `explaind/`, you can install the PyPI release:

```bash
pip install explaind
```

The PyPI package does not include all experiment scripts and artifacts. Clone this repository to reproduce paper experiments.

Optional plotting dependencies are included in `requirements.txt`. The scaled LLM experiments additionally depend on packages used by the external training and checkpoint-conversion stack; install them with `pip install -r requirements-scaled.txt`.

## Repository layout

- [`explaind/`](explaind/): package code for history tracking, EPK prediction, and accelerated StepExplainer experiments.
- [`experiments/`](experiments/): small-scale paper experiments, ablations, and plotting scripts.
- [`scaled_experiments/`](scaled_experiments/): EuroLLM/LLM-scale scoring scripts and data-batch utilities.
- [`tests/`](tests/): unit tests for the StepExplainer and LLaMA utilities.

## Training models with history

To apply ExPLAIND to your own model, retrain it while tracking the parts of the training process needed by the EPK. Full history tracking can be expensive for large models; the scaled experiments use a step-wise variant for this reason.

The modulo-addition training script illustrates the required wrappers:

```python
model = SingleLayerTransformerClassifier().to(device)
model = ModelPath(model, device=device, checkpoint_path="model_checkpoint.pt")
loss_fct = RegularizedCrossEntropyLoss(alpha=alpha, p=reg_pow, device=device)
optimizer = AdamWOptimizerPath(model, checkpoint_path="optimizer_checkpoint.pt")
data_path = DataPath(train_loader, checkpoint_path=checkpoint_path + "data_checkpoint.pt", overwrite=True, full_batch=False)

for epoch in range(epochs):
    for batch in data_path.dataloader:
        x, y = data_path.get_batch(batch)
        optimizer.zero_grad()
        output = model.forward(x)
        loss, reg = loss_fct(output, y, params=model.parameters(), output_reg=True)
        loss.backward()
        optimizer.step()

        model.log_checkpoint()
        optimizer.log_checkpoint()

optimizer.save_checkpoints()
model.save_checkpoints()
data_path.save_checkpoints()
```

Executable examples are available in [`experiments/train_models/modulo_model.py`](experiments/train_models/modulo_model.py), [`experiments/train_models/cifar2_model.py`](experiments/train_models/cifar2_model.py), and [`experiments/train_models/mnist10_model.py`](experiments/train_models/mnist10_model.py).

## Getting EPK predictions and ExPLAIND scores

After training with history, load the checkpoints into `ExactPathKernelModel` and run EPK prediction:

```python
epk = ExactPathKernelModel(
    model=model,
    optimizer=optimizer,
    loss_fn=RegularizedCrossEntropyLoss(alpha=0.0),
    data_path=data_path,
    integral_eps=0.01,
    evaluate_predictions=True,
    keep_param_wise_kernel=True,
    param_wise_kernel_keep_out_dims=True,
)

val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=100, shuffle=False)

predictions = []
for i, (x, y) in enumerate(val_loader):
    torch.cuda.empty_cache()
    pred = epk.predict(x.to(device), y_test=y.to(device), keep_kernel_matrices=True)
    predictions.append((i, pred, y))
```

The validation scripts in [`experiments/validate_epk/`](experiments/validate_epk/) show complete CIFAR-2 and modulo-addition EPK pipelines.

## Experiments, ablations, and plots

See [`experiments/README.md`](experiments/README.md) for camera-ready run commands covering:

- model training with history for modulo addition, CIFAR-2, and MNIST;
- EPK prediction and influence-score computation;
- component attribution, pruning, representation-pipeline retraining/swapping, sensitivity, and LDS-style MNIST data attribution;
- plot regeneration from stored `results/` artifacts.

## Scaled experiments

The scaled EuroLLM experiments live in [`scaled_experiments/euro_llm/`](scaled_experiments/euro_llm/). See [`scaled_experiments/README.md`](scaled_experiments/README.md) for:

- how to sample language batches;
- how to compute EuroLLM StepExplainer scores;
- how to verify loss decomposition scores;
- how to integrate a custom model by subclassing `StepExplainer`.

## Development checks

The repository uses `unittest` tests and has a Ruff configuration in `pyproject.toml`.

```bash
python -m unittest discover tests
python -m ruff check explaind experiments scaled_experiments tests
```

If these commands fail because optional developer dependencies are missing, install the missing tools in your local environment before running the checks.

## Camera-ready suggestions and likely missing files

Suggested follow-up work before release:

- Add a small `requirements-dev.txt` containing `ruff`, `pytest` or a documented `unittest` workflow.
- Track any non-vendored external training package needed beyond `requirements-scaled.txt`.
- Add a short artifact manifest describing expected checkpoint and result paths under `results/`, including files that are too large to commit.
- Add example config files or links for `scaled_experiments/llama1b/training/configs/...`, which are referenced by the EuroLLM scripts but are not present in this checkout.
- Add a CI workflow that installs the package, runs unit tests, and runs Ruff on the core package.
- Add a minimal notebook or script that runs an end-to-end modulo experiment on CPU with tiny settings for quick smoke testing.
- Confirm whether generated BLiMP sample files under `results/blimp_scores/` should be published, regenerated, or documented as external artifacts.

## Contributing and citation

We publish this repository under the MIT license and welcome contributions. If you have a question or idea, reach out to Florian (feichin[at]cis[dot]lmu[dot]de) or open a pull request/issue.

If you use this code, please cite:

```bibtex
@inproceedings{
    eichin2026explaind,
    title={Ex{PLAIND}: Unifying Model, Data, and Training Attribution to Study Model Behavior},
    author={Florian Eichin and Yupei Du and Philipp Mondorf and Maria Matveev and Barbara Plank and Michael A. Hedderich},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=7G6x9QTaN4}
}
```
