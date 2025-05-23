# Experiments and Plots
In this directory, you can find the code implementing the experiments of the paper. Generally, you want to apply ExPLAIND by first training a model with checkpoints needed to compute the EPK. Second, you would run a prediction as defined by the kernel reformulation presented in the paper. Third, as presented in the paper, you can plot run statistics and run additional experiments or ablations.

If you're trying to run ExPLAIND on your own model, consider giving the respective scripts a look to understand the (minor) changes you have to make to your training script and how to get EPK predictions.

## 1. Model training with history

Since all of our experiments rely on either the trained CNN or modulo model, you first need to train them storing the checkpoints needed to run ExPLAIND subsequently. To do this, run:

```python
python -m experiments/train_models/cifar2_model.py
```

for the CIFAR2 model or

```python
python -m experiments/train_models/modulo_model.py
```

for the modulo addition model.

## 2. Compute predictions of the EPK reformulation (and thus the ExPLAIND influence scores)

We have decided to combine the computation of influence scores as well as model predictions into a single EPK reformulation prediction procedure to avoid redundant computations of the same steps. This of course is inefficient for the accumulated parts of the ExPLAIND explanations. In future revisions of this code, we want to implement methods that avoid materializing unnecessary parts of the kernel to speed up computations and reduce memory requirements.

To compute the predictions, you can run the following scripts:


```python
python -m experiments/validate_epk/cifar2_epk_predictions.py
```

for the CIFAR2 model and for the modulo additon model

```python
python -m experiments/validate_epk/modulo_epk_predictions.py
```

## 3. Run additional experiments and plots

To run the additional experiments on model pruning or Grokking from the paper, run the respective scripts in the according subfolders. To generate the corresponding plots, there is usually a plot script right where the experiment is located. Namely, this directory contains scripts for...

- pruning the model weights of the CUFAR2 model in `cifar_pruning/`
- generating the model component attribution explanations in `component_attribution/`
- swapping in the final aligned outer layers around earlier versions of the intermediate representation pipeline in `swap_aligned/`
- training a random model initialized with different versions of the intermediate representation pipeline in `retrain_pipeline/`

Additional scripts for plotting training or EPK prediction statistics can be found `experiments/plots/`.

Note: If you decided for a manual installation and are receiving a `module not found` error, try `export PYTHONPATH=.` in the in the parent directory of this repository. This will tell python to look for the `explaind` package there. 

