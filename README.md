![Screenshot 2025-05-21 at 08 44 04](https://github.com/user-attachments/assets/077ef1a3-2e37-4e3a-8434-cf6673369122)

# ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior

This repository is the official implementation of [Grokking ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior](https://github.com/mainlp/path_kernels). 


## Requirements

We ran all our experiments in python version `3.12`. You can use `conda` to create a fresh environment first:

```
conda create -n explaind python=3.12
conda activate explaind
```

For applying and validating ExPLAIND, the requirements amount to current versions of `torch`, `plotly`, and `numpy`:

```setup
pip install torch torchvision numpy tqdm
```

If you also want to recreate the plots shown in the paper, you additionally need the following packages:

```setup
pip install plotly pandas umap_learn
```


## Training models with history

If you want to apply ExPLAIND to your own model, you need to retrain it tracking relevant parts of the training process by using the wrappers provided in this repository. Note that depending on model size full history tracking can become very expensive. We're currently working on a solution for allowing cheaper partial tracking.

```python
TODO
```

## Getting EPK predictions and kernel accumulations

Once you have the history of the training run you want to explain, you can load them into the EPK module and compute the prediction of the reformulated model as follows:

```python
TODO
```

Note that there are different settings for which (accumulated) slices of the kernel to store during the prediction. Depending on your choices there, runtimes can vary greatly because of GPU I/O and extra matrix computations involved.

## Ablations and other experiments

We run several ablations and experiments to support our findings 


## Contributing

We publish this repository under MIT license and welcome anybody who wants to contribute. If you have a question or an idea, feel free to reach out to Florian (feichin[at]cis[dot]lmu[dot]de) or simply start a pull request/issue.

If you want to use our code for your own projects, please consider citing our work:

```
TODO
```

