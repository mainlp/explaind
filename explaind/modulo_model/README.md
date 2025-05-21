# Modulo Model

This is an implementation of the modulo model used to investigate Grokking in [1] and [2]. For a prime number $p$ (in our case $p=113$), the task is to predict from two integers $a,b\in\\{0, ..., 112\\}$ the modulo addition

$$a + b \ \mathrm{mod} \ p$$

In our case, we use a Transformer model, with a single layer encoder consisting of an token embedding for each integer, as well as two tokens for `+` and `=`, a positional embedding that is added on top, an attention layer, and a feed-foward block. The decoder is a single feed-forward layer mapping back onto the vocabulary of numbers. See `model.py` for details. 

Some files here are remnants of older attempts to implement this. If you want to understand what is going on here, look at `train_model.py`, `loss.py`, `data.py` for the model training and `experiments/validate_epk/validate_modulo_epk_predictions.py` for the EPK experiments.
