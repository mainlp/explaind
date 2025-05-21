# EPK Validation Experiments

The scripts found here assume you have already trained a model whose ExPLAIND checkpoints, i.e. histories of optimizer, data and model, have already been stored.

Once you've completed a run of the EPK model, you can check out `load_matrices.py` to see how you can interact with some of the results that are stored. You can also head to `../plots/plot_mod_epk_stats.py` to plot additional statistics found in the paper about the EPK prediction.

Note that running the EPK predictions for large batches requires a lot of GPU memory and will result in large kernel matrices stored on disk. We're currently working on making things more efficient. For the CNN model, we've already implemented some basic multi-GPU support (see documentation of `explaind.optimizer_path.SGDOptimizerPath`), which we plan to extend later.