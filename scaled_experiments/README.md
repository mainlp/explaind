# Scaled experiments

The scaled experiments adapt ExPLAIND to LLM checkpoints by explaining one optimizer update at a time with `StepExplainer`. The EuroLLM scripts are in [`euro_llm/`](euro_llm/) and assume access to converted Hugging Face checkpoints, original optimizer checkpoints, recovered training batches, and evaluation samples.

## EuroLLM workflow

Run commands from the repository root.

Install the base package dependencies and the optional scaled-experiment dependencies before running these scripts:

```bash
pip install -r requirements.txt
pip install -r requirements-scaled.txt
```

The scaled requirements cover the Hugging Face, Accelerate, SafeTensors, TensorDict, and nnsight packages used by the EuroLLM scripts and accelerated `StepExplainer` implementations.

### 1. Create language batches

```bash
python scaled_experiments/euro_llm/sample_lang_batches.py
```

This samples language-balanced JSONL batches into `results/lang_batches/`. The script streams from Hugging Face datasets and may take a long time depending on network and dataset cache state.

### 2. Compute scores for checkpoints

```bash
python scaled_experiments/euro_llm/verify_blimp_scores.py \
  --conv_checkpoint_dir /path/to/converted/eurollm/1b \
  --checkpt_dir /path/to/original/megatron/checkpoints \
  --score_dir results/euro_llm_scores/verify_blimp \
  --blimp_path results/blimp_scores/blimp_samples.json \
  --model_config_path scaled_experiments/llama1b/training/configs/models/llama1b.json \
  --optimizer_config_path scaled_experiments/llama1b/training/configs/optimizer/adamw.json \
  --split_langs_train
```

Use `compute_hypothetical_blimp_scores.py` for the hypothetical-BLiMP variant and `compute_scores.py` if you want to call the lower-level `compute_scores(...)` function directly from another script.

### 3. Check verification results

```bash
python scaled_experiments/euro_llm/check_verification.py
```

Update the checkpoint root in that script or adapt it into a one-off analysis script for your score directory.

## Integrating your own model with `StepExplainer`

Use `explaind.accelerated.explainer.StepExplainer` when full training-history tracking is too expensive and you can reconstruct a single training update. The sample `Explainer` class at the bottom of `explaind/accelerated/explainer.py` documents the required subclass interface.

To integrate a custom model:

1. Subclass `StepExplainer`.
2. In `load_checkpoint`, load the pre-update model, optimizer, scheduler if used, training batch or grouped training instances, test instances, learning rate, optimizer-state mapping, and update-step id.
3. In `compute_step_model`, reproduce the exact optimizer update being explained, keep a copy of the pre-update model and optimizer state, then call `_wrap_model("", model_before, model_after)`.
4. Set `gradient_scaling`, `train_grad_scaling`, `test_grad_scaling`, and `test_loss_scaling` to match the training recipe, especially if gradient accumulation or clipping was used.
5. Provide `train_instances` and `test_instances` as dictionaries mapping readable type names to tensor batches. Those type names become the score keys.
6. Use `expl_type="loss"` for loss decomposition. Output decomposition is intentionally not implemented in the current accelerated path.
7. Call `compute_influence_scores()`, `save_scores_to_disk()`, and optionally `verify_loss_decomposition()`.

The LLaMA and EuroLLM implementations in `explaind/accelerated/llama_explainer.py` and `explaind/accelerated/eurollm_explainer.py` are concrete references for optimizer restoration, recovered-batch loading, language splitting, and verification.

## Expected external artifacts

These files or directories are referenced by the scaled scripts but are not expected to be committed to this repository:

- converted model checkpoints, e.g. `/path/to/converted/eurollm/1b/...`;
- original Megatron optimizer checkpoints, e.g. `mp_rank_00/model_optim_rng.pt`;
- recovered training batches such as `results/lang_batches/eurollm_phase1_train_batch.jsonl`;
- BLiMP sample files under `results/blimp_scores/`;
- model and optimizer config files under `scaled_experiments/llama1b/training/configs/...` if the external training code is not included.

The required Python packages for these scripts are listed in `requirements-scaled.txt`.
