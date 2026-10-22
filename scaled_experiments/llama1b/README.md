# Copy First, Translate Later

ExPLAIND experiments for [Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining](https://arxiv.org/abs/2604.17633). These scripts decompose changes in word-level translation loss into contributions from training data and model parameters.

Training and evaluation code lives in [mainlp/copy-first-translate-later](https://github.com/mainlp/copy-first-translate-later).

## Requirements

Use Python 3.12 and a CUDA GPU. Run commands from the ExPLAIND repository root.

```bash
pip install -r requirements.txt
pip install -r requirements-scaled.txt
```

For cloze translation and static plots, additionally install:

```bash
pip install sentence-transformers sentencepiece sacremoses kaleido
```

Install the companion training code's dependencies and link its training directory here:

```bash
ln -s /absolute/path/to/training scaled_experiments/llama1b/training
```

The linked directory must contain `src/model.py` with `build_model`, and the model and optimizer configs under `configs/`.

## Data and checkpoints

- **WLT:** Download [mainlp/word-level-translations](https://huggingface.co/datasets/mainlp/word-level-translations), including `query_set`, `example_pool`, and `word_alternatives`. Generate five-shot prompts using the companion evaluation code. The sampling scripts expect CSVs at `<wlt_root>/<src>_to/<src>_<tgt>.csv`, including repetition pairs, with `prompt`, `src_word`, `tgt_word`, and `valid_targets` columns.
- **FLORES:** Script downloads `facebook/flores` (`all`, dev and devtest) automatically.
- **Checkpoints:** Download the required `step{N}` revisions from [mainlp/copy-first-translate-later](https://huggingface.co/mainlp/copy-first-translate-later). Store each under `checkpoint-0-{N}/`, including model, optimizer, and scheduler state.
- **Training batches:** Supply update batches for each checkpoint as `<batch_root>/checkpoint-0-{N}/batch.pt`. Recover the documents from [FineWeb-HQ](https://huggingface.co/datasets/epfml/FineWeb-HQ) and [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ), using the released training manifests and the original tokenization and packing procedure. Alternatively, you can provide any other tokenized batch of the same dimension.

## Generate analysis batches

Update the default WLT path in `get_wlt_entities_and_prompts` in `translation_data_sampling/sample_parallel_flores_batches.py`, then run:

```bash
python scaled_experiments/llama1b/translation_data_sampling/sample_parallel_flores_batches.py
```

This writes `wlt_batches.json` and `parallel_flores_batches/` under `results/translation_data_sampling/`.

The FLORES loader requires `datasets==3.6.0`, which conflicts with the scaled requirements. Run this preparation step in a separate environment with `datasets==3.6.0`, `pandas`, and `tqdm`.

For the concept-occurrence experiments, provide the manually authored English cloze CSVs referenced in `translate_clozes.py`, then run:

```bash
python scaled_experiments/llama1b/translation_data_sampling/translate_clozes.py
python -m scaled_experiments.llama1b.translation_data_sampling.wlt_occurence_data
```

The second script also requires WLT evaluation outputs under `results/eval-results/results/`. It produces `wlt_occurence_batches.json` and `wlt_occurence_train_batches.json`; pass these as test instances and additional training instances to `compute_scores(...)`.

## Compute scores

```bash
python scaled_experiments/llama1b/compute_wlt_parallel_scores.py \
  --checkpt_dir /path/to/checkpoints \
  --recovered_batches_path /path/to/recovered_batches \
  --score_dir results/llama1b_scores \
  --model_config_path scaled_experiments/llama1b/training/configs/models/llama1b.json \
  --optimizer_config_path scaled_experiments/llama1b/training/configs/optimizer/adamw.json \
  --integration_steps 100
```

The wrapper reads batches from `results/translation_data_sampling/`, computes scores per checkpoint, and saves loss-decomposition verification results. These input paths are currently hard-coded; `--translation_samples` does not override them.

## Plot results

```bash
python scaled_experiments/llama1b/plot_scores.py \
  --dir results/llama1b_scores \
  --plotdir results/llama1b_plots \
  --grouping threesteps
```

Use `--grouping type` or `--grouping depth` for other parameter groupings, and `--normalized` for normalized scores. Static export through Kaleido requires Chrome.


## Citing this work

If you use this code, the dataset, or the checkpoints, please cite (to be replaced by the ACL Anthology .bib once EMNLP Proceedings are out):

```
@misc{koerner2026copyfirsttranslatelater,
      title={Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining}, 
      author={Felicia Körner and Maria Matveev and Florian Eichin and Gitta Kutyniok and Barbara Plank and Michael A. Hedderich},
      year={2026},
      eprint={2604.17633},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.17633}, 
}
```