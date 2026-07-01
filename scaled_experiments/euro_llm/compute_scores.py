import os
import json
import argparse
import torch

from explaind.accelerated.eurollm_explainer import EuroLLMExplainer


def compute_scores(
    conv_checkpoint_path: str,
    checkpoint_path: str,
    model_config_path: str,
    optimizer_config_path: str,
    test_instances_path: str = None,
    test_instances: dict = None,
    additional_train_instances: dict = None,
    recovered_batches_path: str = None,
    gradient_accumulation_steps: int = 1,
    num_training_steps: int = 100000,
    max_grad_norm: float = 1.0,
    test_batch_limit: int = -1,
    num_steps: int = 100,
    verify: bool = True,
    score_path: str = None,
    save_all_scores: bool = True,
    split_langs_train: bool = False,
    ignore_train_influences: bool = False,
    compute_sim_scores: bool = False,
    debug: bool = False,
    train_tok_frac: float = 1.0,
):
    """Compute and optionally verify EuroLLM ExPLAIND scores for one checkpoint.

    Args:
        conv_checkpoint_path: Converted Hugging Face checkpoint directory.
        checkpoint_path: Original optimizer checkpoint, usually ``model_optim_rng.pt``.
        model_config_path: JSON config used to rebuild the model architecture.
        optimizer_config_path: JSON config used to rebuild optimizer groups.
        test_instances_path: Optional JSON test-instance file.
        test_instances: Optional in-memory test-instance mapping. If omitted,
            the explainer starts with an empty mapping and may load instances from
            ``test_instances_path`` during checkpoint reconstruction.
        additional_train_instances: Optional extra train groups to score.
        recovered_batches_path: JSONL file containing the recovered update batch.
        gradient_accumulation_steps: Number of training micro-batches in the update.
        num_training_steps: Total training steps used to reconstruct the scheduler.
        max_grad_norm: Gradient clipping threshold used during training.
        test_batch_limit: Maximum number of test batches to score; ``-1`` means all.
        num_steps: Integral approximation steps.
        verify: Whether to save loss-decomposition verification results.
        score_path: Directory where scores and verification files are written.
        save_all_scores: Whether to store full per-parameter scores.
        split_langs_train: Whether to split recovered train batches by language.
        ignore_train_influences: Skip train influence terms when supported.
        compute_sim_scores: Whether to compute pairwise score similarities.
        debug: Enable reduced/debug computation in the explainer.
        train_tok_frac: Fraction of train tokens to keep when limiting train batches.
    """

    if test_instances is None:
        test_instances = {}

    explainer = EuroLLMExplainer(
        test_instances=test_instances,
        test_instances_path=test_instances_path,
        additional_train_instances=additional_train_instances,
        conv_checkpoint_path=conv_checkpoint_path,
        checkpoint_path=checkpoint_path,
        model_config_path=model_config_path,
        optimizer_config_path=optimizer_config_path,
        gradient_accumulation_steps=gradient_accumulation_steps,
        recovered_batches_path=recovered_batches_path,
        num_training_steps=num_training_steps,
        max_grad_norm=max_grad_norm,
        num_steps=num_steps,
        expl_type="loss",
        score_path=score_path,
        test_batch_limit=test_batch_limit,
        save_all_scores=save_all_scores,
        split_langs_train=split_langs_train,
        train_tok_frac=train_tok_frac,
        ignore_train_influences=ignore_train_influences,
        compute_sim_scores=compute_sim_scores,
        debug=debug,
    )
    print("Computing influence scores...")
    explainer.compute_influence_scores()
    print("Computed influence scores.")

    print("Saving influence scores to disk...")
    explainer.save_scores_to_disk()
    print("Saved influence scores to disk.")

    if not os.path.exists(score_path):
        os.makedirs(score_path)

    torch.cuda.empty_cache()

    if verify:
        results = explainer.verify_loss_decomposition()
        with open(os.path.join(score_path, "verification_results.json"), "w") as f:
            json.dump(results, f, indent=4)


def get_parser():
    """Create a CLI parser for scoring a single EuroLLM checkpoint."""
    parser = argparse.ArgumentParser(description="Compute ExPLAIND scores for one EuroLLM checkpoint.")
    parser.add_argument("--conv_checkpoint_path", required=True, help="Converted Hugging Face checkpoint directory.")
    parser.add_argument("--checkpoint_path", required=True, help="Original optimizer checkpoint path.")
    parser.add_argument("--model_config_path", required=True, help="Model config JSON path.")
    parser.add_argument("--optimizer_config_path", required=True, help="Optimizer config JSON path.")
    parser.add_argument("--test_instances_path", default=None, help="Optional JSON test-instance path.")
    parser.add_argument("--recovered_batches_path", default=None, help="Recovered training batch JSONL path.")
    parser.add_argument("--score_path", required=True, help="Output directory for scores.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_training_steps", type=int, default=100000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--test_batch_limit", type=int, default=-1)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--train_tok_frac", type=float, default=1.0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--split_langs_train", action="store_true")
    parser.add_argument("--ignore_train_influences", action="store_true")
    parser.add_argument("--compute_sim_scores", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_save_all_scores", action="store_true")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    compute_scores(
        conv_checkpoint_path=args.conv_checkpoint_path,
        checkpoint_path=args.checkpoint_path,
        model_config_path=args.model_config_path,
        optimizer_config_path=args.optimizer_config_path,
        test_instances_path=args.test_instances_path,
        recovered_batches_path=args.recovered_batches_path,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_training_steps=args.num_training_steps,
        max_grad_norm=args.max_grad_norm,
        test_batch_limit=args.test_batch_limit,
        num_steps=args.num_steps,
        verify=args.verify,
        score_path=args.score_path,
        save_all_scores=not args.no_save_all_scores,
        split_langs_train=args.split_langs_train,
        ignore_train_influences=args.ignore_train_influences,
        compute_sim_scores=args.compute_sim_scores,
        debug=args.debug,
        train_tok_frac=args.train_tok_frac,
    )
