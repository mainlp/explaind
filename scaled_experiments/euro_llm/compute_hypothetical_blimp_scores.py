from compute_scores import compute_scores
import os
import random
import torch
import argparse
import json
import glob
import re


def get_parser():
    """Create the CLI parser for hypothetical BLiMP scoring."""
    parser = argparse.ArgumentParser(
        description="Compute influence scores for all EuroLLM model checkpoints in a directory."
    )
    parser.add_argument(
        "--conv_checkpoint_dir",
        type=str,
        default="/nfs/gdata/fkoerner/conversion/converted/eurollm/1b",
        required=False,
        help="Root directory containing converted checkpoint subdirectories.",
    )
    parser.add_argument(
        "--checkpt_dir",
        type=str,
        default="/nfs/gdata/fkoerner/mega-lm/eurollm/1b/",
        required=False,
        help="Root directory containing megatron checkpoint subdirectories.",
    )
    parser.add_argument(
        "--score_dir",
        type=str,
        default="/nfs/brick/explainables/runs/euro_llm_explaind_scores/hypothetical_blimp_v2/",
        required=False,
        help="Directory to save the computed scores for each checkpoint.",
    )
    parser.add_argument(
        "--integration_steps",
        type=int,
        default=100,
        required=False,
        help="Number of integration steps for influence computation.",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Whether to verify the loss decomposition after computing scores."
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="scaled_experiments/llama1b/training/configs/models/llama1b.json",
        required=False,
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--optimizer_config_path",
        type=str,
        default="scaled_experiments/llama1b/training/configs/optimizer/adamw.json",
        required=False,
        help="Path to the optimizer config file.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3072,
        required=False,
        help="Gradient accumulation steps per device.",
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=310000,
        required=False,
        help="Total number of training steps (param. updates).",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, required=False, help="Max gradient norm for clipping."
    )
    parser.add_argument("--seed", type=int, default=42, required=False, help="Random seed for shuffling checkpoints.")
    parser.add_argument("--blimp_path", type=str, required=False, default="results/blimp_scores/blimp_samples.json")
    parser.add_argument(
        "--recovered_batches_path",
        type=str,
        default="/nfs/brick/recover_batches/checkpoint_recovery_dir/checkpoint_recovery_run/checkpoints/",
        help="Path to recovered batches if any.",
    )
    parser.add_argument(
        "--split_langs_train", action="store_true", help="Whether to split training instances by language."
    )

    return parser


def get_all_checkpoints(checkpoint_root_dir, shuffle=True):
    """Return converted checkpoint directories, optionally shuffled."""
    checkpoint_dirs = glob.glob(f"{checkpoint_root_dir}/*/*")

    if shuffle:
        random.shuffle(checkpoint_dirs)

    return checkpoint_dirs


def is_computed(score_path):
    """Return whether a score directory already contains outputs."""
    return os.path.exists(score_path) and len(os.listdir(score_path)) > 0


def get_instances_from(instances_path):
    """Load instances from a JSON file."""
    with open(instances_path, "r") as f:
        instances_text = json.load(f)
    return instances_text


def main():
    parser = get_parser()
    args = parser.parse_args()

    checkpoint_dirs = get_all_checkpoints(args.conv_checkpoint_dir, shuffle=True)

    additional_parallel_instances = None

    # randomize order of checkpoint dirs
    permutation = random.sample(range(len(checkpoint_dirs)), len(checkpoint_dirs))
    checkpoint_dirs = [checkpoint_dirs[i] for i in permutation]

    for checkpoint_path in checkpoint_dirs:
        torch.cuda.empty_cache()
        checkpoint_name = "/".join(re.findall(r"/([^/]*)", checkpoint_path)[-2:])  # get last two parts of path
        megatron_checkpoint_path = os.path.join(args.checkpt_dir, checkpoint_name, "mp_rank_00/model_optim_rng.pt")

        # exclude last checkpoint
        if checkpoint_name == "iter-0310000":
            print("Skipping last checkpoint (iter-0310000)")
            continue

        score_path = os.path.join(args.score_dir, checkpoint_name + "_scores")

        # take one from other model now (replace with our own later)
        if "second_phase" in checkpoint_name:
            recovered_batch_path = (
                "/mounts/Users/cisintern/feichin/projects/explaind-1/"
                "results/lang_batches/eurollm_phase2_annealed_train_batch.jsonl"
            )
            # only do first phase for now
            continue
        else:
            recovered_batch_path = (
                "/mounts/Users/cisintern/feichin/projects/explaind-1/"
                "results/lang_batches/eurollm_phase1_train_batch.jsonl"
            )

        if is_computed(score_path):
            print(f"Scores already computed for checkpoint: {checkpoint_path}, skipping.")
            continue

        print(f"Computing scores for checkpoint: {checkpoint_path}")

        compute_scores(
            conv_checkpoint_path=checkpoint_path,
            checkpoint_path=megatron_checkpoint_path,
            model_config_path=args.model_config_path,
            optimizer_config_path=args.optimizer_config_path,
            test_instances=None,
            additional_train_instances=additional_parallel_instances,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            recovered_batches_path=recovered_batch_path,
            num_training_steps=args.num_training_steps,
            max_grad_norm=args.max_grad_norm,
            num_steps=args.integration_steps,
            verify=False,
            test_batch_limit=-1,
            split_langs_train=True,
            save_all_scores=False,
            score_path=score_path,
            train_tok_frac=1.0,
            ignore_train_influences=False,
            compute_sim_scores=False,
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
