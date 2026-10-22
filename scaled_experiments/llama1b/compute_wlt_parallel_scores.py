from compute_scores import compute_scores
import os
import random
import torch
import argparse
import json

def get_parser():
    parser = argparse.ArgumentParser(description="Compute influence scores for all LLaMA model checkpoints in a directory.")
    parser.add_argument("--checkpt_dir", type=str,
                        default="checkpoints/",
                        required=False, help="Root directory containing checkpoint subdirectories.")
    parser.add_argument("--score_dir", type=str,
                        default="scores/",
                        required=False, help="Directory to save the computed scores for each checkpoint.")
    parser.add_argument("--integration_steps", type=int,
                        default=100,
                        required=False, help="Number of integration steps for influence computation.")
    parser.add_argument("--verify", action='store_true',
                        help="Whether to verify the loss decomposition after computing scores.")
    parser.add_argument("--model_config_path", type=str,
                        default="scaled_experiments/llama1b/training/configs/models/llama1b.json",
                        required=False, help="Path to the model config file.")
    parser.add_argument("--optimizer_config_path", type=str,
                        default="scaled_experiments/llama1b/training/configs/optimizer/adamw.json",
                        required=False, help="Path to the optimizer config file.")
    parser.add_argument("--num_training_steps", type=int,
                        default=100000,
                        required=False, help="Total number of training steps (param. updates).")
    parser.add_argument("--max_grad_norm", type=float,
                        default=1.0,
                        required=False, help="Max gradient norm for clipping.")
    parser.add_argument("--seed", type=int,
                        default=42,
                        required=False, help="Random seed for shuffling checkpoints.")
    parser.add_argument("--translation_samples", 
                        type=str, 
                        required=False, 
                        default="results/translation_data_sampling/")
    parser.add_argument("--recovered_batches_path", type=str,
                        default="", help="Path to recovered batches if any.")
    parser.add_argument("--split_langs_train", action='store_true',
                        help="Whether to split training instances by language.")
    parser.add_argument("--debug", action='store_true',
                        help="Whether to run in debug mode with fewer instances and steps.")
    return parser


def get_all_checkpoints(checkpoint_root_dir, shuffle=True):
    checkpoint_dirs = []
    for root, dirs, files in os.walk(checkpoint_root_dir):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint-"):
                checkpoint_dirs.append(os.path.join(root, dir_name))

    if shuffle:
        random.shuffle(checkpoint_dirs)

    return checkpoint_dirs

def is_computed(score_path):
    return os.path.exists(score_path) and len(os.listdir(score_path)) > 0

def get_flores_instances_from(instances_path):
    # get all files in instances_path
    fnames = os.listdir(instances_path)
    batches = dict()
    for fname in fnames:
        if fname.endswith(".json"):
            batch_name = fname[:-6]
            batch_path = os.path.join(instances_path, fname)
            with open(batch_path, "r") as f:
                batch_instances = json.load(f)
            batches.update(batch_instances)
    return batches

def main():
    parser = get_parser()
    args = parser.parse_args()

    checkpoint_dirs = get_all_checkpoints(args.checkpt_dir, shuffle=True)

    # get additional parallel translation samples
    wlt_root = "results/translation_data_sampling/"
    parallel_flores_instances = get_flores_instances_from(os.path.join(wlt_root, "parallel_flores_batches/"))
    test_instance_path = os.path.join(wlt_root, "wlt_batches.json")

    for checkpoint_path in checkpoint_dirs:

        torch.cuda.empty_cache()
        checkpoint_name = os.path.basename(checkpoint_path)

        # exclude last checkpoint
        if checkpoint_name == "checkpoint-0-100000":
            print("Skipping last checkpoint (checkpoint-0-100000).")
            continue

        score_path = os.path.join(args.score_dir, checkpoint_name + "_scores")
        recovered_batch_path = os.path.join(args.recovered_batches_path, checkpoint_name, "batch.pt")

        if is_computed(score_path):
            print(f"Scores already computed for checkpoint: {checkpoint_path}, skipping.")
            continue

        print(f"Computing scores for checkpoint: {checkpoint_path}")
        # compute_scores(
        #     checkpoint_path=checkpoint_path,
        #     model_config_path=args.model_config_path,
        #     optimizer_config_path=args.optimizer_config_path,
        #     test_instances_path=args.blimp_path,
        #     gradient_accumulation_steps=args.gradient_accumulation_steps,
        #     recovered_batches_path=args.recovered_batches_path,
        #     num_training_steps=args.num_training_steps,
        #     max_grad_norm=args.max_grad_norm,
        #     num_steps=args.integration_steps,
        #     verify=False,
        #     split_langs_train=args.split_langs_train,
        #     world_size=args.world_size,
        #     score_path=score_path,
        #     save_all_scores=False,
        # )

        compute_scores(
            checkpoint_path=checkpoint_path,
            model_config_path=args.model_config_path,
            optimizer_config_path=args.optimizer_config_path,
            test_instances_path=test_instance_path,
            additional_train_instances=parallel_flores_instances,
            recovered_batches_path=recovered_batch_path,
            num_training_steps=args.num_training_steps,
            max_grad_norm=args.max_grad_norm,
            num_steps=args.integration_steps,
            verify=True,
            test_batch_limit=-1,
            split_langs_train=True,
            save_all_scores=False,
            score_path=score_path,
            train_tok_frac=0.125,
            ignore_train_influences=False,
            compute_sim_scores=False,
            debug=args.debug,
        )

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()