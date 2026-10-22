import os
import json
import sys
import argparse
import torch

sys.path.append('./')
from explaind.accelerated.llama_explainer import LlamaExplainer

def compute_scores(checkpoint_path: str,
                   model_config_path: str,
                   optimizer_config_path: str,
                   test_instances_path: str = None,
                   test_instances: dict = None,
                   additional_train_instances: dict = None,
                   inject_train_instances: dict = None,
                   recovered_batches_path: str = None,
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
                   train_tok_frac: float = 1.0):


    test_instances = {}

    explainer = LlamaExplainer(
        test_instances=test_instances,
        test_instances_path=test_instances_path,
        additional_train_instances=additional_train_instances,
        checkpoint_path=checkpoint_path,
        model_config_path=model_config_path,
        optimizer_config_path=optimizer_config_path,
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
        limit_train_batches=(train_tok_frac < 1.0 and train_tok_frac > 0.0),
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute influence scores for LLaMA model checkpoints.")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/",
                        required=False, help="Path to the checkpoint directory.")
    parser.add_argument("--score_path", type=str,
                        default="scores/",
                        required=False, help="Path to save the computed scores.")
    parser.add_argument("--model_config_path", type=str, 
                        default="scaled_experiments/llama1b/training/configs/models/llama1b.json",
                        # default="scaled_experiments/llama1b/toy_checkpoints/llama_super_tiny.json", 
                        required=False, help="Path to the model config file.")
    parser.add_argument("--optimizer_config_path", type=str, 
                        default="scaled_experiments/llama1b/training/configs/optimizer/adamw.json",
                        required=False, help="Path to the optimizer config file.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps per device.")
    parser.add_argument("--num_training_steps", type=int, default=100000, help="Number of training steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of integration steps for influence scores.")
    parser.add_argument("--recovered_batches_path", type=str, default="", help="Path to recovered batches if any.")

    args = parser.parse_args()

    compute_scores(
        checkpoint_path=args.checkpoint_path,
        model_config_path=args.model_config_path,
        optimizer_config_path=args.optimizer_config_path,
        recovered_batches_path=args.recovered_batches_path,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_training_steps=args.num_training_steps,
        max_grad_norm=args.max_grad_norm,
        num_steps=args.num_steps,
        score_path=args.score_path,
        world_size=args.world_size,
    )