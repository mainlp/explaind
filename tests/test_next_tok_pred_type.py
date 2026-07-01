import sys
sys.path.append('./')
from explaind.accelerated.llama_explainer import LlamaExplainer
import unittest
import torch

def compute_scores(checkpoint_path: str,
                   model_config_path: str,
                   optimizer_config_path: str,
                   split_langs_train: bool = False,
                   test_instances_path: str = None,
                   num_training_steps: int = 2000,
                   max_grad_norm: float = 1.0,
                   num_steps: int = 100,
                   train_tok_frac: float = 1.0,
                   verify: bool = True):


    explainer = LlamaExplainer(
        checkpoint_path=checkpoint_path,
        test_instances_path=test_instances_path,
        model_config_path=model_config_path,
        optimizer_config_path=optimizer_config_path,
        split_langs_train=split_langs_train,
        num_training_steps=num_training_steps,
        max_grad_norm=max_grad_norm,
        num_steps=num_steps,
        train_tok_frac=train_tok_frac,
        expl_type="loss",
    )
    
    explainer.compute_influence_scores()
    explainer.save_scores_to_disk()

    results = explainer.verify_loss_decomposition()

    return explainer, results

class TestNextTokLlamaTiny(unittest.TestCase):
    
    def test_split_by_lang_scores(self):
        checkpoint_path = "scaled_experiments/llama1b/toy_checkpoints/llama_super_tiny/checkpoints/checkpoint-0-2000"
        model_config_path = "scaled_experiments/llama1b/toy_checkpoints/llama_super_tiny.json"
        optimizer_config_path = "scaled_experiments/llama1b/training/configs/optimizer/adamw.json"

        test_batch_path = "scaled_experiments/llama1b/toy_checkpoints/llama_super_tiny/checkpoints/next_tok_batch.json"

        explainer, results = compute_scores(
            checkpoint_path=checkpoint_path,
            test_instances_path=test_batch_path,
            model_config_path=model_config_path,
            optimizer_config_path=optimizer_config_path,
            split_langs_train=True,
            num_training_steps=2000,
            max_grad_norm=1.0,
            num_steps=100,
            train_tok_frac=1.0,
            verify=True
        )

        # todo
        assert abs(results[0]['de_sen_good_0']['actual_change']) > 1e-4
        assert abs(results[0]['de_sen_good_0']['delta']) < 1e-6
        assert abs(results[0]['de_sen_good_2_0']['actual_change']) > 1e-4
        assert abs(results[0]['de_sen_good_2_0']['delta']) < 1e-6
        # assert abs(results[0]['debug_copy_of_0']['s']) > 1e-4
        # assert abs(results[0]['debug_copy_of_0']['delta']) < 2e-6 