import torch
import torch.nn as nn
import os
from copy import deepcopy
from torch.optim import AdamW
from explaind.accelerated.explainer import StepExplainer
import tempfile
import unittest
from accelerate import Accelerator


class DummyReturnVal:
    def __init__(self, loss, outputs):
        self.loss = loss
        self.outputs = outputs
    
    def __getitem__(self, idx):
        return [self.loss, self.outputs][idx]


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.linear.weight.data.fill_(-1.0)
        self.linear.bias.data.fill_(2.0)
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, input_ids, lm_labels, **kwargs):

        input_ids = input_ids.to(self.linear.weight.device)
        lm_labels = lm_labels.to(self.linear.weight.device)    

        f = self.linear(input_ids)
        f = nn.Sigmoid()(f)
        l = self.loss(f, lm_labels)
        return DummyReturnVal(l, f)


class SimpleStepExplainer(StepExplainer):
    def __init__(self, 
                 checkpoint_path,
                 args,
                 test_instances,
                 weight_decay=1.0,
                 learning_rate=0.1,
                 betas=(0.9, 0.999),
                 save_all_scores=False,
                 ignore_train_types=True,
                 change_lr_to=None,
                 expl_type="loss",  
                 num_steps=10,
                 debug=False,):

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.betas = betas
        self.change_lr_to=change_lr_to

        self.test_grad_scaling = 1.0
        self.train_grad_scaling = 1.0
        self.train_loss_scaling = 1.0
        self.test_loss_scaling = 1.0

        self.args = args

        super().__init__(checkpoint_path,
                        save_all_scores=True,
                         test_instances=test_instances,
                         ignore_train_types=ignore_train_types,
                         expl_type=expl_type,
                         num_steps=num_steps,
                         debug=debug)

        

        
    def load_checkpoint(self):

        # simple linear model
        self.model = SimpleModel()

        self.optimizer = AdamW(self.model.named_parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay,
                               betas=self.betas)

        self.train_batch = {
            "input_ids": torch.tensor([[3.0], [0.0], [1.0], [-1.0], [5.0], [2.0], [-10.0], [-3.0],]),
            "lm_labels": torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [0.0]]),
            "type": ["t1", "t0", "t1", "t0", "t1", "t1", "t0", "t0"],
        }

        # self.train_batch = {
        #     "input_ids": torch.tensor([[3.0]]),
        #     "lm_labels": torch.tensor([[1.0]]),
        #     "type": ["t0"],
        # }

        self.train_instances = {
            "t0": self.train_batch
        }

        self.scheduler = None

        self.step_id = 1

        self.accelerator = Accelerator()
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

    def compute_step_model(self):

        self.model.train()

        outputs = self.model(**self.train_batch)
        loss = outputs.loss
        self.accelerator.backward(loss)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # do another step
        if self.change_lr_to is not None:
            self.optimizer.lr = self.change_lr_to

        self.optimizer_pre = deepcopy(self.optimizer)
        self.model_pre = deepcopy(self.model)

        outputs = self.model(**self.train_batch)
        loss = outputs.loss
        self.accelerator.backward(loss)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.gradient_scaling = 1.0
        self.wrapped_model = self._wrap_model("", self.model_pre, self.model)



class TestStepExplainer(unittest.TestCase):

    def setUp(self):
        self.instances_batched = {
            "testa": {
                "input_ids": torch.tensor([[10.0]]),
                "lm_labels": torch.tensor([[1.0]]),
                "type": ["t1"],
            },
            "testb": {
                "input_ids": torch.tensor([[-3.0], [-5.0], [-15.0]]),
                "lm_labels": torch.tensor([[0.0], [0.0], [0.0]]),
                "type": ["t0", "t0", "t0"],
            },
            "testc": {
                "input_ids": torch.tensor([[4.0], [5.0], [6.0]]),
                "lm_labels": torch.tensor([[1.0], [1.0], [1.0]]),
                "type": ["t1", "t1", "t1"],
            },
            "testd": {
                "input_ids": torch.tensor([[1.0], [2.0], [1.0]]),
                "lm_labels": torch.tensor([[1.0], [1.0], [1.0]]),
                "type": ["t1", "t1", "t1"],
            },
            "teste": {
                "input_ids": torch.tensor([[1.0], [-1.0], [-3.0]]),
                "lm_labels": torch.tensor([[1.0], [0.0], [0.0]]),
                "type": ["t1", "t0", "t0"],
            }
        }

    def test_simple_step_explainer(self):
        test_instances = self.instances_batched

        explainer = SimpleStepExplainer(
            checkpoint_path=tempfile.gettempdir(),
            args=None,
            test_instances=test_instances,
            expl_type="loss",
            num_steps=1000,
            debug=True,

        )

        explainer.compute_influence_scores()

        scores = explainer.all_scores[('t0', 'testa', 'loss')]

        # Check if scores are computed for the linear layer's weight and bias
        self.assertIn("linear.weight", scores)
        self.assertIn("linear.bias", scores)

        # Check if the scores have the correct shape
        self.assertEqual(scores["linear.weight"].shape, torch.Size([1, 1]))
        self.assertEqual(scores["linear.bias"].shape, torch.Size([1]))

        result = explainer.verify_loss_decomposition()
        for key, val in result[0].items():
            if key in {'reg_ker_fm_weights(reg,ker,fm)', 'reg_ker_weights_no_fm(reg,ker)'}:
                # skip these terms for this test
                continue
            print(val)
            is_close = (abs(val['delta']) / abs(val['actual_change']))
            print(is_close)
            is_close = is_close < 1e-3
            self.assertTrue(is_close, f"Loss decomposition verification failed for {key}")

        print("Verification result:", result)

    
    def test_regularization(self):

        test_instances = self.instances_batched

        explainer = SimpleStepExplainer(
            checkpoint_path=tempfile.gettempdir(),
            args=None,
            test_instances=test_instances,
            expl_type="loss",
            num_steps=1000,
            debug=True,
            weight_decay=1000000.0,
            learning_rate=0.0000001,
        )

        explainer.compute_influence_scores()

        scores = explainer.all_scores[('t0', 'testa', 'loss')]

        # Check if scores are computed for the linear layer's weight and bias
        self.assertIn("linear.weight", scores)
        self.assertIn("linear.bias", scores)

        # Check if the scores have the correct shape
        self.assertEqual(scores["linear.weight"].shape, torch.Size([1, 1]))
        self.assertEqual(scores["linear.bias"].shape, torch.Size([1]))

        result = explainer.verify_loss_decomposition()
        for key, val in result[0].items():
            if key in {'reg_ker_fm_weights(reg,ker,fm)', 'reg_ker_weights_no_fm(reg,ker)'}:
                # skip these terms for this test
                continue
            print(val)
            is_close = (abs(val['delta']) / abs(val['actual_change']))
            print(is_close)
            is_close = is_close < 1e-2
            self.assertTrue(is_close, f"Loss decomposition verification failed for {key}")

        print("Verification result:", result)

    def test_kernel_term(self):
        test_instances = self.instances_batched

        explainer = SimpleStepExplainer(
            checkpoint_path=tempfile.gettempdir(),
            args=None,
            test_instances=test_instances,
            expl_type="loss",
            num_steps=1000,
            debug=True,
            betas=(0.0, 0.999),
            weight_decay=0.0,
            learning_rate=0.1,
        )

        explainer.compute_influence_scores()

        scores = explainer.all_scores[('t0', 'testa', 'loss')]

        # Check if scores are computed for the linear layer's weight and bias
        self.assertIn("linear.weight", scores)
        self.assertIn("linear.bias", scores)

        # Check if the scores have the correct shape
        self.assertEqual(scores["linear.weight"].shape, torch.Size([1, 1]))
        self.assertEqual(scores["linear.bias"].shape, torch.Size([1]))

        result = explainer.verify_loss_decomposition()
        for key, val in result[0].items():
            if key in {'reg_ker_fm_weights(reg,ker,fm)', 'reg_ker_weights_no_fm(reg,ker)'}:
                # skip these terms for this test
                continue
            print(val)
            is_close = (abs(val['delta']) / abs(val['actual_change']))
            print(is_close)
            is_close = is_close < 1e-2
            self.assertTrue(is_close, f"Loss decomposition verification failed for {key}")

        print("Verification result:", result)

    def test_first_moment_term(self):
        test_instances = self.instances_batched

        explainer = SimpleStepExplainer(
            checkpoint_path=tempfile.gettempdir(),
            args=None,
            test_instances=test_instances,
            expl_type="loss",
            num_steps=1000,
            debug=True,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            learning_rate=0.1,
        )

        explainer.compute_influence_scores()

        scores = explainer.all_scores[('t0', 'testa', 'loss')]

        # Check if scores are computed for the linear layer's weight and bias
        self.assertIn("linear.weight", scores)
        self.assertIn("linear.bias", scores)

        # Check if the scores have the correct shape
        self.assertEqual(scores["linear.weight"].shape, torch.Size([1, 1]))
        self.assertEqual(scores["linear.bias"].shape, torch.Size([1]))

        result = explainer.verify_loss_decomposition()
        for key, val in result[0].items():
            if key in {'reg_ker_fm_weights(reg,ker,fm)', 'reg_ker_weights_no_fm(reg,ker)'}:
                # skip these terms for this test
                continue
            print(val)
            is_close = (abs(val['delta']) / abs(val['actual_change']))
            print(is_close)
            is_close = is_close < 1e-2
            self.assertTrue(is_close, f"Loss decomposition verification failed for {key}")

        print("Verification result:", result)


    def test_fundamental_thm_of_algebra(self):
        # yup...

        test_instances = self.instances_batched

        explainer = SimpleStepExplainer(
            checkpoint_path=tempfile.gettempdir(),
            args=None,
            test_instances=test_instances,
            expl_type="loss",
            num_steps=100,
            debug=True,
            weight_decay=0.0,
            learning_rate=0.1,
        )

        # compute integral only
        explainer.compute_test_integral(test_instances["testb"], 0)
        integral = explainer.wrapped_model.test_gradients
        print("Computed integral:", integral)

        f0 = explainer.wrapped_model.model0(**test_instances["testb"])[0]
        explainer.accelerator.backward(f0)
        g0 = {k: p.grad.clone() for k, p in explainer.wrapped_model.model0.named_parameters() if p.grad is not None}    
        p0 = {k: p.clone() for k, p in explainer.wrapped_model.model0.named_parameters()}

        f1 = explainer.wrapped_model.model1(**test_instances["testb"])[0]
        explainer.accelerator.backward(f1)
        g1 = {k: p.grad.clone() for k, p in explainer.wrapped_model.model1.named_parameters() if p.grad is not None}
        p1 = {k: p.clone() for k, p in explainer.wrapped_model.model1.named_parameters()}

        diff_g = {k: g1[k] - g0[k] for k in g0.keys()}
        diff_f = f1 - f0
        diff_p = {k: p1[k] - p0[k] for k in p0.keys()}

        
        for k in diff_g.keys():
            print("Parameter:", k)
            print(f"Integral: {integral[k].item()}, Diff grad: {diff_g[k].item()}, Diff param: {diff_p[k].item()}, diff f: {diff_f.item()}")
            print((diff_f / diff_p[k]).item())