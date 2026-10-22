import torch
from transformers import get_wsd_schedule
import os
import json
from safetensors.torch import load_file

from scaled_experiments.llama1b.training.src.model import build_model
from accelerate import Accelerator
from copy import deepcopy
import re
from tqdm import tqdm
from transformers import DataCollatorWithFlattening

from explaind.accelerated.explainer import StepExplainer

class DummyDataloader:
    def __iter__(self):
        return self
    def __next__(self):
        return None
    def __len__(self):
        return 1
    def load_state(self, state):
        return self
    def save_state(self, state):
        return self

### copied from training repo (update if necessary)
def get_scheduler_and_optimizer(optim_cfg, model, num_training_steps=100000):
    
    if isinstance(optim_cfg, str):
        optim_cfg = json.load(open(optim_cfg, "r"))

    sched_cfg = optim_cfg.pop("scheduler", None)

    opt_name = optim_cfg.pop("optimizer")
    OptClass = getattr(torch.optim, opt_name)
    optimizer = OptClass(model.named_parameters(), **optim_cfg)

    opt_state_mapping = {}
    for param_group in optimizer.state_dict()["param_groups"]:
        for id, name in zip(param_group["params"], param_group["param_names"]):
            opt_state_mapping[name] = id

    scheduler = None
    if sched_cfg is not None:
        scheduler = get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps * sched_cfg["warmup_ratio"]),
            num_decay_steps=int(num_training_steps * sched_cfg["decay_ratio"]),
            num_training_steps=num_training_steps,
            warmup_type=sched_cfg["warmup_type"],
            decay_type=sched_cfg["decay_type"],
            min_lr_ratio=sched_cfg["min_lr_ratio"],
            num_cycles=sched_cfg.get("num_cycles", 0.5),
        )

    return optimizer, scheduler, opt_state_mapping

def restore_checkpoint(
    accelerator,
    optimizer,
    scheduler,
    checkpoint_dir,
    dataloader,
    epoch=-1,
    step=-1,
    model=None,
    mode="train",
):
    if not (epoch == -1 or step == -1):
        checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint-{epoch}-{step}")
    
    accelerator.print(f"Restoring from checkpoint: {checkpoint_dir}")
    if mode == "train":
        accelerator.load_state(checkpoint_dir, load_kwargs={"weights_only": False})

        # Fix scheduler-optimizer learning rate synchronization after state loading
        # This is necessary because loading scheduler state doesn't automatically update
        # the optimizer's learning rates to match the scheduler's current state
        if scheduler is not None and optimizer is not None:
            current_lr = scheduler.get_last_lr()
            for param_group, lr in zip(optimizer.param_groups, current_lr):
                param_group["lr"] = lr
            accelerator.print(f"Synchronized learning rate to {current_lr[0]:.8f}")

        # batch = torch.load(f"{checkpoint_dir}/batch.pt")
        # consume dataloader to the right step
        # for _ in range(epoch):
        #     for j, _ in enumerate(dataloader):
        #         pass
        # for j, _ in enumerate(dataloader):
        #     if j == step:
        #         break
    else:
        assert model is not None, "Model must be provided for evaluation mode"
        weights_path = os.path.join(checkpoint_dir, "model.safetensors")
        raw_model = accelerator.unwrap_model(model)
        # simpler than loading through accelerate, as we don't need
        # optimizer/scheduler states for eval
        state_dict = load_file(weights_path, device="cpu")
        raw_model.load_state_dict(state_dict, strict=False)
        raw_model.to(accelerator.device)
        accelerator.print("Model weights loaded for evaluation.")
    
    return accelerator, optimizer, scheduler, dataloader, epoch, step


class LlamaExplainer(StepExplainer):
    def __init__(self, 
                 checkpoint_path,
                 model_config_path,
                 optimizer_config_path,
                 additional_train_instances: dict = None,
                 test_instances_path: str = None,
                 test_instances: dict = None,
                 split_langs_train: bool = False,
                 # split_langs_test: bool = False,
                 # gradient_accumulation_steps: int = 32,
                 num_training_steps: int = 100000,
                 recovered_batches_path: str = None,
                 max_grad_norm: float = 1.0,  # as per default in train.py
                 limit_update_batches: bool = False,
                 limit_train_batches: bool = False,
                 train_tok_frac: int = 0.1,
                 max_batch_len: int = 10240,
                 max_score_batch_len: int = 10240,
                 type_field: str = "language_ids",
                 ignore_train_influences: bool = False,
                 **kwargs
                ):

        self.checkpoint_path = checkpoint_path
        self.model_config_path = model_config_path
        self.optimizer_config_path = optimizer_config_path
        self.split_langs_train = split_langs_train
        self.test_instances_path = test_instances_path
        self.additional_train_instances = additional_train_instances
        self.train_instances = None
        self.dataloader = DummyDataloader()
        self.num_training_steps = num_training_steps
        self.max_grad_norm = max_grad_norm
        self.ignore_train_influences = ignore_train_influences
        self.type_field = type_field

        self.recovered_batches_path = recovered_batches_path

        # infer from number of batches in recovered batch
        self.gradient_accumulation_steps = 1 # will be set after loading batch + injected instances

        # self.num_train_batch_per_lang_max = num_train_batch_per_lang_max
        # self.num_train_batch_per_lang_min = num_train_batch_per_lang_min
        self.train_tok_frac = train_tok_frac
        self.limit_update_batches = limit_update_batches
        self.limit_train_batches = limit_train_batches
        self.max_batch_len = max_batch_len
        self.max_score_batch_len = max_score_batch_len

        super().__init__(checkpoint_path=checkpoint_path, test_instances=test_instances, **kwargs)

        if self.test_instances is not None:
            # put on device right away
            for key, batch in self.test_instances.items():
                batch['labels'] = batch['input_ids'].to(self.model.device)
                if "target_ids" in batch:
                    batch['target_ids'] = batch['input_ids'].to(self.model.device)
                batch['position_ids'] = batch['position_ids'].to(self.model.device)
                batch['cu_seq_lens_q'] = batch['cu_seq_lens_q'].to(self.model.device)
                batch['cu_seq_lens_k'] = batch['cu_seq_lens_k'].to(self.model.device)
                batch[self.type_field] = batch[self.type_field].to(self.model.device)
                batch['input_ids'] = batch['input_ids'].to(self.model.device)
                self.test_instances[key] = batch

    
    def load_checkpoint(self):
        """
        Load checkpoints.
        """
        self.accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
        self.model, self.tokenizer = build_model(self.model_config_path, self.accelerator)

        self.optimizer, self.scheduler, self.optimizer_state_mapping = get_scheduler_and_optimizer(
            self.optimizer_config_path, self.model, num_training_steps=self.num_training_steps)

        self.model, self.optimizer, self.scheduler, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.dataloader)
    
        self.accelerator, self.optimizer, self.scheduler, self.dataloader, self.start_epoch, self.step = \
            restore_checkpoint(
                accelerator=self.accelerator, 
                optimizer=self.optimizer, 
                scheduler=self.scheduler, 
                checkpoint_dir=self.checkpoint_path, 
                dataloader=self.dataloader,
                model=self.model,
                #epoch=0,
                #step=self.step_id,
            )

        # will be set in compute_step_model
        self.clip_norm = 0.0
        self.clip_norm_factor = 1.0
        self.gradient_scaling = 1.0

        self.load_train_batch()
        print(f"Loaded batch from {self.checkpoint_path}/batch.pt")

        self.lr = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]

        # self.test_instances = None
        # self.train_instances = None

        rx = re.compile(r"checkpoint-(\d+)-(\d+)")
        self.step_id = int(rx.search(os.path.basename(self.checkpoint_path)).group(2))

        # mainly here for debugging purposes:
        self.train_grad_scaling = 1.0 # / self.gradient_accumulation_steps  # self.gradient_accumulation_steps
        self.test_grad_scaling = 1.0 # self.gradient_accumulation_steps
        self.test_loss_scaling = 1.0 / self.gradient_accumulation_steps

    
    def load_train_batch(self):
        if not self.recovered_batches_path:
            train_batch = torch.load(self.checkpoint_path + "/batch.pt")
        else:
            train_batch = torch.load(self.recovered_batches_path)
       
        # turn dict into a list of the batches
        # for this model, we have historically stored batches as a list of lists (one per gpu) of dicts (one per batch)
        if 0 not in train_batch.keys():
            # for single batch+acc case
            train_batch = [[train_batch]]
        
        list_of_samples = [train_batch[rank] for rank in range(0, len(train_batch))]
        # concat lists
        list_of_samples = sum(list_of_samples, [])
        
        self.postprocess_train_batch(list_of_samples)


    def postprocess_train_batch(self, list_of_samples):
        # we need to populate the train_instances field that holds the train batch + simulated instances used for 
        # model update step and score computations

        if self._split_instances_by_type:
            # split instances by language
            self.train_instances = self._split_instances_by_type(list_of_samples)
        else: 
            self.train_instances = dict([(f"train_sample_{i}", batch) for i, batch in enumerate(list_of_samples)])

           
        if self.additional_train_instances and not self.debug:
            print("Adding additional train instances...")

            self.additional_tokenized = self.tokenize_and_collate(self.additional_train_instances, max_batch_len=self.max_batch_len)
            for k, v in self.additional_tokenized.items():
                key = f"*{k}"
                batch = v

                # is this necessary?
                batch[self.type_field] = torch.tensor([-1] * v['input_ids'].shape[1], device=self.model.device)

                assert key not in self.train_instances.keys(), f"Key {key} already exists in train instances, cannot add additional instance with same key."

                self.train_instances[key] = batch
                print(f"Added additional train instance: {key} of len {batch['input_ids'].shape[1]}")

        if self.debug:
            self.train_instances = {f"debug_{i}": batch for i, batch in enumerate(list(self.train_instances.values())[:10])}
            print("Debug mode: using only first 10 train batches.")

        if self.gradient_accumulation_steps != len(self.train_instances):
            # reset acc steps etc
            self.accelerator.gradient_accumulation_steps = len(self.train_instances)
            self.gradient_accumulation_steps = len(self.train_instances)
            self.test_loss_scaling = 1.0 / self.gradient_accumulation_steps
            print(f"Adjusted gradient accumulation steps to match number of train batches: {self.gradient_accumulation_steps}")

        print(f"Total number of train batches: {len(self.train_instances)}")

        # self.train_batch["type"] = self.train_batch[self.type_field].tolist()
        for key, batch in self.train_instances.items():
            batch['labels'] = batch['input_ids'].to(self.model.device)
            if "target_ids" in batch:
                batch['target_ids'] = batch['input_ids'].to(self.model.device)
            batch['position_ids'] = batch['position_ids'].to(self.model.device)
            batch['cu_seq_lens_q'] = batch['cu_seq_lens_q'].to(self.model.device)
            batch['cu_seq_lens_k'] = batch['cu_seq_lens_k'].to(self.model.device)
            batch[self.type_field] = batch[self.type_field].to(self.model.device)
            batch['input_ids'] = batch['input_ids'].to(self.model.device)
            self.train_instances[key] = batch

    def _combine_instances(self, batcher: list):
        # make single dict of lang_id -> batch, do not duplicate lang_ids
        combined = {}
        for i, batch in enumerate(batcher):
            for lang_id, instance in batch.items():
                combined_key = f"lang_{lang_id}"
                combined[combined_key] = instance
        return combined

    def _split_instances_by_type(self, batcher: list):
        """
        For llama, we have batches that are already packed by language (but may contain multiple languages if packing was done across languages).
        We want to split these into separate batches by language, so that we can do language-specific upweighting and analysis.

        Also should work for other types by which one would like to split train instances.

        Note: Current implementation is pretty messy and involves a lot of copying of tensors, which is not ideal. We can optimize this later if necessary, 
        but for now we prioritize correctness and getting something working. Probably, there is a much simpler, less repetitious solution.
        """
        #lang_ids = batch[self.type_field]
        #unique_langs = torch.unique(lang_ids).tolist()
        instances = {}
        lengths = {}  # in case we downsample, keep relative num tokens stable across langs

        # iterate over all batches and combine into big per-lang batches
        for batch in batcher:
            
            # if lang is -1, ignore
            if torch.all(batch[self.type_field] == -1):
                continue
            
            # split packed batch by position ids and combine by lang ids
            for i in range(0, len(batch['cu_seq_lens_q']) - 1):

                langt = batch[self.type_field][i]
                lang = langt.item()
                id0, id1 = batch['cu_seq_lens_q'][i].item(), batch['cu_seq_lens_q'][i+1].item()
                input_ids = batch['input_ids'][:,id0:id1]
                position_ids = batch['position_ids'][:,id0:id1]
                labels = batch['labels'][:,id0:id1]
                target_ids = None
                if 'target_ids' in batch:
                    target_ids = batch['target_ids'][:,id0:id1]
                num_batch_samples = None
                if 'num_batch_samples' in batch:
                    num_batch_samples = batch['num_batch_samples'] # , torch.tensor([1.0], device=batch['input_ids'].device))

                if lang not in instances.keys():
                    instances[lang] = {
                        'input_ids': input_ids,
                        'position_ids': position_ids,
                        'labels': labels,
                        'cu_seq_lens_q': torch.tensor([0, id1 - id0], device=batch['cu_seq_lens_q'].device),
                        'cu_seq_lens_k': torch.tensor([0, id1 - id0], device=batch['cu_seq_lens_k'].device),
                        self.type_field: torch.tensor([langt], device=batch[self.type_field].device),
                        'instance_len': id1 - id0,

                    }
                    if target_ids is not None:
                        instances[lang]['target_ids'] = target_ids
                    if num_batch_samples is not None:
                        instances[lang]['num_batch_samples'] = num_batch_samples
                    lengths[lang] = input_ids.shape[1]
                else:
                    # print(instances[lang]['input_ids'].shape)
                    # print(input_ids.shape)
                    instances[lang]['input_ids'] = torch.cat([instances[lang]['input_ids'], input_ids], dim=1)
                    instances[lang]['position_ids'] = torch.cat([instances[lang]['position_ids'], position_ids], dim=1)
                    instances[lang]['labels'] = torch.cat([instances[lang]['labels'], labels], dim=1)
                    if target_ids is not None:
                        instances[lang]['target_ids'] = torch.cat([instances[lang]['target_ids'], target_ids], dim=1)
                    last_cu_q = instances[lang]['cu_seq_lens_q'][-1].item()
                    last_cu_k = instances[lang]['cu_seq_lens_k'][-1].item()
                    instances[lang]['cu_seq_lens_q'] = torch.cat(
                        [instances[lang]['cu_seq_lens_q'], torch.tensor([last_cu_q + (id1 - id0)], device=batch['cu_seq_lens_q'].device)],
                        dim=-1
                    )
                    instances[lang]['cu_seq_lens_k'] = torch.cat(
                        [instances[lang]['cu_seq_lens_k'], torch.tensor([last_cu_k + (id1 - id0)], device=batch['cu_seq_lens_k'].device)],
                        dim=-1
                    )
                    if num_batch_samples is not None:
                        instances[lang]['num_batch_samples'] = torch.cat([instances[lang]['num_batch_samples'], num_batch_samples], dim=0)
                    instances[lang]['instance_len'] += (id1 - id0)
                    lengths[lang] += input_ids.shape[1]

        # if self.num_train_batch_per_lang_max > 0 and self.num_train_batch_per_lang_min >= 0 and self.limit_batches:
        if self.limit_update_batches and self.train_tok_frac > 0 and self.train_tok_frac < 1.0:
            # assign largest lang (by num tokens/total len) max num of batches
            # determine fractions of that for other langs
            # assign multiple of 4000, batched, for each lang
            print("Limiting number of batches per lang...")
            langs_sorted_by_num_toks = list(lengths.keys())
            langs_sorted_by_num_toks = sorted(langs_sorted_by_num_toks, key=lambda x: lengths[x], reverse=True)

            num_toks = {}

            for lang in langs_sorted_by_num_toks:
                num_toks[lang] = int(lengths[lang] * self.train_tok_frac)

            for lang in langs_sorted_by_num_toks:
                print(f"{lang}: {num_toks[lang]}")

            # limit batches to that amount, throw away rest
            for lang in langs_sorted_by_num_toks:
                instances[lang]['input_ids'] = instances[lang]['input_ids'][:, :num_toks[lang]]
                instances[lang]['position_ids'] = instances[lang]['position_ids'][:, :num_toks[lang]]
                instances[lang]['labels'] = instances[lang]['labels'][:, :num_toks[lang]]
                if 'target_ids' in instances[lang]:
                    instances[lang]['target_ids'] = instances[lang]['target_ids'][:, :num_toks[lang]]
                instances[lang]['cu_seq_lens_q'] = instances[lang]['cu_seq_lens_q'][instances[lang]['cu_seq_lens_q'] <= num_toks[lang]]
                instances[lang]['cu_seq_lens_k'] = instances[lang]['cu_seq_lens_k'][instances[lang]['cu_seq_lens_k'] <= num_toks[lang]]
                if 'num_batch_samples' in instances[lang]:
                    instances[lang]['num_batch_samples'] = instances[lang]['num_batch_samples'][:num_toks[lang]]

                # do we need to fix last position for any of these?
                if instances[lang]['cu_seq_lens_q'][-1] < num_toks[lang]:
                    last_cu_q = instances[lang]['cu_seq_lens_q'][-1].item()
                    instances[lang]['cu_seq_lens_q'] = torch.cat(
                        [instances[lang]['cu_seq_lens_q'], torch.tensor([num_toks[lang]], device=instances[lang]['cu_seq_lens_q'].device)],
                        dim=-1
                    )
                    last_cu_k = instances[lang]['cu_seq_lens_k'][-1].item()
                    instances[lang]['cu_seq_lens_k'] = torch.cat(
                        [instances[lang]['cu_seq_lens_k'], torch.tensor([num_toks[lang]], device=instances[lang]['cu_seq_lens_k'].device)],
                        dim=-1
                    )

        # put back into batches
        # batched_instances = {}
        result = {}
        for lang in lengths.keys():
            # create batches of max_batch_len tokens up until end of sequences
            # batched_instances[lang] = []
            current_batch = {
                'input_ids': [],
                'position_ids': [],
                'labels': [],
                'cu_seq_lens_q': [0],
                'cu_seq_lens_k': [0],
                self.type_field: torch.tensor([lang], device=instances[lang]['cu_seq_lens_q'].device),
            }
            if 'target_ids' in instances[lang]:
                current_batch['target_ids'] = []
            if 'num_batch_samples' in instances[lang]:
                current_batch['num_batch_samples'] = []
            current_len = 0
            counter = 0
            id0, id1 = 0, 0
            for i in range(1, len(instances[lang]['cu_seq_lens_q'])):
                # id0 = instances[lang]['cu_seq_lens_q'][i].item()
                id0 = id1
                id1 = instances[lang]['cu_seq_lens_q'][i].item()
                seq_len = id1 - id0

                # print(seq_len)

                if current_len + seq_len > self.max_batch_len and current_len > 0:
                    # finalize current batch
                    current_batch['cu_seq_lens_q'] = torch.tensor(current_batch['cu_seq_lens_q'], device=instances[lang]['cu_seq_lens_q'].device)
                    current_batch['cu_seq_lens_k'] = torch.tensor(current_batch['cu_seq_lens_k'], device=instances[lang]['cu_seq_lens_k'].device)
                    current_batch['input_ids'] = torch.cat(current_batch['input_ids'], dim=1)
                    current_batch['position_ids'] = torch.cat(current_batch['position_ids'], dim=1)
                    current_batch['labels'] = torch.cat(current_batch['labels'], dim=1)
                    if 'target_ids' in current_batch:
                        current_batch['target_ids'] = torch.cat(current_batch['target_ids'], dim=1)
                    if 'num_batch_samples' in current_batch:
                        current_batch['num_batch_samples'] = torch.cat(current_batch['num_batch_samples'], dim=0)
                    result[f"{lang}_{counter}"] = deepcopy(current_batch)
                    #current_batch = {f"{lang}_{counter}" : current_batch}
                    #batched_instances.append(current_batch)
                    counter += 1

                    # start new batch
                    current_batch = {
                        'input_ids': [],
                        'position_ids': [],
                        'labels': [],
                        'cu_seq_lens_q': [0],
                        'cu_seq_lens_k': [0],
                        self.type_field: torch.tensor([lang], device=instances[lang]['cu_seq_lens_q'].device),
                    }
                    if 'target_ids' in instances[lang]:
                        current_batch['target_ids'] = []
                    if 'num_batch_samples' in instances[lang]:
                        current_batch['num_batch_samples'] = []
                    current_len = 0

                # add sequence to current batch
                # print(id0, id1)
                current_batch['input_ids'].append(instances[lang]['input_ids'][:, id0:id1])
                current_batch['position_ids'].append(instances[lang]['position_ids'][:, id0:id1])
                current_batch['labels'].append(instances[lang]['labels'][:, id0:id1])
                if 'target_ids' in instances[lang]:
                    current_batch['target_ids'].append(instances[lang]['target_ids'][:, id0:id1])
                last_cu_q = current_batch['cu_seq_lens_q'][-1]
                last_cu_k = current_batch['cu_seq_lens_k'][-1]
                current_batch['cu_seq_lens_q'].append(last_cu_q + seq_len)
                current_batch['cu_seq_lens_k'].append(last_cu_k + seq_len)
                if 'num_batch_samples' in instances[lang]:
                    current_batch['num_batch_samples'].append(instances[lang]['num_batch_samples'][id0:id1])
                current_len += seq_len

            # finalize last batch
            if len(current_batch['input_ids']) > 0:
                current_batch['cu_seq_lens_q'] = torch.tensor(current_batch['cu_seq_lens_q'], device=instances[lang]['cu_seq_lens_q'].device)
                current_batch['cu_seq_lens_k'] = torch.tensor(current_batch['cu_seq_lens_k'], device=instances[lang]['cu_seq_lens_k'].device)
                current_batch['input_ids'] = torch.cat(current_batch['input_ids'], dim=1)
                current_batch['position_ids'] = torch.cat(current_batch['position_ids'], dim=1)
                current_batch['labels'] = torch.cat(current_batch['labels'], dim=1)
                if 'target_ids' in current_batch:
                    current_batch['target_ids'] = torch.cat(current_batch['target_ids'], dim=1)
                if 'num_batch_samples' in current_batch:
                    current_batch['num_batch_samples'] = torch.cat(current_batch['num_batch_samples'], dim=0)
                result[f"{lang}_{counter}"] = deepcopy(current_batch)
                #current_batch = {f"{lang}_{counter}" : current_batch}
                #batched_instances.append(current_batch)    

        # print(result.keys())
        print({k: v['input_ids'].shape for k, v in result.items()})
        # print total number of tokens
        total_tokens = 0
        for k, v in result.items():
            total_tokens += v['input_ids'].shape[1]
        print(f"Total number of tokens in split train instances: {total_tokens}")

        # compare against batcher
        original_tokens = 0
        for batch in batcher:
            original_tokens += batch['input_ids'].shape[1]
        print(f"Total number of tokens in original train batcher: {original_tokens}")
        return result

    def _perform_update_step(self):

        self.model.train()

        self.model_pre = deepcopy(self.model)

        losses = []
        for key, batch in tqdm(deepcopy(list(self.train_instances.items())[::-1]), desc="Performing parameter update accumulating gradients..."):
            
            with self.accelerator.accumulate(self.model):

                # forward and backward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                # add batch len to batch, so we can use it for correclty scaling later
                batch['num_batch_samples'] = torch.tensor([batch['input_ids'].shape[1] - 1  for _ in range(batch['input_ids'].shape[1])], device=self.model.device)
                self.train_instances[key] = batch

                if self.accelerator.sync_gradients:
                    print('clipping gradients')
                    
                    parameters = [p for p in self.model.parameters() if p.requires_grad]
                    self.clip_norm = torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None])
                    self.clip_norm_factor = self.max_grad_norm / (self.clip_norm + 1e-6)
                    self.gradient_scaling = torch.clamp(self.clip_norm_factor, max=self.max_grad_norm)

                    torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), self.max_grad_norm, self.clip_norm)
                    # self.accelerator.clip_grad_value_(self.model.parameters(), self.gradient_scaling)
                    print(f'pre-clip norm: {self.clip_norm}, clip factor: {self.clip_norm_factor}, applied scaling: {self.gradient_scaling}')
                else:
                    # assert params did not change
                    assert all(torch.equal(p, p_pre) for p, p_pre in zip(self.model.parameters(), self.model_pre.parameters())), "Parameters changed during accumulation step!"
                
                # update parameters (only triggers if accumulation steps reached)
                self.optimizer.step()

                # we do not actually need to step scheduler
                #if self.scheduler is not None:
                #     self.scheduler.step()

                losses.append(loss.detach().item())

        # print(losses)
        print(f"Accumulated training loss: {sum(losses)}")
        print(f"Average training loss: {sum(losses) / len(losses)}")
        self.optimizer.zero_grad()
        
        torch.cuda.empty_cache()

        self._set_test_and_train_instances()

        return loss.item()

    def _set_test_and_train_instances(self):

        if self.test_instances_path and not self.test_instances:
            print("Loading test instances...")
            # load test instances
            with open(self.test_instances_path, "r") as f:
                test_instances_text = json.load(f)

            self.test_instances = self.tokenize_and_collate(test_instances_text, max_batch_len=self.max_batch_len, max_batch_per_label=1)
            print(f"Loaded {len(self.test_instances)} test instances from {self.test_instances_path}")

        elif not self.test_instances:
            # self.test_instances = {}
            print("Using copies of train instances as test instances.")
            self.test_instances = {"copy_of_first_train_batch" : list(self.train_instances.values())[0]}
            if len(list(self.train_instances.values())) > 1:
               # add another
                self.test_instances["copy_of_second_train_batch"] = list(self.train_instances.values())[1]

        if self.limit_train_batches:
            self.train_instances = self._stratified_filter(self.train_instances)

        if self.debug:
            # take only first 2 test instances
            keys = list(self.test_instances.keys())
            keys = keys[:2]
            self.test_instances = {k: self.test_instances[k] for k in keys}

    def _stratified_filter(self, instances):
        # limit number of train batches to self.train_tok_frac
        # make sure distribution over the types is preserved as much as possible
        total_len = sum([batch['input_ids'].shape[1] for batch in instances.values()])
        target_len = int(total_len * self.train_tok_frac)
        print(f"Limiting train batches to {target_len} tokens (fraction {self.train_tok_frac} of total {total_len} tokens)")
        # keys are type+_+id, values are batches
        # sort by type
        instances_by_type = {}
        for key, batch in instances.items():
            data_type = re.match(r"(.+)_\d+", key).group(1)
            if data_type not in instances_by_type.keys():
                instances_by_type[data_type] = []
            instances_by_type[data_type].append((key, batch))
        # determine how many tokens to take from each type
        tokens_by_type = {}
        for data_type, batches in instances_by_type.items():
            tokens_by_type[data_type] = sum([batch['input_ids'].shape[1] for _, batch in batches])
        # take fraction of tokens from each type
        filtered_instances = {}
        for data_type, batches in instances_by_type.items():
            if data_type.startswith("*"):
                print(f"Keeping all instances of type {data_type} (total {tokens_by_type[data_type]} tokens)")
                for key, batch in batches:
                    filtered_instances[key] = batch
                continue
            target_len_type = int(tokens_by_type[data_type] * self.train_tok_frac)
            print(f"Limiting type {data_type} to {target_len_type} tokens (fraction {self.train_tok_frac} of total {tokens_by_type[data_type]} tokens)")
            # if it begins with * it's an instance we have added and want to keep
            current_len = 0
            for key, batch in batches:
                batch_len = batch['input_ids'].shape[1]
                filtered_instances[key] = batch
                current_len += batch_len
                if current_len >= target_len_type:
                    break
        
        print(f"Total tokens after filtering: {sum([batch['input_ids'].shape[1] for batch in filtered_instances.values()])}")
        print(f"Total batches after filtering: {len(filtered_instances)}")
        return filtered_instances

    def compute_step_model(self):
        """
        Perform one update step.
        """
        _ = self._perform_update_step()
        self.wrapped_model = self._wrap_model("", self.model_pre, self.model)

    def _collate_packing(self, batch):
        for b in batch:
            if "labels" in b.keys() and isinstance(b["labels"], torch.Tensor):
                # for some reason expects list??
                b["labels"] = b["labels"].tolist()

        data_collator = DataCollatorWithFlattening(return_tensors="pt", separator_id=self.tokenizer.pad_token_id, return_flash_attn_kwargs=True)
        packed = data_collator(batch)
        # attach languages 
        packed[self.type_field] = torch.tensor(
                [ex.get(self.type_field, -1) for ex in batch], dtype=torch.long
            )
        packed={
            "input_ids": packed["input_ids"].to(self.model.device),
            "labels": packed["labels"].to(self.model.device),
            "position_ids": packed["position_ids"].to(self.model.device),
            "cu_seq_lens_q": packed["cu_seq_lens_q"].to(self.model.device),
            "cu_seq_lens_k": packed["cu_seq_lens_k"].to(self.model.device),
            self.type_field: packed[self.type_field].to(self.model.device)
        }
        return packed

    
    def _collate_padded(self, batch):
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
        type_ids = torch.tensor([ex.get(self.type_field, -1) for ex in batch],  dtype=torch.long)
        # dynamic padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = input_ids.masked_fill(attention_mask == 0, -100)

        return {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
            "labels": labels.to(self.model.device),
            self.type_field: type_ids.to(self.model.device)
        }
    
    def tokenize_and_collate(self, text_instances, max_batch_len=-1, max_batch_per_label=-1):
        tokenized = {}

        for label, samples in text_instances.items():
            # tokenize texts and collate to create model-compatible batch
            tokenized_samples = []
            current_batch_len = 0
            i = 0
            for sample in samples:
                
                if isinstance(sample, str):
                    tokenized_samples.append(self.tokenizer(sample))
        
                elif isinstance(sample, dict):
                    # infer what to do with sample
                    prediction_type = "whole_sequence"
                    if "prediction_type" in sample:
                        prediction_type = sample["prediction_type"]
                    
                    if prediction_type == "whole_sequence":
                        # same as above, just tokenize and collate list of dicts with "text" field
                        tokenized_samples.append(self.tokenizer(sample["text"]))

                    elif prediction_type == "targets":
                        # tokenize text, but create labels only for specified tokens (rest should be -100)
                        tokenized_samples.append(self.tokenize_tok_preds(sample["text"]))
                
                current_batch_len += len(tokenized_samples[-1]['input_ids'])

                if current_batch_len >= max_batch_len and max_batch_len > 0:
                    # collate current batch and start new one
                    collated = self._collate_packing(tokenized_samples[:-1])
                    tokenized[f"{label}_{i}"] = collated
                    tokenized_samples = tokenized_samples[-1:]
                    current_batch_len = len(tokenized_samples[0]['input_ids'])
                    i += 1

                if i >= max_batch_per_label and max_batch_per_label > 0:
                    break

            if len(tokenized_samples) > 0:
                collated = self._collate_packing(tokenized_samples)
                # throw away very small batches
                if collated['input_ids'].shape[1] > 0.5 * max_batch_len or max_batch_len <= 0 or i == 0:
                    tokenized[f"{label}_{i}"] = collated
        
        return tokenized

    def tokenize_tok_preds(self, text, ignore_index=-100, bot=r"\[TARGET\]", eot=r"\[\/TARGET\]"):
        # tokenize text, but create labels only for specified tokens (rest should be -100)
        # text_instances contain text where the special tokens [TARGET] <target_sequence> [/TARGET] indicate a subsequence for which we want to compute a next token prediction score

        # split at [TARGET] (bot) and [/TARGET] (eot) to find target sequences (could be multiple)
        segments = re.split(f"({bot}.*{eot})", text)

        # tokenize one by one, marking labels as -100 except for tokens immediately following target sequences
        input_ids = []
        attention_mask = []
        labels = []
        for i, segment in enumerate(segments):
            m = re.match(f"{bot}(.*){eot}", segment)
            if m:
                # this is a target segment, we want to predict the token immediately following it
                target_text = m.group(1)
                
                tokenized = self.tokenizer(target_text, add_special_tokens=False)
                target_tokens = tokenized["input_ids"]

                input_ids.extend(target_tokens)
                attention_mask.extend(tokenized["attention_mask"])

                # the tokens that are part of the sequence should be predicted
                labels[-1] = target_tokens[0]
                labels.extend(target_tokens[1:] + [-100])  # predict next token after target sequence, shift labels by 1
        
            else:
                # non-target segment, just tokenize normally with labels -100 (will not contribute to loss)
                tokenized = self.tokenizer(segment, add_special_tokens=False)
                tokens = tokenized["input_ids"]
                input_ids.extend(tokens)
                attention_mask.extend(tokenized["attention_mask"])
                labels.extend([-100] * len(tokens))

        # add bos and eos tokens if necessary
        if self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            attention_mask = [1] + attention_mask
            labels = [-100] + labels

        if self.tokenizer.eos_token_id is not None:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
            attention_mask = attention_mask + [1]
            labels = labels + [-100]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

        


        