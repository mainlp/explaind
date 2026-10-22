from explaind.accelerated.llama_explainer import LlamaExplainer
from typing import Optional
import torch
from transformers import AutoConfig, AutoTokenizer
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM
from huggingface_hub import HfApi
from accelerate import Accelerator
import os
import re
import glob
import json


LANGS = [
    "eng_Latn",
    "deu_Latn",
    "spa_Latn",
    "fra_Latn",
    "ita_Latn",
    "por_Latn",
    "cmn_Hani",
    "jpn_Jpan",
    # "ind_Latn",
]

NEEDED_FWHQ = [
    "rus_Cyrl",
    "pol_Latn",
    "nld_Latn",
    "arb_Arab",
    "tur_Latn",
    "swe_Latn",
    "ces_Latn",
    "ell_Grek",
    ""
]

NEEDED_HPLT = [
    "kor_Hang",
    "hin_Deva",
    "hun_Latn",
    "ron_Latn",
    "fin_Latn",
    "ukr_Cyrl",
    "slk_Latn",
    "dan_Latn",
    "lit_Latn",
    "lvs_Latn",
    "slv_Latn",
    "est_Latn",
    "bul_Cyrl",
    "nob_Latn",
    "cat_Latn",
    "hrv_Latn",
    "gle_Latn",
    "mlt_Latn",
    "glg_Latn",
]

ID2LANG = {i: lang for i, lang in enumerate(LANGS + NEEDED_FWHQ + NEEDED_HPLT)}
LANG2ID = {lang: i for i, lang in ID2LANG.items()}


###################### UTILS ####################################
# largely copied from our evaluation code, see: TODO
ELLM_PATH = "/nfs/gdata/fkoerner/conversion/converted/eurollm/1b"
ELLM_P2_OFFSET = 287000
ELLM_PHASE_MAPPING = {
    "train_annealing_megatronlm": "phase_1",
    "second_phase_annealing_more_multilingual_fw": "phase_2",
}
MODEL_PATHS = glob.glob(f"{ELLM_PATH}/*/*")


def get_step_number(path):
    if re.search(r"step(\d+)", path):
        return int(re.search(r"step(\d+)", path).group(1))
    elif re.search(r"iter_(\d+)", path):
        match = re.search(r"(.*/)([^/]*)(/iter_\d+)", path)
        if match:
            phase = match.group(2)
            phase = ELLM_PHASE_MAPPING[phase]
        iteration = re.search(r"iter_(\d+)", path).group(1)
        return f"{phase}_{iteration}" if match else iteration
    else:
        return int("inf")

def sort_key(s):
    match = re.match(r"phase_(\d+)_(\d+)", s)
    if match:
        phase, number = match.groups()
        phase = int(phase)
        number = int(number)
        if phase == 2:
            number += ELLM_P2_OFFSET # offset phase 2
        return number
    else:
        return 0

def get_all_refs(model_name: str):
    refs = HfApi().list_repo_refs(model_name, repo_type="model")
    ref_names = []
    for branch in refs.branches:
        if "Apertus" in model_name and "longctx" in branch.name:
            continue
        ref_names.append(branch.name)
    # fallback to tag names
    if len(ref_names) < 2:
        for tag in refs.tags:
            if "Apertus" in model_name and "longctx" in tag.name:
                continue
            ref_names.append(tag.name)
    return ref_names

def build_ellm_model(conv_path, ckpt_path, model_name="utter-project/EuroLLM-1.7B",):

    # step_num = get_step_number(model_name)
    config = AutoConfig.from_pretrained(model_name)

    step_num = int(conv_path.split("_")[-1])
    if "second_phase" in conv_path:
        step_num += ELLM_P2_OFFSET

    model = AutoModelForCausalLM.from_pretrained(conv_path,)
    ## load optimizer state
    # we have two param groups that we have to recreate
    param_groups = [{"params": []}, {"params": []}]

    for np in model.named_parameters():
        name, param = np
        if name.endswith("input_layernorm.weight") or name.endswith("post_attention_layernorm.weight") or name == "model.norm.weight":
            param_groups[1]["params"].append(np)
        else:
            param_groups[0]["params"].append(np)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    optimizer = torch.optim.AdamW(param_groups)

    # for some reason megatron stored the weights of attention and mlp layers combined into single tensors
    # we need to revert this combination to load the optimizer state correctly
    torch_state = revert_megatron_param_combinations(state["optimizer"]["optimizer"]["state"], check_against=optimizer.state_dict()["param_groups"])
    ckpt_param_groups = state["optimizer"]["optimizer"]["param_groups"]

    
    # we use a dummy scheduler as we cannot exactly recreate the original one here
    scheduler_params = state['opt_param_scheduler']
    def scheduler_rule(step):
        max_lr = scheduler_params['max_lr']
        min_lr = scheduler_params['min_lr']
        warmup = scheduler_params['lr_warmup_steps'] / 3072  # batch size (account for some megatron fuckery with the steps)
        if step < warmup:
            return step / warmup

        decay = scheduler_params["lr_decay_steps"]
        if step < decay:
            return 1.0

        if scheduler_params["lr_decay_style"] == "linear":
            return max(0, 1.0 - (step - decay) / warmup)  # warmup steps is the same number as decay steps in EuroLLM

        # else: constant
        return min_lr / max_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: scheduler_rule(s))

    torch_param_groups = optimizer.state_dict()["param_groups"]

    # update with values from ckpt
    for i in range(len(ckpt_param_groups)):
        torch_param_groups[i]["lr"] = ckpt_param_groups[i]["lr"]
        torch_param_groups[i]["betas"] = ckpt_param_groups[i]["betas"]
        torch_param_groups[i]["eps"] = ckpt_param_groups[i]["eps"]
        torch_param_groups[i]["weight_decay"] = ckpt_param_groups[i]["weight_decay"]
        torch_param_groups[i]["step"] = step_num
        
    for i in range(len(torch_state)):
        torch_state[i]["step"] = step_num

    correct_state_dict = {
        "state": torch_state,
        "param_groups": torch_param_groups,
    }
    optimizer.load_state_dict(correct_state_dict)

    ### load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer_name = tokenizer.name_or_path
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, optimizer, scheduler, config

def revert_megatron_param_combinations(state, check_against):
    # this is a hacky solution for the model at hand and would not generalize
    group0 = []
    group1 = []
    for k, v in state.items():
        if v["exp_avg"].shape[0] == 11264:
            # mlp combined
            group0.append({
                "exp_avg": v["exp_avg"][:5632, :],
                "exp_avg_sq": v["exp_avg_sq"][:5632, :],
            })
            group0.append({
                "exp_avg": v["exp_avg"][5632:, :],
                "exp_avg_sq": v["exp_avg_sq"][5632:, :],
            })
        elif v["exp_avg"].shape[0] == 4096 and v["exp_avg"].shape[1] == 2048:
            # attention combined
            group0.append({
                "exp_avg": v["exp_avg"][:2048, :],
                "exp_avg_sq": v["exp_avg_sq"][:2048, :],
            })
            group0.append({
                "exp_avg": v["exp_avg"][2048:3072, :],
                "exp_avg_sq": v["exp_avg_sq"][2048:3072, :],
            })
            group0.append({
                "exp_avg": v["exp_avg"][3072:, :],
                "exp_avg_sq": v["exp_avg_sq"][3072:, :],
            })
        else:
            if v["exp_avg"].shape[0] == 2048 and len(v["exp_avg"].shape) == 1:
                group1.append(v)
            else:
                group0.append(v)

    # check shapes match
    new_state = dict([(i, group0[i]) for i in range(len(group0))])
    new_state.update(dict([(i + len(group0), group1[i]) for i in range(len(group1))]))

    for i in range(len(group0)):
        print(f"Checking param {i} shape: {group0[i]['exp_avg'].shape} - {check_against[0]["param_names"][i]}")
        
    for i in range(len(group1)):
        print(f"Checking param {i} shape: {group1[i]['exp_avg'].shape} - {check_against[1]["param_names"][i]}")
        
    return new_state



    
###################### END UTILS ####################################


class EuroLLMExplainer(LlamaExplainer):
    def __init__(self, 
                 *args,
                 conv_checkpoint_path: str,
                 trust_remote_code: bool = False,
                 revision: Optional[str] = None,
                 **kwargs):

        #self.checkpoint_path = model_checkpoint
        self.trust_remote_code = trust_remote_code
        self.conv_checkpoint_path = conv_checkpoint_path
        self.revision = revision
        self.kwargs_ = kwargs

        self.test_instances = None
        self.train_instances = None

        super().__init__(*args, **kwargs)

    
    def load_checkpoint(self):
        """
        Load checkpoints.
        """
        self.accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
        self.model, self.tokenizer, self.optimizer, self.scheduler, self.config = build_ellm_model(
            self.conv_checkpoint_path,
            self.checkpoint_path,
            model_name=self.kwargs_.get("model_name", "utter-project/EuroLLM-1.7B"),
        )

        self.model, self.optimizer, self.scheduler, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.dataloader)

        # will be set in compute_step_model
        self.clip_norm = 0.0
        self.clip_norm_factor = 1.0
        self.gradient_scaling = 1.0

        self.load_train_batch()
        print(f"Loaded batch from {self.checkpoint_path} and {self.conv_checkpoint_path}")

        self.lr = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]

        rx = re.compile(r"iter-(\d+)")
        self.step_id = self.optimizer.param_groups[0]['step']

        # mainly here for debugging purposes:
        self.train_grad_scaling = 1.0 # / self.gradient_accumulation_steps  # self.gradient_accumulation_steps
        self.test_grad_scaling = 1.0 # self.gradient_accumulation_steps
        self.test_loss_scaling = 1.0 / self.gradient_accumulation_steps
    
    def load_train_batch(self):
        """
        Load training batch from recovered batches path.
        """
        if self.recovered_batches_path is None:
            raise ValueError("Recovered batches path is not provided.")

        batch_path = self.recovered_batches_path
        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"Recovered batch file not found at {batch_path}")

        batch_samples = []

        with open(self.recovered_batches_path, "r") as f:
            for line in f:
                dict_sample = json.loads(line.strip())
                batch_samples.append(dict_sample)
        
        self.train_batcher = []
        current_batch = []
        current_len = 0
        for sample in batch_samples:
            text = sample["text"]
            batch = self.tokenizer(
                text,
                padding="do_not_pad",
                truncation=True,
                max_length=self.max_batch_len,
            )
            batch["language_ids"] = torch.tensor(LANG2ID[sample.get("lang_code", "unk")])
            current_len += len(batch['input_ids'])

            if current_len >= self.max_batch_len:
                # collate
                collated_batch = self._collate_packing(current_batch)
                self.train_batcher.append(collated_batch)
                current_len = 0
                current_batch = []
            
            current_batch.append(batch)
        # also add final batch
        collated_batch = self._collate_packing(current_batch)

        self.train_batcher.append(collated_batch)
        self.train_instances = {f"batch_{i}": batch for i, batch in enumerate(self.train_batcher)}

        # self.train_batch = [self.train_batcher]  # this is legacy multi rank training stuff, perhaps should be fixed at some point
        # do additional postprocessing that we can re-use from the llama explainer again
        self.postprocess_train_batch()
        