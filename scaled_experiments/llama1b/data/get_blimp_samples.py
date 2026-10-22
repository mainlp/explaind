import os
from datasets import load_from_disk
import json
import random

import sys
sys.path.append(".")


MULTI_BLIMP_PATH = None
MULTI_BLIMP_LANGS = None # needs to be set


def load_and_sample_multi_blimp(
    dataset_path=MULTI_BLIMP_PATH,
    tokenizer_name="",
    untokenized=True,
    output_prefix="results/blimp_scores/",
    langs=MULTI_BLIMP_LANGS,
    force=True,
    batch_size=128,
    seed=42,
    max=4096,
):
    print(MULTI_BLIMP_PATH)
    print(MULTI_BLIMP_LANGS)
    random.seed(seed)
    if not untokenized:
        results_path = output_prefix + f"blimp_samples_{tokenizer_name}.json"
        additional_results_path = output_prefix + f"blimp_additional_samples_{tokenizer_name}.json"
    else:
        results_path = output_prefix + "blimp_samples.json"
        additional_results_path = output_prefix + "blimp_additional_samples.json"
    # full_data_path = output_prefix + "blimp_dataset.json"
    if os.path.exists(results_path) and not force:
        print(
            f"Found existing MultiBLiMP sample at {results_path}, skipping sampling."
        )
        return

    # datasets = {}
    samples = {}
    additional_samples = {}

    for lang in langs:

        tokenizer_name = tokenizer_name.replace("/", "_")

        if not untokenized:
            lang_specific_path = f"{dataset_path}_{lang}_{tokenizer_name}"
        else:
            lang_specific_path = f"{dataset_path}_{lang}"

        if not os.path.exists(lang_specific_path):
            print(
                f"MultiBLiMP dataset for lang {lang} with tokenizer {tokenizer_name} not found at {lang_specific_path}, skipping."
            )
            continue

        dataset = load_from_disk(lang_specific_path)
        sample = random.sample(range(len(dataset)), k=batch_size) 

        rest = list(set(range(len(dataset))) - set(sample))
        additional_sample = random.sample(rest, k=batch_size)

        if not untokenized:
            samples[lang + "_input_ids_good"] = [dataset[i]["input_ids_good"] for i in sample]
            samples[lang + "_input_ids_bad"] = [dataset[i]["input_ids_bad"] for i in sample]
            additional_samples[lang + "_input_ids_good"] = [dataset[i]["input_ids_good"] for i in additional_sample]
            additional_samples[lang + "_input_ids_bad"] = [dataset[i]["input_ids_bad"] for i in additional_sample]
        else:
            samples[lang + "_sen_good"] = [dataset[i]["sen"] for i in sample]
            samples[lang + "_sen_bad"] = [dataset[i]["wrong_sen"] for i in sample]
            additional_samples[lang + "_sen_good"] = [dataset[i]["sen"] for i in additional_sample]
            additional_samples[lang + "_sen_bad"] = [dataset[i]["wrong_sen"] for i in additional_sample]

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(samples, ensure_ascii=False, indent=4) + "\n")

    with open(additional_results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(additional_samples, ensure_ascii=False, indent=4) + "\n")

    return samples

if __name__ == "__main__":
    load_and_sample_multi_blimp()