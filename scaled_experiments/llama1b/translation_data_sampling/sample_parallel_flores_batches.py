import datasets
from huggingface_hub import HfApi
from tqdm import tqdm
import random
import os
import json
import csv
import pandas as pd
import re

LANGS_setting1 = [
    "eng_Latn",
    "deu_Latn",
    "fra_Latn",
    "zho_Hans",
]

# for later
LANGS_wlt = [
    "eng_Latn",
    "deu_Latn",
    "fra_Latn",
    "zho_Hans",
]

LANG_pairs_bi = [
    ("eng_Latn", "deu_Latn"),
    ("eng_Latn", "zho_Hans"),
    ("zho_Hans", "deu_Latn"),
    ("fra_Latn", "deu_Latn"),
]


LANGS_predictions = [
    "eng_Latn",
    "fra_Latn",
    "zho_Hans",
]


LANGS_parallel_setting2 = [
    "eng_Latn",
    "spa_Latn",
    "zho_Hans",
    "ind_Latn",
]

lang2lab = {
    "eng_Latn": "en",
    "deu_Latn": "de",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "zho_Hans": "zh",
    "jpn_Jpan": "ja",
    "ind_Latn": "id",
}

lab2lang = dict([(lab, lang) for lang, lab in lang2lab.items()])


LANGS = [
    "eng_Latn",
    "deu_Latn",
    "spa_Latn",
    "fra_Latn",
    "ita_Latn",
    "por_Latn",
    "zho_Hans",
    "jpn_Jpan",
    "ind_Latn",
]

wlt_prefixes = {
    "eng_Latn": " Answer:",
    "deu_Latn": " Antwort:",
    "fra_Latn": " Réponse :",
    "zho_Hans": "答案：",
}

FRENCH_CORNER_CASES = {"plat", "livre"}

def get_flores_data_for_langs(langs):
    # load FLORES dataset
    flores_name = "facebook/flores"
    # requires datasets==3.6.0
    flores = datasets.load_dataset(flores_name, "all")

    # filter to only needed langs
    flores_filtered = {}
    for lang in langs:
        flores_filtered[lang] = flores["dev"][f"sentence_{lang}"] + flores["devtest"][f"sentence_{lang}"]
    
    return flores_filtered

def get_wlt_data(langs, restrict_to_words=None, path="results/wlt_data/llm-latent-language/data/langs/"):
    labs = [lang2lab[lang] for lang in langs]
    data = {}
    for lab in labs:
        if os.path.exists(os.path.join(path, lab, "text.csv")):
            data_path = os.path.join(path, lab, "text.csv")
        else:
            data_path = os.path.join(path, lab, "clean.csv")
        # read csv
        with open(data_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # filter rows to only those in restrict_to_words, do via substring matching
        if restrict_to_words is not None:
            filtered = []
            for row in rows:
                if row[2] in restrict_to_words or row[1] in restrict_to_words or row[1] in FRENCH_CORNER_CASES or row[2] in FRENCH_CORNER_CASES:
                    filtered.append(row)
            rows = filtered
        data[lab2lang[lab]] = rows
    
    return data

def convert_wlt_to_flores(wlt_data):
    flores_format_data = {}
    for lang, rows in wlt_data.items():
        flores_format_data[lang] = []
        for row in rows:
            # each row is [question, answer]
            prompt = row[4]
            # rstrip answer part (everything after prefix)
            answer_prefix = wlt_prefixes[lang]
            answer_start = prompt.find(answer_prefix)
            sample = prompt[:answer_start]
            # remove "
            sample = sample.replace('"', '').strip()
            flores_format_data[lang].append(sample)

    return flores_format_data
        
def get_all_pairs(langs):
    lang_pairs = []
    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):
            if i >= j:
                continue
            lang_pairs.append((lang1, lang2))
    return lang_pairs

def get_parallel_batch(num_per_setting, 
                       lang_combs, 
                       output_dir, 
                       used_flores_indices, 
                       flores_data,
                       in_doc=False):
    if num_per_setting > 0:
        num_per_pair = num_per_setting // len(lang_combs)
    else:
        # use all data
        num_per_pair = len(flores_data[lang_combs[0][0]])

    all_batches = {}

    for lang_comb in lang_combs:

        if len(lang_comb) == 2:
            lang1, lang2 = lang_comb
            if not in_doc:
                lang_code = f"parallel_global_{lang2lab[lang1]}_{lang2lab[lang2]}"
            else:
                lang_code = f"parallel_in_doc_{lang2lab[lang1]}_{lang2lab[lang2]}"
        else:
            if not in_doc:
                lang_code = "parallel_global_multi"
            else:
                lang_code = "parallel_in_doc_multi"
        
        batch = []
        id = 0
        for i in range(num_per_pair):
            if num_per_setting > 0:
                # sample ids
                while id in used_flores_indices and len(used_flores_indices) < len(flores_data["eng_Latn"]):
                    id = random.sample(range(len(flores_data["eng_Latn"])), 1)[0]
                used_flores_indices.add(id)
            else:
                # just get all samples
                id = i

            if not in_doc:
                s = []
                for lang in lang_comb:
                    sample = {
                        "text": flores_data[lang][id],
                        "lang_code": lang_code,
                    }
                    s.append(sample)   
            else:
                random_order = random.sample(lang_comb, len(lang_comb))
                concatenated_text = " ".join([flores_data[lang][id] for lang in random_order])
                s = [{
                    "text": concatenated_text,
                    "lang_code": lang_code,
                }]
                
            batch = batch + s
    
        all_batches[lang_code] = batch
    
    # not nice but does the job
    output_path = ""
    if len(lang_comb) == 2:
        if not in_doc:
            lang_code = "parallel_global"
        else:
            lang_code = "parallel_in_doc"
    else:
        if not in_doc:
            lang_code = "parallel_global_multi"
        else:
            lang_code = "parallel_in_doc_multi"
    output_path = os.path.join(output_dir, f"{lang_code}.json")

    with open(output_path, "w") as f:
        json.dump(all_batches, f, ensure_ascii=False, indent=4)
            
    return all_batches, used_flores_indices

def get_wlt_entities_and_prompts(langs, path="/nfs/mainlp01/explainables-project/fkoerner/explainables-eval/data/word_level_translations/"):
    df = None
    batches_pairs = []
    for lang in langs:
        lab = lang2lab[lang]
        to_path = os.path.join(path, f"{lab}_to")
        for lang2 in langs:
            lab2 = lang2lab[lang2]
            from_path = os.path.join(to_path, f"{lab}_{lab2}.csv")
            df_new = pd.read_csv(from_path)
            df_new["from_lang"] = lang
            df_new["to_lang"] = lang2
            if df is None:
                df = df_new
            else:
                # append new rows
                df = pd.concat([df, df_new], ignore_index=True)
    return df

def make_wlt_batches(bidirectional_pairs, wlt_langs, df):
    batches = {}
    # language pairs where we want to study both directions
    for lang1, lang2 in bidirectional_pairs:
        batch = {}
        df_to = df[((df["from_lang"] == lang1) & (df["to_lang"] == lang2))]
        df_from = df[((df["from_lang"] == lang2) & (df["to_lang"] == lang1))]
        
        batch_to = make_wlt_batch_from_df(df_to)
        batch_from = make_wlt_batch_from_df(df_from)

        batches[f"wlt_{lang1}_to_{lang2}"] = batch_to
        batches[f"wlt_{lang2}_to_{lang1}"] = batch_from

    # remaining language pairs where we don't care about translation direction
    remaining_pairs = set(get_all_pairs(wlt_langs)) - set(bidirectional_pairs) - set([(b, a) for a, b in bidirectional_pairs])
    for lang1, lang2 in remaining_pairs:
        batch_key = f"wlt_bi_dir_{lang1}_{lang2}"
        df_bi = df[((df["from_lang"] == lang1) & (df["to_lang"] == lang2)) | ((df["from_lang"] == lang2) & (df["to_lang"] == lang1))]
        batch = make_wlt_batch_from_df(df_bi)
        batches[batch_key] = batch

    # en-all and all-en pairs
    df_to_en = df[((df["to_lang"] == "eng_Latn") & (df["from_lang"] != "eng_Latn"))]
    batches["wlt_all_to_en"] = make_wlt_batch_from_df(df_to_en)

    df_from_en = df[((df["from_lang"] == "eng_Latn") & (df["to_lang"] != "eng_Latn"))]
    batches["wlt_en_to_all"] = make_wlt_batch_from_df(df_from_en)

    # all to all
    df_all = df
    batches["wlt_all_to_all"] = make_wlt_batch_from_df(df_all)

    # copy batches, i.e. en-en, de-de, etc.
    for lang in wlt_langs:
        df_copy = df[((df["from_lang"] == lang) & (df["to_lang"] == lang))]
        batch_key = f"wlt_copy_{lang}"
        batches[batch_key] = make_wlt_batch_from_df(df_copy) 

    # store batches
    with open("results/translation_data_sampling/wlt_batches.json", "w") as f:
        json.dump(batches, f, ensure_ascii=False, indent=4)

    return batches

def make_wlt_batch_from_df(df, 
                           all_valid_targets=False, 
                           prompt_key="prompt", 
                           tgt_key="tgt_word", 
                           valid_key="valid_targets",
                           gold_labels=True,
                           pred_key=None,
                           ):
    batch = []

    for _, row in df.iterrows():
        text = row[prompt_key]

        if gold_labels:
            if pred_key is not None:
                # try chosing the one predicted by the model if it's in the valid targets, otherwise just use the gold target
                valid_targets = [row[pred_key]] if row[pred_key] in row[valid_key].split("|") else [row[tgt_key]]
            elif not all_valid_targets:
                valid_targets = [row[tgt_key]]
            else:
                valid_targets = row[valid_key].split("|")
            for valid_target in valid_targets:
                sample = {
                    "text": text + "[TARGET]" + valid_target + "[/TARGET]\"",
                    "prediction_type": "targets",
                }
                batch.append(sample)
        else:
            if pred_key is None:
                raise ValueError("pred_key must be provided if gold_labels is False")
            pred_target = row[pred_key]
            sample = {
                "text": text + "[TARGET]" + pred_target + "[/TARGET]\"",
                "prediction_type": "predictions",
            }
            batch.append(sample)

    return batch




def main():

    ## get parallel data from FLORES
    flores_data = get_flores_data_for_langs(LANGS)
    
    used_flores_indices = set()
    output_dir = "results/translation_data_sampling/parallel_flores_batches/"
    os.makedirs(output_dir, exist_ok=True)

    # evenly spread flores samples across the four scenarios
    num_scenarios = 3
    num_per_setting_pair = len(flores_data[LANGS_parallel_setting2[0]]) // num_scenarios
    num_per_setting_multi = num_per_setting_pair // 2

    # we want to have x2 data for the pair-parallel scenarios to have a fair distribution of data
    # (multi samples get 4 translations, the other's only 2)

    # create parallel data batches for all our language pairs, do not use same sentence twice
    lang_pairs = get_all_pairs(LANGS_setting1)

    # parallel scenario 1: batches of pairs of languages from setting 1, each in a separate document
    batch, used_flores_indices = get_parallel_batch(num_per_setting_pair, lang_pairs, output_dir, used_flores_indices, flores_data, in_doc=False)
        
    # parallel scenario 2: batches of pairs of languages from setting 1, translated seqs in same doc, concatenated
    batch, used_flores_indices = get_parallel_batch(num_per_setting_pair, lang_pairs, output_dir, used_flores_indices, flores_data, in_doc=True)

    # parallel scenario 3: batches of all languages, each in a separate document
    batch, used_flores_indices = get_parallel_batch(num_per_setting_multi, [LANGS_setting1], output_dir, used_flores_indices, flores_data, in_doc=False)

    # parallel scenario 4: batches of all languages, translated seqs in same doc, concatenated
    batch, used_flores_indices = get_parallel_batch(num_per_setting_multi, [LANGS_setting1], output_dir, used_flores_indices, flores_data, in_doc=True)
    
    ## get parallel data from WLT
    df_wlt = get_wlt_entities_and_prompts(langs=LANGS_setting1)
    # restrict to only the words we use
    words = set(df_wlt["tgt_word"].unique().tolist() + df_wlt["src_word"].unique().tolist())

    ## get test batches from our data
    wlt_batches = make_wlt_batches(LANG_pairs_bi, LANGS_wlt, df_wlt)

    # load correct clozes
    # wlt_data = get_wlt_data(LANGS_wlt, restrict_to_words=words)
    # wlt_data = convert_wlt_to_flores(wlt_data)

    # for l, rows in wlt_data.items():
    #     print(f"WLT data for {l}: {len(rows)} samples.")

    # output_dir = "results/translation_data_sampling/parallel_wlt_batches/"
    # os.makedirs(output_dir, exist_ok=True)
    
    # ## ignore for now
    # # parallel scenario: batches of pairs of languages from WLT, each in a separate document
    # batch, _ = get_parallel_batch(-1, get_all_pairs(LANGS_wlt), output_dir, set(), wlt_data, in_doc=False)

    # # parallel scenario: batches of pairs of languages from WLT, translated seqs in same doc, concatenated
    # batch, _ = get_parallel_batch(-1, get_all_pairs(LANGS_wlt), output_dir, set(), wlt_data, in_doc=True)

    # # parallel scenario: batches of all WLT languages, each in a separate document
    # batch, _ = get_parallel_batch(-1, [LANGS_wlt], output_dir, set(), wlt_data, in_doc=False)

    # ## this scenario first
    # # parallel scenario: batches of all WLT languages, translated seqs in same doc, concatenated
    # batch, _ = get_parallel_batch(-1, [LANGS_wlt], output_dir, set(), wlt_data, in_doc=True)

    print("WLT Test Batches created.")
    print("Number of samples in each WLT batch:", len(wlt_batches))

    print("All parallel FLORES batches, WLT batches created and saved.")




if __name__ == "__main__":
    main()