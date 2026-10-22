import pandas as pd
import os
from tqdm import tqdm
import json
import re
import plotly.express as px
from scaled_experiments.llama1b.translation_data_sampling.sample_parallel_flores_batches import LANGS_setting1, LANGS, get_wlt_entities_and_prompts, LANG_pairs_bi, lang2lab, lab2lang, make_wlt_batch_from_df

# set seed
import random
random.seed(42)
import numpy as np
np.random.seed(42)

def list_results_dirs(results_dir):
    """List all result directories with epoch and step information."""
    checkpoints = []
    for d in os.listdir(results_dir):
        if not d.startswith("epoch"):
            continue
        epochstr, stepstr = d.split("-")
        checkpoints.append(
            {
                "epoch": int(epochstr[5:]),
                "step": int(stepstr[4:]),
                "path": os.path.join(results_dir, d),
            }
        )
    keys = ["epoch", "step"]
    checkpoints.sort(key=lambda x: tuple(x[k] for k in keys))
    return checkpoints

def load_all_result_files_from(checkpoint_dir):
    """Load all result files from a checkpoint directory."""
    checkpoint_dfs = []
    print(checkpoint_dir)
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith(".json"):
            items = []
            fname = os.path.join(checkpoint_dir, fname)
            with open(fname, "r", encoding="utf-8") as f:
                test_data = json.load(f)
                for item in test_data:
                    for pred in test_data[item]:
                        new_item = {}
                        for k, v in pred.items():
                            new_item[k] = v
                        items.append(new_item)
            checkpoint_dfs.append(pd.DataFrame(items))

    return pd.concat(checkpoint_dfs, ignore_index=True)


def load_all_predictions_into_df(results_dir):
    """Load all predictions from result directories into a single DataFrame."""
    checkpoints = list_results_dirs(results_dir)
    assert len(checkpoints) > 0, f"No checkpoints found in {results_dir}"
    print(checkpoints[0])
    all_dfs = []
    print(f"Found {len(checkpoints)} checkpoints in {results_dir}. Loading predictions...")
    for checkpoint in tqdm(checkpoints):
        checkpoint_df = load_all_result_files_from(checkpoint["path"])
        checkpoint_df["epoch"] = checkpoint["epoch"]
        checkpoint_df["step"] = checkpoint["step"]
        all_dfs.append(checkpoint_df)
    print(f"Loaded {len(all_dfs)} prediction files from {results_dir}")
    return pd.concat(all_dfs, ignore_index=True)

def plot_instruction_following(df):
    task_regex = r'[^\'\"]+\"\n.+'
    df["follows_task"] = df["prediction"].apply(lambda x: bool(re.match(task_regex, x)))
    plot_df = df.groupby(["epoch", "step", "src_lang", "tgt_lang"]).agg(
        total_predictions=pd.NamedAgg(column="follows_task", aggfunc="count"),
        task_following_predictions=pd.NamedAgg(column="follows_task", aggfunc="sum"),
    ).reset_index()
    
    # plot timelines
    fig = px.line(plot_df, x="step", y="task_following_predictions", color="src_lang", line_dash="tgt_lang",
                  title="Number of Task-Following Predictions Over Time")
    
    # store
    fig.write_html("results/task_following_predictions_over_time.html")
    
    plot_df = df.groupby(["epoch", "step", "src_lang"]).agg(
        total_predictions=pd.NamedAgg(column="follows_task", aggfunc="count"),
        task_following_predictions=pd.NamedAgg(column="follows_task", aggfunc="sum"),
    ).reset_index()
    
    # plot timelines
    fig = px.line(plot_df, x="step", y="task_following_predictions", color="src_lang",
                  title="Number of Task-Following Predictions Over Time")
    
    # store
    fig.write_html("results/task_following_predictions_over_time_src.html")
    fig.write_image("results/task_following_predictions_over_time_src.png")
    
    plot_df = df.groupby(["epoch", "step", "tgt_lang"]).agg(
        total_predictions=pd.NamedAgg(column="follows_task", aggfunc="count"),
        task_following_predictions=pd.NamedAgg(column="follows_task", aggfunc="sum"),
    ).reset_index()
    
    # plot timelines
    fig = px.line(plot_df, x="step", y="task_following_predictions", color="tgt_lang",
                  title="Number of Task-Following Predictions Over Time")
    
    # store
    fig.write_html("results/task_following_predictions_over_time_tgt.html")
    fig.write_image("results/task_following_predictions_over_time_tgt.png")


def get_data(run_dir="results/eval-results/results/"):

    if os.path.exists("results/all_predictions.csv"):
        print("Loading predictions from CSV...")
        df = pd.read_csv("results/all_predictions.csv")
    else:
        print("Loading predictions from result directories (slow)...")
        df = load_all_predictions_into_df(run_dir)

    final_step_df = df[df["step"] == df["step"].max()]

    success_df = final_step_df.groupby("en_word").agg(
            total_occurrences=pd.NamedAgg(column="en_word", aggfunc="count"),
            success_count=pd.NamedAgg(column="accuracy", aggfunc="sum"),
        ).reset_index().sort_values(by="success_count", ascending=False)

    lang_trio = [ "en", "de", "zh"]
    trio_df = final_step_df[final_step_df["src_lang"].isin(lang_trio) & final_step_df["tgt_lang"].isin(lang_trio) & (final_step_df["src_lang"] != final_step_df["tgt_lang"])]
    print(trio_df.columns)
    trio_success_df = trio_df.groupby("en_word").agg(
            total_occurrences=pd.NamedAgg(column="en_word", aggfunc="count"),
            success_count=pd.NamedAgg(column="accuracy", aggfunc="sum"),
            tgt_words=pd.NamedAgg(column="target_word", aggfunc=lambda x: x.tolist()),
            src_words=pd.NamedAgg(column="prompt", aggfunc=lambda x: x.tolist()),
            pred_words=pd.NamedAgg(column="generated_word", aggfunc=lambda x: x.tolist())
        ).reset_index().sort_values(by="success_count", ascending=False)
    # sort trio_df by success count and then lang (use a merge)
    trio_df["success_count"] = trio_df["en_word"].map(trio_success_df.set_index("en_word")["success_count"])
    trio_df = trio_df.sort_values(by=["success_count", "en_word", "src_lang", "tgt_lang"], ascending=[False, True, True, True])


    langx4 = ["en", "de", "zh", "fr"]
    x4_success_df = final_step_df[final_step_df["src_lang"].isin(langx4) & final_step_df["tgt_lang"].isin(langx4)].groupby("en_word").agg(
            total_occurrences=pd.NamedAgg(column="en_word", aggfunc="count"),
            success_count=pd.NamedAgg(column="accuracy", aggfunc="sum"),
        ).reset_index().sort_values(by="success_count", ascending=False)

    ### store as a checkpoint
    if not os.path.exists("results/all_predictions.csv"):
        df.to_csv("results/all_predictions.csv", index=False)

    ### load and join with prompts
    df_wlt = get_wlt_entities_and_prompts(langs=LANGS)
    # restrict to only the words we use
    words = set(df_wlt["tgt_word"].unique().tolist() + df_wlt["src_word"].unique().tolist())

    df["join_key"] = df["en_word"] + "_" + df["src_lang"] + "_" + df["tgt_lang"]
    df_wlt["join_key"] = df_wlt["en_word"] + "_" + df_wlt["from_lang"].apply(lambda x: lang2lab[x]) + "_" + df_wlt["to_lang"].apply(lambda x: lang2lab[x])

    df_j = df.merge(df_wlt, on="join_key", how="left")
    assert df_j["prompt_y"].isna().sum() == 0, "Some predictions did not have a corresponding prompt in the WLT data."

    for line in trio_success_df.itertuples():
        print(f"Word: {line.en_word}, Success Count: {line.success_count}, Total Occurrences: {line.total_occurrences}")

    return df_j, trio_success_df, x4_success_df

def print_preds_for_words(df_j, words, langs=["en", "de", "zh"]):
    df_word_preds = df_j[df_j["en_word_x"].isin(words) & (df_j["step"] == df_j["step"].max()) & (df_j["src_lang"].isin(langs)) & (df_j["tgt_lang"].isin(langs)) & (df_j["src_lang"] != df_j["tgt_lang"])]
    df_word_preds = df_word_preds.sort_values(by=["en_word_x", "src_lang", "tgt_lang"], ascending=[True, True, True])
    for line in df_word_preds.itertuples():
        print(f"Word: {line.en_word_x}, Src Lang: {line.src_lang}, Tgt Lang: {line.tgt_lang}, Prediction: {line.generated_word}")
        print(f"Success: {line.accuracy}, correct answer: {line.tgt_word}, valid answers: {line.valid_targets}")
        print(f"predicted: {line.generated_word}") 
        print("-" * 50)

def load_and_test_clozes():
    # for exploration and debugging
    # load clozes to check which words have changed
    clozes_path = "results/wlt_data/clozes_fixed_ws_260226.csv"
    df_clozes = pd.read_csv(clozes_path)

    # find words in df_j that are missing from df_clozes and vice versa
    df_j_words = set(df_j["en_word_x"].unique())
    clozes_words = set(df_clozes["eng"].unique())
    missing_from_clozes = df_j_words - clozes_words
    missing_from_j = clozes_words - df_j_words
    print(f"Words in df_j but missing from clozes: {missing_from_clozes}")
    print(f"Words in clozes but missing from df_j: {missing_from_j}")  # the ones for the extra control batch

    single_path = "results/wlt_data/single_trio_pred_clozes_260226.csv"
    clozes_single = pd.read_csv(single_path)

def get_batch_from_df(df):
    pass


### pick single preds, good, imbalanced, bad
df_j, trio_success_df, x4_success_df = get_data()

plot_instruction_following(df_j)

for line in trio_success_df.itertuples():
    print(f"Word: {line.en_word}, Success Count: {line.success_count}, Total Occurrences: {line.total_occurrences}")
    print(f"Src Words: {line.src_words}")
    print(f"Tgt Words: {line.tgt_words}")
    print(f"Prd Words: {line.pred_words}")

### pick three words (one for each lang pair) that are correct in both directions, imbalanced, and incorrect for all
single_trio_preds = [
    "potent", # good
    "assured", # imbalanced
    "artificial", # bad
]

print_preds_for_words(df_j, single_trio_preds)

lang_trio = [ "en", "de", "zh"]

single_samples = df_j[df_j["en_word_x"].isin(single_trio_preds) & (df_j["step"] == df_j["step"].max()) & (df_j["src_lang"].isin(lang_trio)) & (df_j["tgt_lang"].isin(lang_trio)) & (df_j["src_lang"] != df_j["tgt_lang"])].sort_values(by=["en_word_x", "src_lang", "tgt_lang"], ascending=[True, True, True])
copy_single_batch = df_j[df_j["en_word_x"].isin(single_trio_preds) & (df_j["step"] == df_j["step"].max()) & (df_j["src_lang"].isin(lang_trio)) & (df_j["src_lang"] == df_j["tgt_lang"])].sort_values(by=["en_word_x", "src_lang"], ascending=[True, True])

# TODO: put prompts into test batches
# TODO: get clozes: 3 single occurence for en zh de, 1 translated to all three langs, i.e. 4 per word
# TODO: get all corresponding copy preds and put into a single batch
### total: 18 test + 1 copy

### 25 samples each: Two batches (all langs) of successful and unsuccessful translation -> /w single occurence
### 25 samples each: Two batches (all langs) of successful and unsuccessful translation -> /w fully parallel
remaining_words = set(trio_success_df["en_word"].unique().tolist())
available_words_good = set(trio_success_df[trio_success_df["success_count"] > 1]["en_word"].tolist())

# sample correct predictions from available words, i.e. triplets of en_word, src_lang, tgt_lang where the prediction was correct
df_j_good = df_j[df_j["en_word_x"].isin(available_words_good) & (df_j["step"] == df_j["step"].max()) & (df_j["src_lang"].isin(lang_trio)) & (df_j["tgt_lang"].isin(lang_trio)) & (df_j["src_lang"] != df_j["tgt_lang"]) & (df_j["accuracy"] == 1)]
good_sample = df_j_good.groupby("en_word_x").apply(lambda x: x.sample(1)).reset_index(drop=True)
good_sample1 = good_sample.iloc[:25]
good_sample2 = good_sample.iloc[25:50]
copy_batch_good = df_j[df_j["en_word_x"].isin(set(good_sample.iloc[0:50]["en_word_x"].unique())) & (df_j["step"] == df_j["step"].max()) & (df_j["src_lang"].isin(lang_trio)) & (df_j["src_lang"] == df_j["tgt_lang"])].sort_values(by=["en_word_x", "src_lang"], ascending=[True, True])

remaining_words = remaining_words - set(good_sample1["en_word_x"].unique().tolist()) - set(good_sample2["en_word_x"].unique().tolist())

bad_sample = df_j[df_j["en_word_x"].isin(remaining_words) & (df_j["step"] == df_j["step"].max()) & (df_j["src_lang"].isin(lang_trio)) & (df_j["tgt_lang"].isin(lang_trio)) & (df_j["src_lang"] != df_j["tgt_lang"]) & (df_j["accuracy"] == 0)].groupby("en_word_x").apply(lambda x: x.sample(1)).reset_index(drop=True)
bad_sample1 = bad_sample.iloc[:25]
bad_sample2 = bad_sample.iloc[25:50]
### for all: copy task
copy_batch_bad = df_j[df_j["en_word_x"].isin(set(bad_sample.iloc[0:50]["en_word_x"].unique())) & (df_j["step"] == df_j["step"].max()) & (df_j["src_lang"].isin(lang_trio)) & (df_j["src_lang"] == df_j["tgt_lang"])].sort_values(by=["en_word_x", "src_lang"], ascending=[True, True])

### train: one batch of single and one batch of parallel, but from the context clozes
### train: flores batches (part of them)

load_and_test_clozes()

batches_test = {
    "good_sample1": make_wlt_batch_from_df(good_sample1, prompt_key="prompt_y", tgt_key="tgt_word", pred_key="generated_word"),
    "good_sample2": make_wlt_batch_from_df(good_sample2, prompt_key="prompt_y", tgt_key="tgt_word", pred_key="generated_word"),
    "copy_batch_good": make_wlt_batch_from_df(copy_batch_good, prompt_key="prompt_y", tgt_key="tgt_word", pred_key="generated_word"),
    "bad_sample1": make_wlt_batch_from_df(bad_sample1, prompt_key="prompt_y", tgt_key="tgt_word", pred_key="generated_word"),
    "bad_sample2": make_wlt_batch_from_df(bad_sample2, prompt_key="prompt_y", tgt_key="tgt_word", pred_key="generated_word"),
    "copy_batch_bad": make_wlt_batch_from_df(copy_batch_bad, prompt_key="prompt_y", tgt_key="tgt_word", pred_key="generated_word"),
}

# one for each line of that df
single_test_batches = {
    f"{line.en_word_x}_{line.src_lang}_{line.tgt_lang}": make_wlt_batch_from_df(single_samples[single_samples["en_word_x"] == line.en_word_x][single_samples["src_lang"] == line.src_lang][single_samples["tgt_lang"] == line.tgt_lang], prompt_key="prompt_y", tgt_key="tgt_word", pred_key="generated_word")
    for i, line in enumerate(single_samples.itertuples()) 
}
single_test_batches["copy"] = make_wlt_batch_from_df(copy_single_batch, prompt_key="prompt_y", tgt_key="tgt_word", pred_key="generated_word")

all_test_batches = {**batches_test, **single_test_batches}
print(all_test_batches.keys())
print(f"Total test batches: {len(all_test_batches)}")

with open("results/translation_data_sampling/wlt_occurence_batches.json", "w", encoding="utf-8") as f:
    json.dump(all_test_batches, f, ensure_ascii=False, indent=4)



### now to the train examples
translated_clozes_path = "results/wlt_data/clozes_translated.csv"
df_clozes = pd.read_csv(translated_clozes_path)

# for each row, get translation from highest scoring model in terms of cos similarity of backtranslation
langs = ["deu_Latn", "zho_Hans", "jpn_Jpan", "por_Latn", "ita_Latn", "fra_Latn", "spa_Latn", "ind_Latn",]
models = ["opusmt", "gemma", "nllb"]
sim_field = "_back_sim_"
cloze_field = "_cloze_"


for lang in langs:
    print(f"Processing language: {lang}")
    def get_best_cloze(row, lang):
        best_sim = -1
        best_cloze = None
        for model in models:
            sim = row[f"{lang}{sim_field}{model}"]
            if sim > best_sim:
                best_sim = sim
                best_cloze = row[f"{lang}{cloze_field}{model}"] 
        return best_cloze

    df_clozes[f"best_{lang}"] = df_clozes.apply(lambda row: get_best_cloze(row, lang), axis=1)

df_clozes["best_eng_Latn"] = df_clozes["eng_cloze"]


def make_train_batch_from_clozes(df, langs):
    batch = []
    for row in df.itertuples():
        for lang in langs:
            cloze = row[f"best_{lang}"]
            batch.append({
                "text": cloze,
                "lang_code": lang,
            })
    return batch

### get train batches as specified above
single_cloze_df = df_clozes[df_clozes["eng_word"].isin(single_trio_preds)]

### drop the ones where freq is not NaN (duplicates from larger dataset)
single_cloze_df = single_cloze_df[single_cloze_df["freq"].isna()]

langs_studied = ["deu_Latn", "zho_Hans", "eng_Latn"]

# get one row from each word for parallel batch
df_parallel = single_cloze_df.groupby("eng_word").apply(lambda x: x.sample(1)).reset_index(drop=True)
# remove from
single_cloze_df = single_cloze_df[~single_cloze_df["best_eng_Latn"].isin(df_parallel["best_eng_Latn"].tolist())]

def get_single_occ_for_langs(single_cloze_df, langs_studied, sep_by_lang=True, affix=""):
    single_occ_batch = {}
    for word in single_cloze_df["eng_word"].unique():
        print(f"Word: {word}")
        for row, lang in zip(single_cloze_df[single_cloze_df["eng_word"] == word].iterrows(), langs_studied):
            row = row[1]
            cloze = row[f"best_{lang}"]
            print(f"Lang: {lang}, Cloze: {cloze}")

            if sep_by_lang:
                key = f"{affix}clozes_single_occ_{lang}"
            else:
                key = f"{affix}clozes_single_occ"

            if key not in single_occ_batch:
                single_occ_batch[key] = []

            # remove any " from the cloze
            clean_cloze = cloze.replace('"', '').replace("'", "")
            single_occ_batch[key].append({
                "text": clean_cloze,
                "lang_code": lang,
            })
    return single_occ_batch

single_occ_batch = get_single_occ_for_langs(single_cloze_df, langs_studied, sep_by_lang=True, affix="individual_")

def get_parallel_for_langs(df_parallel, langs_studied, affix=""):
    parallel_cloze_batch = {}
    # and last cloze goes into parallel batch for each word
    for row in df_parallel.itertuples():
        word = row.eng_word
        print(f"Word: {word}")
        text = ""
        for lang in langs_studied:
            cloze = getattr(row, f"best_{lang}")
            print(f"Lang: {lang}, Cloze: {cloze}")
            key = f"{affix}clozes_parallel"

            if key not in parallel_cloze_batch:
                parallel_cloze_batch[key] = []

            # remove any " from the cloze
            clean_cloze = cloze.replace('"', '').replace("'", "")
            if text == "":
                text = clean_cloze
            else:
                text += " " + clean_cloze
        parallel_cloze_batch[key].append({
            "text": text,
            "lang_code": lang,
        })
    return parallel_cloze_batch

parallel_cloze_batch = get_parallel_for_langs(df_parallel, langs_studied, affix="individual_")

# and for the larger test batches
# remove the samples from singke scenario (but keep the original one)
df_clozes = df_clozes[~df_clozes["freq"].isna()]
df_good_sample1_single = df_clozes[df_clozes["eng_word"].isin(good_sample1["en_word_x"].unique())].join(good_sample1.set_index("en_word_x"), on="eng_word", how="inner", rsuffix="_pred")

def get_single_src_and_tgt_batches(df_good_sample1_single, lab2lang, src=True, affix="batched_"):
    batches = {}
    for row in df_good_sample1_single.itertuples():
        word = row.eng_word
        if src:
            lang = row.src_lang
            key = f"{affix}single_src"
        else:
            lang = row.tgt_lang
            key = f"{affix}single_tgt"
        cloze = getattr(row, f"best_{lab2lang[lang]}")
        print(f"Word: {word}, Lang: {lang}, Cloze: {cloze}")

        if key not in batches:
            batches[key] = []

        clean_cloze = cloze.replace('"', '').replace("'", "")
        batches[key].append({
            "text": clean_cloze,
            "lang_code": lang,
        })
    return batches

good_sample1_single_src_batches = get_single_src_and_tgt_batches(df_good_sample1_single, lab2lang, src=True, affix="batched_good_sample1_")
good_sample1_single_tgt_batches = get_single_src_and_tgt_batches(df_good_sample1_single, lab2lang, src=False, affix="batched_good_sample1_")

df_bad_sample1_single = df_clozes[df_clozes["eng_word"].isin(bad_sample1["en_word_x"].unique())].join(bad_sample1.set_index("en_word_x"), on="eng_word", how="inner", rsuffix="_pred")
bad_sample1_single_src_batches = get_single_src_and_tgt_batches(df_bad_sample1_single, lab2lang, src=True, affix="batched_bad_sample1_")
bad_sample1_single_tgt_batches = get_single_src_and_tgt_batches(df_bad_sample1_single, lab2lang, src=False, affix="batched_bad_sample1_")


def get_parallel_batches(df_sample, lab2lang, affix="batched_"):
    batches = {}
    for row in df_sample.itertuples():
        word = row.eng_word
        print(f"Word: {word}")
        text = ""
        for lang in set([row.src_lang, row.tgt_lang]):
            cloze = getattr(row, f"best_{lab2lang[lang]}")
            print(f"Lang: {lang}, Cloze: {cloze}")
            key = f"{affix}clozes_parallel"

            if key not in batches:
                batches[key] = []

            clean_cloze = cloze.replace('"', '').replace("'", "")
            if text == "":
                text = clean_cloze
            else:
                text += " " + clean_cloze
        batches[key].append({
            "text": text,
            "lang_code": lang,
        })
    return batches

df_good_sample2_parallel = df_clozes[df_clozes["eng_word"].isin(good_sample2["en_word_x"].unique())].join(good_sample2.set_index("en_word_x"), on="eng_word", how="inner", rsuffix="_pred")
df_bad_sample2_parallel = df_clozes[df_clozes["eng_word"].isin(bad_sample2["en_word_x"].unique())].join(bad_sample2.set_index("en_word_x"), on="eng_word", how="inner", rsuffix="_pred")

good_sample2_parallel_batches = get_parallel_batches(df_good_sample2_parallel, lab2lang, affix="batched_good_sample2_")
bad_sample2_parallel_batches = get_parallel_batches(df_bad_sample2_parallel, lab2lang, affix="batched_bad_sample2_")

# take remaining clozes for one control_single and one control_parallel batch
df_remaining_clozes = df_clozes[~df_clozes["eng_word"].isin(good_sample1["en_word_x"].unique()) & ~df_clozes["eng_word"].isin(good_sample2["en_word_x"].unique()) & ~df_clozes["eng_word"].isin(bad_sample1["en_word_x"].unique()) & ~df_clozes["eng_word"].isin(bad_sample2["en_word_x"].unique())]
# 15 for single and 10 for parallel
df_control_single = df_remaining_clozes.iloc[:15]
# sample random src and tgt langs
df_control_single["src_lang"] = df_control_single.apply(lambda row: random.choice(lang_trio), axis=1)
# sample tgt random but different from src
def sample_tgt(row):
    tgt_langs = [lang for lang in lang_trio if lang != row.src_lang]
    return random.choice(tgt_langs)
df_control_single["tgt_lang"] = df_control_single.apply(sample_tgt, axis=1)
control_single_batch = get_single_src_and_tgt_batches(df_control_single, lab2lang, src=True, affix="batched_control_single_")

df_control_para = df_remaining_clozes.iloc[15:]
df_control_para["src_lang"] = df_control_para.apply(lambda row: random.choice(lang_trio), axis=1)
df_control_para["tgt_lang"] = df_control_para.apply(sample_tgt, axis=1)
control_parallel_batch = get_parallel_batches(df_control_para, lab2lang, affix="batched_control_parallel_")

train_batches = {}
train_batches.update(single_occ_batch)
train_batches.update(parallel_cloze_batch)
# train_batches.update(good_sample1_single_src_batches)  # don't create parallel data. We are interested in occ of target
train_batches.update(good_sample1_single_tgt_batches)
# train_batches.update(bad_sample1_single_src_batches)
train_batches.update(bad_sample1_single_tgt_batches)
train_batches.update(good_sample2_parallel_batches)
train_batches.update(bad_sample2_parallel_batches)
train_batches.update(control_single_batch)
train_batches.update(control_parallel_batch)

print(train_batches.keys())
print(f"Total train batches: {len(train_batches)}")

with open("results/translation_data_sampling/wlt_occurence_train_batches.json", "w", encoding="utf-8") as f:
    json.dump(train_batches, f, ensure_ascii=False, indent=4)