import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from copy import deepcopy

import custom_gemma_prompt

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # for debugging

torch.set_float32_matmul_precision('high')

run_all = False

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

# IDP 639-1
LANG2GEMMAID = {
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

# load English clozes
def get_cloze_data(data_path="results/wlt_data/clozes_fixed_ws_260226.csv"):
    df = pd.read_csv(data_path)

    # check, whether all clozes contain target word in """
    df["contains_target"] = df.apply(lambda row: f'"{row["eng"]}"' in row["cloze"], axis=1)
    assert df["contains_target"].all(), "Not all clozes contain the target word in quotes."

    # rename cloze column
    df = df.rename(columns={"cloze": "eng_cloze", "eng": "eng_word"})

    # preprocess/clean clozes from trailing white space etc.
    df["eng_cloze"] = df["eng_cloze"].str.replace("\n", " ")
    df["eng_cloze"] = df["eng_cloze"].str.strip()

    return df

def get_model_and_tokenizer_nllb():
    # get translation model
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
    return model, tokenizer

def get_translate_gemma_model_and_tokenizer():
    model_id = "google/translategemma-27B-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
    model.eval()
    return model, processor

def get_opus_mt_model_and_tokenizer(source_lang, target_lang):
    source_lang = LANG2GEMMAID[source_lang]
    target_lang = LANG2GEMMAID[target_lang]

    if source_lang == "en" and target_lang == "pt":
        # no standard model
        model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
    elif source_lang == "ja" and target_lang == "en":
        model_name = "Helsinki-NLP/opus-mt-jap-en"
    elif source_lang == "en" and target_lang == "ja":
        model_name = "Helsinki-NLP/opus-mt-en-jap"
    elif source_lang == "pt":
        source_lang = "roa"
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    else:
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.to("cuda")
    return model, tokenizer

def translate_clozes_gemma(clozes, model, processor, source_lang="eng_Latn", target_lang="fra_Latn", batch_size=5):
    source_lang = LANG2GEMMAID[source_lang]
    target_lang = LANG2GEMMAID[target_lang]
    translations = []

    all_messages = [[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": cloze
                    }
                ],
            }
        ] for cloze in clozes]

    translations = []

    for i in tqdm(range(0, len(all_messages), batch_size)):
        messages = all_messages[i:min(i + batch_size, len(all_messages))]
        print("Processing batch")
        inputs = processor.apply_chat_template(
                        messages, 
                        tokenize=True,
                        chat_template=custom_gemma_prompt.prompt,
                        add_generation_prompt=True, 
                        padding=True, 
                        padding_side="left", 
                        return_dict=True, 
                        return_tensors="pt"
                    ).to(model.device, dtype=torch.bfloat16)
        input_lens = inputs['input_ids'].ne(processor.tokenizer.pad_token_id).sum(dim=1)

        torch.cuda.empty_cache()

        print("Generating translations")
        with torch.inference_mode():
            generations = model.generate(**inputs, do_sample=False, max_new_tokens=128)
        
        generations = [gen[input_len:] for gen, input_len in zip(generations, input_lens)]

        print("Decoding translations")
        batch_translations = processor.batch_decode(generations, skip_special_tokens=True)
        translations.extend(batch_translations)

    # normalize quotation marks to ""
    translations = [normalize_quotation_marks(t) for t in translations]
    
    # remove prompt stuf that is being kept for some reason
    translations = [t.strip("\n") for t in translations]
    translations = [t.split("\n")[-1] if "\n" in t else t for t in translations]

    print(translations)

    return translations

def normalize_quotation_marks(text):
    # replace different types of quotation marks with standard "
    quotation_marks = ['“', '”', '«', '»', '„', '‟', '‹', '›', '❝', '❞', '❮', '❯', '〝', '〞', '＂', '˶', '˷', '「', '」', '『', '』']
    for mark in quotation_marks:
        text = text.replace(mark, '"')
    return text

def translate_clozes_opus_mt(clozes, model=None, tokenizer=None, source_lang="eng_Latn", target_lang="fra_Latn"):

    # load correct model
    model, tokenizer = get_opus_mt_model_and_tokenizer(source_lang, target_lang)

    # prepare input text
    inputs = tokenizer(clozes, return_tensors="pt", padding=True, truncation=True, padding_side="left")
    inputs.to(model.device)

    # generate translation
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_length=128)
    
    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translated_text = [normalize_quotation_marks(t) for t in translated_text]
    return translated_text


def translate_clozes_nllb(clozes, model, tokenizer, source_lang="eng_Latn", target_lang="fra_Latn"):

    if tokenizer.src_lang != source_lang:
        tokenizer.src_lang = source_lang

    # prepare input text with language tokens
    inputs = tokenizer(clozes, return_tensors="pt", padding=True, truncation=True, padding_side="left")
    inputs.to(model.device)

    # generate translation
    with torch.inference_mode():
        outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang), max_length=128)

    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    translated_text = [normalize_quotation_marks(t) for t in translated_text]

    return translated_text

def translate_to_all_langs(df, model, tokenizer, 
                           translate_func=translate_clozes_nllb,
                           #source_lang="eng_Latn",
                           #target_lang="fra_Latn",
                           translation_model_name="nllb"):

    # translate clozes
    for lang in LANGS:
        if lang == "eng_Latn":
            continue
        cname = f"{lang}_cloze_{translation_model_name}"
        print(f"Translating clozes to {lang}...")
        df[cname] = translate_func(df["eng_cloze"].tolist(), model, tokenizer,
                                             source_lang="eng_Latn", target_lang=lang)

        print("Extracting target words from translated clozes using \" matching...")
        df[f"{lang}_word_{translation_model_name}"] = df.apply(
            lambda row: row[cname].split(f'"')[1] if f'"' in row[cname] else "", axis=1)
        print(f"Failed for {df[df[f'{lang}_word_{translation_model_name}'] == ''].shape[0]} clozes.")

    return df


def backtranslate(df, model, tokenizer,
                translate_func=translate_clozes_nllb,
                translation_model_name="nllb"):
    
    # backtranslate and add emb similarity scores
    for lang in LANGS:
        if lang == "eng_Latn":
            continue
        cname_back = f"{lang}_back_cloze_{translation_model_name}"
        cname = f"{lang}_cloze_{translation_model_name}"
        print(f"Backtranslating clozes from {lang} to English...")
        df[cname_back] = translate_func(df[cname].tolist(), model, tokenizer, source_lang=lang, target_lang="eng_Latn")
    
    return df

def score_backtranslations(df,
                            emb_model_name="Qwen/Qwen3-Embedding-4B", model_names=["gemma", "nllb"]):
    emb_model = SentenceTransformer(emb_model_name)
    emb_model.eval()
    emb_model.to("cuda")

    for lang in LANGS:
        if lang == "eng_Latn":
            continue
        for translation_model_name in model_names:
            cname_back = f"{lang}_back_cloze_{translation_model_name}"
            # cname = f"{lang}_cloze_{translation_model_name}"
            # compute embedding similarity
            emb_orig = emb_model.encode(df["eng_cloze"].tolist(), convert_to_tensor=True, device="cuda")
            emb_back = emb_model.encode(df[cname_back].tolist(), convert_to_tensor=True, device="cuda")

            cos_sim_scores = torch.nn.functional.cosine_similarity(emb_orig, emb_back).cpu().numpy()
            # model_sim_scores = emb_model.similarity(emb_orig, emb_back).cpu().numpy()

            df[f"{lang}_back_sim_{translation_model_name}"] = cos_sim_scores

            print(f"Avg. cosine similarity for {lang}: {cos_sim_scores.mean()}")
            print(f"Min cosine similarity for {lang}: {cos_sim_scores.min()}")
    return df

def translation_pipeline(df, suffix="", run_all=False):
    ### get gemma and translate
    model, processor = None, None
    if not os.path.exists(f"results/wlt_data/clozes_translated_gemma.csv") and not os.path.exists(f"results/wlt_data/clozes_backtranslated_gemma.csv"):
        model, processor = get_translate_gemma_model_and_tokenizer()

    if not os.path.exists("results/wlt_data/clozes_translated_gemma.csv") and not run_all:
        df = translate_to_all_langs(df, model, processor,
                                    translate_func=translate_clozes_gemma,
                                    translation_model_name="gemma")
        df.to_csv("results/wlt_data/clozes_translated_gemma.csv", index=False)
    else:
        df = pd.read_csv("results/wlt_data/clozes_translated_gemma.csv")

    torch.cuda.empty_cache()

    if not os.path.exists("results/wlt_data/clozes_backtranslated_gemma.csv") and not run_all:
        df = backtranslate(df, model, processor,
                            translate_func=translate_clozes_gemma,
                            translation_model_name="gemma")
        df.to_csv("results/wlt_data/clozes_backtranslated_gemma.csv", index=False)
    else:
        df = pd.read_csv("results/wlt_data/clozes_backtranslated_gemma.csv")

    # delete model to free memory
    del model
    del processor
    torch.cuda.empty_cache()

    if not os.path.exists("results/wlt_data/clozes_backtranslated_gemma_scored.csv") and not run_all:
        df = score_backtranslations(df,
                                    emb_model_name="Qwen/Qwen3-Embedding-4B",
                                    model_names=["gemma"])
        df.to_csv("results/wlt_data/clozes_backtranslated_gemma_scored.csv", index=False)
    else:
        df = pd.read_csv("results/wlt_data/clozes_backtranslated_gemma_scored.csv")

    torch.cuda.empty_cache()


    ### get nllb translations and check
    model, tokenizer = None, None
    if not os.path.exists("results/wlt_data/clozes_translated_nllb.csv") and not os.path.exists("results/wlt_data/clozes_backtranslated_nllb.csv"):
        model, tokenizer = get_model_and_tokenizer_nllb()
        model.eval()
        model.to("cuda")

    if not os.path.exists("results/wlt_data/clozes_translated_nllb.csv") and not run_all:
        df = translate_to_all_langs(df, model, tokenizer,
                                    translate_func=translate_clozes_nllb,
                                    translation_model_name="nllb")
        df.to_csv("results/wlt_data/clozes_translated_nllb.csv", index=False)
    else:
        df = pd.read_csv("results/wlt_data/clozes_translated_nllb.csv")

    torch.cuda.empty_cache()

    if not os.path.exists("results/wlt_data/clozes_backtranslated_nllb.csv") and not run_all:
        df = backtranslate(df, model, tokenizer,
                            translate_func=translate_clozes_nllb,
                            translation_model_name="nllb")
        df.to_csv("results/wlt_data/clozes_backtranslated_nllb.csv", index=False)
    else:
        df = pd.read_csv("results/wlt_data/clozes_backtranslated_nllb.csv")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    if not os.path.exists("results/wlt_data/clozes_backtranslated_nllb_scored.csv") and not run_all:
        df = score_backtranslations(df,
                                    emb_model_name="Qwen/Qwen3-Embedding-4B",
                                    model_names=["nllb"])
        df.to_csv("results/wlt_data/clozes_backtranslated_nllb_scored.csv", index=False)
    else:
        df = pd.read_csv("results/wlt_data/clozes_backtranslated_nllb_scored.csv")

    torch.cuda.empty_cache()


    ### last fallback: OPUS-MT
    if not os.path.exists("results/wlt_data/clozes_translated_opusmt.csv") and not run_all:
        df = translate_to_all_langs(df, model=None, tokenizer=None,
                                    translate_func=translate_clozes_opus_mt,
                                    translation_model_name="opusmt")
        df.to_csv("results/wlt_data/clozes_translated_opusmt.csv", index=False)
    else:
        df = pd.read_csv("results/wlt_data/clozes_translated_opusmt.csv")

    torch.cuda.empty_cache()

    if not os.path.exists("results/wlt_data/clozes_backtranslated_opusmt.csv") and not run_all:
        df = backtranslate(df, model=None, tokenizer=None,
                            translate_func=translate_clozes_opus_mt,
                            translation_model_name="opusmt")
        df.to_csv("results/wlt_data/clozes_backtranslated_opusmt.csv", index=False)
    else:
        df = pd.read_csv("results/wlt_data/clozes_backtranslated_opusmt.csv")

    torch.cuda.empty_cache()

    if not os.path.exists("results/wlt_data/clozes_backtranslated_opusmt_scored.csv") and not run_all:
        df = score_backtranslations(df,
                                    emb_model_name="Qwen/Qwen3-Embedding-4B",
                                    model_names=["opusmt"])
        df.to_csv("results/wlt_data/clozes_backtranslated_opusmt_scored.csv", index=False)
    else:
        df = pd.read_csv("results/wlt_data/clozes_backtranslated_opusmt_scored.csv")

    torch.cuda.empty_cache()

    df.to_csv("results/wlt_data/clozes_translated.csv", index=False)

    return df

# get data
df_data = get_cloze_data()
df_triplet = get_cloze_data("results/wlt_data/single_trio_pred_clozes.csv")

# concat
df_to_translate = pd.concat([df_data, df_triplet], ignore_index=True)

df = translation_pipeline(df_to_translate, run_all=run_all)

# print final stats
for lang in LANGS:
    if lang == "eng_Latn":
        continue
    all_failed = None
    for model_name in ["gemma", "nllb", "opusmt"]:
        cname_word = f"{lang}_word_{model_name}"
        cname_back_sim = f"{lang}_back_sim_{model_name}"
        fails = (df[cname_word] == '') | (df[cname_word].isnull()) | (df[cname_word] == 'NaN')
        print(f"Final stats for {lang} using {model_name}:")
        print(f"  - Failed extractions: {fails.sum()} out of {df.shape[0]}")
        print(f"  - Avg. backtranslation similarity: {df[cname_back_sim].mean()}")
        if all_failed is None:
            all_failed = fails
        else:
            # do an or
            all_failed = (all_failed & fails)

    print()
    print(f"Total failed extractions for {lang} using both models: {all_failed.sum()}. Instances are:")
    for i, row in df[all_failed].iterrows():
        print("=============")
        print(row["eng_word"], row["eng_cloze"], row[f"{lang}_cloze_gemma"], row[f"{lang}_cloze_nllb"], row[f"{lang}_cloze_opusmt"], sep="\n")
        print("=============")
    print("\n\n")

# save only the translated words in a separate csv for easier access during evaluation
df_words_only = deepcopy(df)
word_cols = ["eng_word", "eng_cloze"]
for lang in LANGS:
    if lang == "eng_Latn":
        continue
    for model_name in ["gemma", "nllb", "opusmt"]:
        cname_word = f"{lang}_word_{model_name}"
        word_cols.append(cname_word)

df_words_only = df_words_only[word_cols]
df_words_only.to_csv("results/wlt_data/clozes_translated_words_only.csv", index=False)