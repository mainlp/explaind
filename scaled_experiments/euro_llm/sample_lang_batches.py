import datasets
from huggingface_hub import HfApi
from tqdm import tqdm
import random
import os
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

NEEDED_FWHQ = ["rus_Cyrl", "pol_Latn", "nld_Latn", "arb_Arab", "tur_Latn", "swe_Latn", "ces_Latn", "ell_Grek"]

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

# I read these with a ruler from the paper like an savage. Do we have exact numbers anywhere?
EUROLLM_FRACS = {
    "eng_Latn": 0.5,
    "deu_Latn": 0.05,
    "spa_Latn": 0.05,
    "fra_Latn": 0.05,
    "ita_Latn": 0.05,
    "por_Latn": 0.03,
    "cmn_Hani": 0.03,
    "rus_Cyrl": 0.02,
    "pol_Latn": 0.02,
    "nld_Latn": 0.02,
    "jpn_Jpan": 0.02,
    "arb_Arab": 0.01,
    "tur_Latn": 0.01,
    "swe_Latn": 0.01,
    "kor_Hang": 0.01,
    "ces_Latn": 0.01,
    "hin_Deva": 0.005,
    "ell_Grek": 0.005,
    "hun_Latn": 0.005,
    "ron_Latn": 0.005,
    "fin_Latn": 0.004,
    "ukr_Cyrl": 0.004,
    "slk_Latn": 0.004,
    "dan_Latn": 0.004,
    "lit_Latn": 0.004,
    "lvs_Latn": 0.004,
    "slv_Latn": 0.004,
    "est_Latn": 0.004,
    "bul_Cyrl": 0.004,
    "nob_Latn": 0.001,
    "cat_Latn": 0.001,
    "hrv_Latn": 0.0005,
    "gle_Latn": 0.0005,
    "mlt_Latn": 0.0005,
    "glg_Latn": 0.0005,
}

EUROLLM_FRACS_ANNEALED = {
    "eng_Latn": 0.325,
    "deu_Latn": 0.06,
    "spa_Latn": 0.06,
    "fra_Latn": 0.06,
    "ita_Latn": 0.06,
    "por_Latn": 0.035,
    "cmn_Hani": 0.035,
    "rus_Cyrl": 0.025,
    "pol_Latn": 0.025,
    "nld_Latn": 0.025,
    "jpn_Jpan": 0.025,
    "arb_Arab": 0.015,
    "tur_Latn": 0.015,
    "swe_Latn": 0.015,
    "kor_Hang": 0.015,
    "ces_Latn": 0.015,
    "hin_Deva": 0.01,
    "ell_Grek": 0.01,
    "hun_Latn": 0.01,
    "ron_Latn": 0.01,
    "fin_Latn": 0.008,
    "ukr_Cyrl": 0.008,
    "slk_Latn": 0.008,
    "dan_Latn": 0.008,
    "lit_Latn": 0.008,
    "lvs_Latn": 0.008,
    "slv_Latn": 0.008,
    "est_Latn": 0.008,
    "bul_Cyrl": 0.008,
    "nob_Latn": 0.002,
    "cat_Latn": 0.002,
    "hrv_Latn": 0.001,
    "gle_Latn": 0.001,
    "mlt_Latn": 0.001,
    "glg_Latn": 0.001,
}

BATCH_SIZE = 3072


def get_data_from_hf(lang, num_samples=1):
    """Stream a small number of examples for one FineWeb2-HQ language."""
    if lang.endswith("_missing"):
        print(f"Language {lang} marked as missing, returning empty data.")
        return []

    ds = datasets.load_dataset("epfml/FineWeb2-HQ", lang, stream=True)
    data = []

    for i in range(num_samples):
        sample = next(iter(ds["train"]))
        data.append(sample["text"])

    return data


def get_first_n_from_all_parquets(
    corpus_name="epfml/FineWeb2-HQ",
    langs=None,
    n_samples_per_lang=1000,
    max_parquets_per_lang=10,
    store_path="results",
):
    """Sample up to ``n_samples_per_lang`` examples per language from streamed parquet shards."""
    if langs is None:
        langs = []

    repo = HfApi().dataset_info(corpus_name)
    samples = []
    i = 0
    parquet_names = []
    for f in tqdm(repo.siblings):
        i += 1
        fname = f.rfilename
        lang = fname.rsplit("/", 1)[0]
        is_parquet = fname.endswith(".parquet")
        if lang in langs and is_parquet:
            parquet_names.append(fname)

    sampled_parquet_names = []
    # sample 10 parquets from each lang
    for lang in langs:
        parquet_names = [
            f.rfilename for f in repo.siblings if f.rfilename.startswith(lang) and f.rfilename.endswith(".parquet")
        ]

        if len(parquet_names) == 0:
            print(f"No parquet files found for language {lang}, skipping.")
            continue
        if len(parquet_names) > max_parquets_per_lang:
            # sample subset of parquets
            ids = list(range(len(parquet_names)))
            random.shuffle(ids)
            lang_parquet_names = [parquet_names[i] for i in ids[:max_parquets_per_lang]]
        else:
            lang_parquet_names = parquet_names
        sampled_parquet_names.append((lang, lang_parquet_names))

    # load samples
    for lang, parquet_names in sampled_parquet_names:
        json_path = f"{store_path}/{lang}_samples.json"
        lang_samples = []
        if os.path.exists(json_path):
            print(f"Loading samples for language {lang} from existing file {json_path}...")
            with open(json_path, "r") as f:
                for line in f:
                    dict_sample = json.loads(line.strip())
                    lang_samples.append(dict_sample)
            print(f"Loaded {len(lang_samples)} samples for language {lang} from existing file.")
        else:
            for fname in parquet_names:
                print(f"Loading hf data for language {lang} from file {fname}...")
                ds = datasets.load_dataset(corpus_name, data_files=fname, streaming=True)
                count = n_samples_per_lang // len(parquet_names)
                for sample in ds["train"]:
                    if sample.get("ts", ""):
                        sample.pop("ts")
                    lang_samples.append(sample)
                    count -= 1
                    if count == 0:
                        break
                print(f"Loaded {n_samples_per_lang // len(parquet_names)} samples for language {lang}.")
            # store samples
            with open(json_path, "w") as f:
                for sample in lang_samples:
                    f.write(json.dumps(sample) + "\n")
        # store lang code
        for sample in lang_samples:
            sample["lang_code"] = lang
            if "embeddings" in sample:
                del sample["embeddings"]
        samples.extend(lang_samples)

    return samples


def create_train_batch_with_fracs(fracs, samples, total_size=BATCH_SIZE):
    """Create a JSON-serializable training batch matching language fractions."""
    batch = []
    for lang, frac in fracs.items():
        n_samples = int(frac * total_size)
        print(f"Sampling {n_samples} samples for language {lang} with fraction {frac}.")
        lang_samples = [
            random.choice([s for s in samples if s.get("lang_code", "") == lang or s.get("language", "") == lang])
            for _ in range(n_samples)
        ]
        batch.extend(lang_samples)
    print(f"Created train batch with total size {len(batch)}.")
    return batch


def main():
    """Sample EuroLLM phase-one and phase-two language batches into ``results/lang_batches``."""
    eurollm_coverage = sum(EUROLLM_FRACS.values())
    print(f"First phase EuroLLM coverage: {eurollm_coverage * 100:.2f}%")

    second_eurollm_coverage = sum(EUROLLM_FRACS_ANNEALED.values())
    print(f"Second phase EuroLLM coverage: {second_eurollm_coverage * 100:.2f}%")

    if not os.path.exists("results/lang_batches/"):
        os.makedirs("results/lang_batches/")

    # get all parquet names here
    # repo = HfApi().dataset_info("epfml/FineWeb2-HQ")

    all_langs_fwhq = NEEDED_FWHQ + LANGS
    samples1 = get_first_n_from_all_parquets(
        corpus_name="epfml/FineWeb2-HQ", langs=all_langs_fwhq, store_path="results/lang_batches/"
    )
    print(f"Total FWHQ2 samples collected: {len(samples1)}")

    # save to disc
    with open("results/eurollm_fineweb2hq_samples.jsonl", "w") as f:
        for sample in samples1:
            f.write(json.dumps(sample) + "\n")

    # get koarean sample from https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned/tree/main/kor_Hang
    # repo_ko = HfApi().dataset_info("HPLT/HPLT2.0_cleaned")
    samples2 = get_first_n_from_all_parquets(
        corpus_name="HPLT/HPLT2.0_cleaned", langs=NEEDED_HPLT, store_path="results/lang_batches/"
    )
    print(f"Total HPLT samples collected: {len(samples2)}")

    # save to disc
    with open("results/eurollm_hplt_samples.jsonl", "w") as f:
        for sample in samples2:
            f.write(json.dumps(sample) + "\n")

    all_samples = samples1 + samples2

    # get english samples from fineweb-hq
    eng_samples = get_first_n_from_all_parquets(
        corpus_name="epfml/FineWeb-HQ", langs=["data"], n_samples_per_lang=5000, store_path="results/lang_batches/"
    )
    print(f"Total English samples from FineWeb-HQ collected: {len(eng_samples)}")

    # add language code to english samples
    for sample in eng_samples:
        sample["lang_code"] = "eng_Latn"

    all_samples.extend(eng_samples)
    print(f"Total samples collected: {len(all_samples)}")

    # create train batch
    train_batch = create_train_batch_with_fracs(EUROLLM_FRACS, all_samples, total_size=BATCH_SIZE)

    with open("results/lang_batches/eurollm_phase1_train_batch.jsonl", "w") as f:
        for sample in train_batch:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved EuroLLM phase 1 train batch with {len(train_batch)} samples.")

    # create annealed train batch
    annealed_train_batch = create_train_batch_with_fracs(EUROLLM_FRACS_ANNEALED, all_samples, total_size=BATCH_SIZE)
    with open("results/lang_batches/eurollm_phase2_annealed_train_batch.jsonl", "w") as f:
        for sample in annealed_train_batch:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved EuroLLM phase 2 annealed train batch with {len(annealed_train_batch)} samples.")


if __name__ == "__main__":
    main()
