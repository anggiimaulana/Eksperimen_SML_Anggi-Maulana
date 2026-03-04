import json
import logging
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Path
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR         = os.path.join(BASE_DIR, "TwitterEmotion_raw")
OUTPUT_DIR      = os.path.join(BASE_DIR, "preprocessing", "twitter_emotion_preprocessing")
PATH_DATASET_1  = os.path.join(RAW_DIR, "EmoTweetID-Human.csv")
PATH_DATASET_2  = os.path.join(RAW_DIR, "Twitter_Emotion_Dataset.csv")
PATH_SLANG_DICT = os.path.join(RAW_DIR, "kamus_singkatan.xlsx")

RANDOM_SEED = 42
TEST_SIZE   = 0.2
MAX_LENGTH  = 128

np.random.seed(RANDOM_SEED)


# 1. LOAD DATA
def load_datasets():
    """Memuat dan menggabungkan dua dataset tweet emosi."""
    log.info("Memuat Dataset 1: %s", PATH_DATASET_1)
    df1 = pd.read_csv(PATH_DATASET_1)
    df1.columns = ["id", "tweet", "label"]
    df1 = df1[["tweet", "label"]]

    log.info("Memuat Dataset 2: %s", PATH_DATASET_2)
    df2 = pd.read_csv(PATH_DATASET_2, sep=None, engine="python")
    df2.columns = ["label", "tweet"]
    df2 = df2[["tweet", "label"]]

    df = pd.concat([df1, df2], ignore_index=True)
    log.info("Dataset digabungkan: %d baris", len(df))
    return df


def load_slang_dict():
    """Memuat kamus slang bahasa Indonesia dari file .xlsx."""
    log.info("Memuat kamus slang: %s", PATH_SLANG_DICT)
    df_slang = pd.read_excel(PATH_SLANG_DICT, header=None)
    slang_dict = dict(zip(df_slang[0], df_slang[1]))
    log.info("Kamus slang: %d entri", len(slang_dict))
    return slang_dict


# 2. CLEANING
def remove_duplicates(df):
    """Menghapus tweet duplikat."""
    before = len(df)
    df = df.drop_duplicates(subset=["tweet"]).reset_index(drop=True)
    log.info("Duplikat dihapus: %d → %d baris", before, len(df))
    return df


def handle_missing_values(df):
    """Menghapus baris dengan nilai null."""
    before = len(df)
    df = df.dropna(subset=["tweet", "label"]).reset_index(drop=True)
    log.info("Missing values dihapus: %d → %d baris", before, len(df))
    return df


# 3. TEXT PREPROCESSING
def build_preprocess_fn(slang_dict):
    """
    Membuat fungsi preprocessing teks dengan kamus slang yang sudah dimuat.
    Sama persis dengan fungsi di notebook eksperimen.
    """
    def preprocess_text(text):
        # Step 1: lowercase
        text = str(text).lower()
        # Step 2: Hapus URL, mention, hashtag
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@[A-Za-z0-9_]+", "", text)
        text = re.sub(r"#\w+", "", text)
        # Step 3: Normalisasi karakter berulang
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        # Step 4: Hapus karakter non-alfabet
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        # Step 5: Normalisasi slang
        words = [slang_dict.get(w, w) for w in text.split()]
        text  = " ".join(words)
        # Step 6: Hapus kata tawa berlebihan
        text = re.sub(r"\b(wk|ha|he|hi|ho)+\b", "", text, flags=re.IGNORECASE)
        # Step 7: Bersihkan whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return preprocess_text


def apply_preprocessing(df, slang_dict):
    """Menerapkan preprocessing ke seluruh dataset."""
    log.info("Menerapkan preprocessing teks...")
    fn = build_preprocess_fn(slang_dict)
    df["clean_tweet"] = df["tweet"].apply(fn)

    before = len(df)
    df = df[df["clean_tweet"].str.strip() != ""].reset_index(drop=True)
    log.info("Teks kosong dihapus: %d → %d baris", before, len(df))
    return df


# 4. LABEL ENCODING & SPLIT
def encode_labels(df):
    """Label encoding pada kolom 'label'."""
    df["label"] = df["label"].str.strip().str.lower()
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    label_mapping = {
        label: int(idx)
        for label, idx in zip(le.classes_, le.transform(le.classes_))
    }
    num_labels = len(le.classes_)

    log.info("Label encoding: %d kategori", num_labels)
    for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        count = (df["label"] == label).sum()
        log.info("  %d → %-12s (%d data)", idx, label, count)

    return df, le, label_mapping, num_labels


def split_dataset(df):
    """Train-test split dengan stratifikasi."""
    df_train, df_val = train_test_split(
        df, test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label_id"],
    )
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    log.info("Split: train=%d (%.0f%%), val=%d (%.0f%%)",
             len(df_train), len(df_train)/len(df)*100,
             len(df_val),   len(df_val)/len(df)*100)
    return df_train, df_val


# 5. SAVE OUTPUT
def save_outputs(df, df_train, df_val, le, label_mapping, num_labels):
    """Menyimpan hasil preprocessing ke OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = ["clean_tweet", "label", "label_id"]

    df_train[cols].to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    df_val[cols].to_csv(  os.path.join(OUTPUT_DIR, "val.csv"),   index=False)
    df[["tweet"] + cols].to_csv(os.path.join(OUTPUT_DIR, "full.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    metadata = {
        "num_labels"    : num_labels,
        "label_mapping" : label_mapping,
        "total_samples" : len(df),
        "train_samples" : len(df_train),
        "val_samples"   : len(df_val),
        "max_length"    : MAX_LENGTH,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Output disimpan di: %s", OUTPUT_DIR)
    log.info("  train.csv        : %d baris", len(df_train))
    log.info("  val.csv          : %d baris", len(df_val))
    log.info("  full.csv         : %d baris", len(df))
    log.info("  label_encoder.pkl")
    log.info("  metadata.json")


# 6. PIPELINE UTAMA
def run_pipeline():
    log.info("=" * 60)
    log.info("MEMULAI PIPELINE PREPROCESSING — Anggi Maulana")
    log.info("=" * 60)

    df         = load_datasets()
    slang_dict = load_slang_dict()

    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = apply_preprocessing(df, slang_dict)

    df, le, label_mapping, num_labels = encode_labels(df)
    df_train, df_val = split_dataset(df)

    save_outputs(df, df_train, df_val, le, label_mapping, num_labels)

    log.info("=" * 60)
    log.info("✅ PIPELINE SELESAI!")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()