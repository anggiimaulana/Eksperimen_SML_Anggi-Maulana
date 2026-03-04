"""
automate_NamaSiswa.py
=====================
Script otomatisasi preprocessing dataset Indonesian Twitter Emotion
untuk fine-tuning IndoBERT pada sistem klasifikasi emosi Myfess.

Penggunaan:
    python automate_NamaSiswa.py \
        --dataset1 path/ke/EmoTweetID-Human.csv \
        --dataset2 path/ke/Twitter_Emotion_Dataset.csv \
        --slang    path/ke/kamus_singkatan.csv \
        --output   twitter_emotion_preprocessing/

Output:
    - twitter_emotion_preprocessing/train.csv
    - twitter_emotion_preprocessing/val.csv
    - twitter_emotion_preprocessing/full.csv
    - twitter_emotion_preprocessing/label_encoder.pkl
    - twitter_emotion_preprocessing/metadata.json
"""

import argparse
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Konstanta
RANDOM_SEED = 42
TEST_SIZE   = 0.2
MAX_LENGTH  = 128


# 1. LOAD DATA

def load_datasets(path_dataset1: str, path_dataset2: str) -> pd.DataFrame:
    """
    Memuat dan menggabungkan dua dataset tweet emosi Indonesia.

    Args:
        path_dataset1: Path ke EmoTweetID-Human.csv
        path_dataset2: Path ke Twitter_Emotion_Dataset.csv

    Returns:
        DataFrame gabungan dengan kolom ['tweet', 'label']
    """
    log.info("Memuat Dataset 1: %s", path_dataset1)
    df1 = pd.read_csv(path_dataset1)
    df1.columns = ["id", "tweet", "label"]
    df1 = df1[["tweet", "label"]]

    log.info("Memuat Dataset 2: %s", path_dataset2)
    df2 = pd.read_csv(path_dataset2, sep=None, engine="python")
    df2.columns = ["label", "tweet"]
    df2 = df2[["tweet", "label"]]

    df = pd.concat([df1, df2], ignore_index=True)
    log.info("Dataset digabungkan: %d baris total", len(df))
    return df


def load_slang_dict(path_slang: str) -> dict:
    """
    Memuat kamus slang/singkatan bahasa Indonesia.

    Args:
        path_slang: Path ke kamus_singkatan.csv

    Returns:
        Dictionary {slang: normal}
    """
    log.info("Memuat kamus slang: %s", path_slang)
    df_slang = pd.read_csv(path_slang, sep=None, engine="python", header=None)
    slang_dict = dict(zip(df_slang[0], df_slang[1]))
    log.info("Kamus slang dimuat: %d entri", len(slang_dict))
    return slang_dict


# 2. CLEANING

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus tweet duplikat."""
    before = len(df)
    df = df.drop_duplicates(subset=["tweet"])
    df = df.reset_index(drop=True)
    log.info("Duplikat dihapus: %d → %d baris", before, len(df))
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris dengan nilai null pada kolom tweet atau label."""
    before = len(df)
    df = df.dropna(subset=["tweet", "label"])
    df = df.reset_index(drop=True)
    log.info("Missing values dihapus: %d → %d baris", before, len(df))
    return df


# 3. TEXT PREPROCESSING

def build_preprocess_fn(slang_dict: dict):
    """
    Membuat fungsi preprocessing dengan kamus slang yang sudah dimuat.

    Args:
        slang_dict: Dictionary kamus slang {slang: normal}

    Returns:
        Fungsi preprocess_text(text) -> str
    """
    def preprocess_text(text: str) -> str:
        """
        Membersihkan dan menormalisasi teks tweet bahasa Indonesia.

        Tahapan:
            1. Konversi ke lowercase
            2. Hapus URL, mention (@username), hashtag (#tag)
            3. Normalisasi karakter berulang (maks 2 huruf)
            4. Hapus karakter non-alfabet
            5. Normalisasi slang menggunakan kamus
            6. Hapus kata tawa berlebihan (wk, ha, he, hi, ho)
            7. Hapus whitespace berlebih
        """
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
        words = text.split()
        words = [slang_dict.get(word, word) for word in words]
        text = " ".join(words)

        # Step 6: Hapus kata tawa berlebihan
        text = re.sub(r"\b(wk|ha|he|hi|ho)+\b", "", text, flags=re.IGNORECASE)

        # Step 7: Bersihkan whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    return preprocess_text


def apply_preprocessing(df: pd.DataFrame, slang_dict: dict) -> pd.DataFrame:
    """
    Menerapkan preprocessing teks ke seluruh dataset.

    Args:
        df: DataFrame dengan kolom 'tweet'
        slang_dict: Kamus slang

    Returns:
        DataFrame dengan kolom 'clean_tweet' ditambahkan
    """
    log.info("Menerapkan preprocessing teks...")
    preprocess_fn = build_preprocess_fn(slang_dict)
    df["clean_tweet"] = df["tweet"].apply(preprocess_fn)

    # Hapus baris kosong setelah preprocessing
    before = len(df)
    df = df[df["clean_tweet"].str.strip() != ""]
    df = df.reset_index(drop=True)
    log.info("Teks kosong dihapus: %d → %d baris", before, len(df))

    return df


# 4. LABEL ENCODING & SPLIT

def encode_labels(df: pd.DataFrame):
    """
    Melakukan label encoding pada kolom 'label'.

    Args:
        df: DataFrame dengan kolom 'label'

    Returns:
        Tuple (df_encoded, label_encoder, label_mapping, num_labels)
    """
    # Standardisasi label
    df["label"] = df["label"].str.strip().str.lower()

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    label_mapping = {
        label: int(idx)
        for label, idx in zip(le.classes_, le.transform(le.classes_))
    }
    num_labels = len(le.classes_)

    log.info("Label encoding selesai: %d kategori", num_labels)
    for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        count = (df["label"] == label).sum()
        log.info("  %d → %-12s (%d data)", idx, label, count)

    return df, le, label_mapping, num_labels


def split_dataset(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED):
    """
    Membagi dataset menjadi training dan validation set dengan stratifikasi.

    Args:
        df: DataFrame dengan kolom 'label_id'
        test_size: Proporsi data validasi (default 0.2)
        seed: Random seed

    Returns:
        Tuple (df_train, df_val)
    """
    df_train, df_val = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label_id"],
    )
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    log.info(
        "Dataset dibagi: train=%d (%.0f%%), val=%d (%.0f%%)",
        len(df_train), len(df_train) / len(df) * 100,
        len(df_val),   len(df_val)   / len(df) * 100,
    )
    return df_train, df_val


# 5. SAVE OUTPUT

def save_outputs(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    le: LabelEncoder,
    label_mapping: dict,
    num_labels: int,
    output_dir: str,
) -> None:
    """
    Menyimpan seluruh output preprocessing ke direktori yang ditentukan.

    Args:
        df: Dataset lengkap setelah preprocessing
        df_train: Dataset training
        df_val: Dataset validasi
        le: LabelEncoder yang sudah di-fit
        label_mapping: Mapping label ke ID
        num_labels: Jumlah kategori label
        output_dir: Direktori output
    """
    os.makedirs(output_dir, exist_ok=True)
    cols = ["clean_tweet", "label", "label_id"]

    # Simpan CSV
    df_train[cols].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val[cols].to_csv(  os.path.join(output_dir, "val.csv"),   index=False)
    df[["tweet"] + cols].to_csv(os.path.join(output_dir, "full.csv"), index=False)

    # Simpan label encoder
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Simpan metadata
    metadata = {
        "num_labels"    : num_labels,
        "label_mapping" : label_mapping,
        "total_samples" : len(df),
        "train_samples" : len(df_train),
        "val_samples"   : len(df_val),
        "max_length"    : MAX_LENGTH,
        "test_size"     : TEST_SIZE,
        "random_seed"   : RANDOM_SEED,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Output disimpan di: %s", output_dir)
    log.info("  - train.csv          (%d baris)", len(df_train))
    log.info("  - val.csv            (%d baris)", len(df_val))
    log.info("  - full.csv           (%d baris)", len(df))
    log.info("  - label_encoder.pkl")
    log.info("  - metadata.json")


# 6. PIPELINE UTAMA

def run_preprocessing_pipeline(
    path_dataset1: str,
    path_dataset2: str,
    path_slang: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing dari awal hingga simpan output.

    Args:
        path_dataset1: Path ke EmoTweetID-Human.csv
        path_dataset2: Path ke Twitter_Emotion_Dataset.csv
        path_slang   : Path ke kamus_singkatan.csv
        output_dir   : Direktori output hasil preprocessing

    Returns:
        DataFrame hasil preprocessing lengkap
    """
    log.info("=" * 60)
    log.info("MEMULAI PIPELINE PREPROCESSING")
    log.info("=" * 60)

    # 1. Load data
    df          = load_datasets(path_dataset1, path_dataset2)
    slang_dict  = load_slang_dict(path_slang)

    # 2. Cleaning
    df = remove_duplicates(df)
    df = handle_missing_values(df)

    # 3. Text preprocessing
    df = apply_preprocessing(df, slang_dict)

    # 4. Label encoding & split
    df, le, label_mapping, num_labels = encode_labels(df)
    df_train, df_val = split_dataset(df)

    # 5. Simpan output
    save_outputs(df, df_train, df_val, le, label_mapping, num_labels, output_dir)

    log.info("=" * 60)
    log.info("PIPELINE SELESAI")
    log.info("=" * 60)

    return df


# ENTRY POINT

def parse_args():
    parser = argparse.ArgumentParser(
        description="Otomatisasi preprocessing Indonesian Twitter Emotion Dataset"
    )
    parser.add_argument(
        "--dataset1", required=True,
        help="Path ke EmoTweetID-Human.csv"
    )
    parser.add_argument(
        "--dataset2", required=True,
        help="Path ke Twitter_Emotion_Dataset.csv"
    )
    parser.add_argument(
        "--slang", required=True,
        help="Path ke kamus_singkatan.csv"
    )
    parser.add_argument(
        "--output", default="twitter_emotion_preprocessing",
        help="Direktori output (default: twitter_emotion_preprocessing)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing_pipeline(
        path_dataset1=args.dataset1,
        path_dataset2=args.dataset2,
        path_slang=args.slang,
        output_dir=args.output,
    )
