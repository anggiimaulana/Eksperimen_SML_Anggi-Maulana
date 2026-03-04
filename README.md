# Eksperimen_SML_NamaSiswa

Repository eksperimen untuk submission **Membangun Sistem Machine Learning** — Dicoding x IBM 2026.

## 📁 Struktur Folder

```
Eksperimen_SML_NamaSiswa/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions (Advanced)
├── namadataset_raw/                   # Dataset mentah (tidak di-track Git)
│   ├── EmoTweetID-Human.csv
│   ├── Twitter_Emotion_Dataset.csv
│   └── kamus_singkatan.csv
├── twitter_emotion_preprocessing/     # Output preprocessing (auto-generated)
│   ├── train.csv
│   ├── val.csv
│   ├── full.csv
│   ├── label_encoder.pkl
│   └── metadata.json
├── Eksperimen_NamaSiswa.ipynb         # Notebook eksperimen utama
├── automate_NamaSiswa.py              # Script preprocessing otomatis (Skilled)
└── README.md
```

## 🗂️ Dataset

**Indonesian Twitter Emotion Dataset**
- Sumber: [Kaggle](https://www.kaggle.com/datasets/dennisherdi/indonesian-twitter-emotion)
- Dua dataset digabungkan: EmoTweetID-Human + Twitter Emotion Dataset
- Kategori emosi: anger, fear, happy, love, sadness, surprise

## 🚀 Cara Menjalankan Preprocessing

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Jalankan preprocessing otomatis
python automate_NamaSiswa.py \
  --dataset1 namadataset_raw/EmoTweetID-Human.csv \
  --dataset2 namadataset_raw/Twitter_Emotion_Dataset.csv \
  --slang    namadataset_raw/kamus_singkatan.csv \
  --output   twitter_emotion_preprocessing
```

## ⚙️ GitHub Actions

Workflow preprocessing otomatis akan berjalan ketika:
1. Ada push ke branch `main` yang mengubah file dataset atau script
2. Di-trigger manual via `workflow_dispatch`

Output berupa dataset terproses akan langsung di-commit kembali ke repository.
