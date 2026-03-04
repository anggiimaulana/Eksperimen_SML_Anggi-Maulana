# Eksperimen_SML_Anggi-Maulana

Repository eksperimen **Kriteria 1** — Membangun Sistem Machine Learning  
Dicoding x IBM 2026 | Anggi Maulana

---

## 📁 Struktur Folder

```
Eksperimen_SML_Anggi-Maulana/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          ← GitHub Actions (Advanced)
├── namadataset_raw/                   ← Dataset mentah (push ke GitHub)
│   ├── EmoTweetID-Human.csv
│   ├── Twitter_Emotion_Dataset.csv
│   └── kamus_singkatan.csv
└── preprocessing/
    ├── Eksperimen_Anggi-Maulana.ipynb ← Jalankan di Colab
    ├── automate_Anggi-Maulana.py      ← Jalankan di VSCode/Actions
    └── twitter_emotion_preprocessing/ ← Output (auto-generated)
        ├── train.csv
        ├── val.csv
        ├── full.csv
        ├── label_encoder.pkl
        └── metadata.json
```

---

## 🚀 Cara Menjalankan

### Step 1 — Jalankan Notebook di Colab
1. Upload folder `namadataset_raw/` ke Google Drive atau clone repo ini di Colab
2. Buka `preprocessing/Eksperimen_Anggi-Maulana.ipynb` di Colab
3. Jalankan semua cell dari atas ke bawah
4. Pastikan tidak ada error

### Step 2 — Jalankan Script Otomatis di VSCode
```bash
# Di terminal VSCode, dari root folder repo
pip install pandas numpy scikit-learn

python preprocessing/automate_Anggi-Maulana.py
```

### Step 3 — Push ke GitHub
```bash
git init
git add .
git commit -m "first commit: eksperimen preprocessing"
git branch -M main
git remote add origin https://github.com/anggi-maulana/Eksperimen_SML_Anggi-Maulana.git
git push -u origin main
```

### Step 4 — GitHub Actions Otomatis Berjalan
Setelah push, cek tab **Actions** di GitHub repo.  
Workflow akan otomatis jalankan preprocessing dan commit hasilnya.

---

## 📊 Dataset

**Indonesian Twitter Emotion** — Kaggle  
Kategori: anger, fear, happy, love, sadness, surprise
