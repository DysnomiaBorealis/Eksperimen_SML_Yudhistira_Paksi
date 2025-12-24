# Eksperimen_SML_Yudhistira-Paksi

Repository untuk eksperimen machine learning pada dataset spam berbahasa Indonesia.

## Deskripsi Proyek

Proyek ini merupakan implementasi sistem deteksi spam untuk pesan berbahasa Indonesia menggunakan teknik Natural Language Processing (NLP) dan Machine Learning. Eksperimen ini mencakup proses eksplorasi data, preprocessing, dan pembuatan model klasifikasi.

## Dataset

- **Nama Dataset**: Indo Spam Dataset
- **File**: `indo_spam_raw.csv`
- **Kolom**:
  - `Kategori`: Label kelas (Spam/ham)
  - `Pesan`: Teks pesan dalam bahasa Indonesia
- **Jumlah Sampel**: ~1100+ pesan
- **Distribusi Kelas**: Spam dan Ham (tidak spam)

## Struktur Repository

```
Eksperimen_SML_Yudhistira-Paksi/
├── .workflow/
│   └── workflows/
│       └── preprocess.yml          # GitHub Actions workflow
├── indo_spam_raw.csv               # Dataset mentah (raw)
├── preprocessing/
│   ├── Eksperimen_Yudhistira_Paksi.ipynb    # Notebook eksperimen
│   ├── automate_Yudhistira_Paksi.py         # Script otomatisasi preprocessing
│   └── indo_spam_preprocessing.csv          # Dataset hasil preprocessing
└── README.md                       # Dokumentasi proyek
```

## Cara Menjalankan Eksperimen

### Prerequisites

Pastikan Anda telah menginstall:
- Python 3.8+
- pip

### Instalasi Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Atau menggunakan requirements.txt:

```bash
pip install -r requirements.txt
```

### Menjalankan Notebook Eksperimen

1. Buka Google Colab atau Jupyter Notebook
2. Upload file `preprocessing/Eksperimen_Yudhistira_Paksi.ipynb`
3. Upload dataset `indo_spam.csv` ke direktori yang sama
4. Jalankan semua cell secara berurutan

### Menjalankan Script Otomatisasi

Untuk menjalankan preprocessing secara otomatis:

```bash
python preprocessing/automate_Yudhistira_Paksi.py
```

Script ini akan:
1. Membaca dataset mentah (`indo_spam_raw.csv`)
2. Melakukan preprocessing (cleaning, tokenization, dll)
3. Menyimpan hasil ke `preprocessing/indo_spam_preprocessing.csv`
4. **Mengembalikan data yang siap dilatih** (training-ready data):
   - `X_train_tfidf`: Fitur training yang sudah divektorisasi
   - `X_test_tfidf`: Fitur testing yang sudah divektorisasi
   - `y_train`: Label training
   - `y_test`: Label testing
   - `vectorizer`: TF-IDF vectorizer yang sudah di-fit

## GitHub Actions Workflow

Repository ini dilengkapi dengan GitHub Actions untuk otomatisasi preprocessing.

### Cara Kerja Workflow

1. **Trigger**: Workflow akan berjalan otomatis ketika:
   - Ada push ke branch `main` yang mengubah `indo_spam_raw.csv` atau script preprocessing
   - Dapat juga di-trigger secara manual melalui GitHub Actions UI

2. **Proses**:
   - Setup Python environment
   - Install dependencies (pandas, numpy, scikit-learn)
   - Menjalankan script preprocessing
   - Commit dan push hasil preprocessing kembali ke repository
   - Upload hasil sebagai artifact

3. **Output**:
   - File `preprocessing/indo_spam_preprocessing.csv` akan di-update otomatis
   - Preprocessed dataset tersedia sebagai artifact selama 30 hari

### Cara Mengaktifkan GitHub Actions

1. Buat repository di GitHub dengan nama `Eksperimen_SML_Yudhistira-Paksi`
2. Push semua file ke repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Spam detection experiment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/Eksperimen_SML_Yudhistira-Paksi.git
   git push -u origin main
   ```
3. GitHub Actions akan otomatis tersedia dan berjalan

## Tahapan Preprocessing

Script otomatisasi ini mengkonversi langkah-langkah dari notebook eksperimen dengan struktur yang berbeda namun tahapan yang sama:

### Notebook Eksperimen
1. Import Library
2. Memuat Dataset
3. Exploratory Data Analysis (EDA)
   - Cek info dataset, duplikat, missing values
   - Cek distribusi kelas dan visualisasi
   - Analisis panjang pesan dan word count
4. Data Preprocessing
   - Menangani data kosong
   - Menghapus data duplikat
   - Text cleaning dan encoding
   - Encode target variable
   - Split data (training/testing)
   - Vektorisasi (TF-IDF)
   - Simpan data preprocessing
   - Validasi model (Naive Bayes)

### Script Otomatisasi (automate_Yudhistira_Paksi.py)
Konversi dari notebook ke struktur fungsi modular:

1. **load_dataset()** - Memuat dataset mentah
2. **handle_missing_values()** - Menangani nilai kosong
3. **remove_duplicates()** - Menghapus duplikat
4. **perform_text_cleaning_and_encoding()** - Cleaning teks dan encoding label
5. **split_data()** - Membagi data training dan testing
6. **vectorize_text()** - TF-IDF vectorization
7. **save_preprocessed_data()** - Menyimpan hasil preprocessing

**Fungsi Utama**: `preprocess_and_prepare_data()` menjalankan semua tahapan dan **mengembalikan data yang siap untuk training model**.

## Hasil Eksperimen

Setelah menjalankan eksperimen, Anda akan mendapatkan:
- Dataset yang telah di-preprocessing
- Visualisasi distribusi data
- Model klasifikasi dengan metrik evaluasi (accuracy, precision, recall, F1-score)
- Confusion matrix
- Data yang siap untuk training model lebih lanjut


## Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan eksperimen.

---

**Note**: Pastikan untuk menjalankan preprocessing sebelum melakukan training model lebih lanjut.

