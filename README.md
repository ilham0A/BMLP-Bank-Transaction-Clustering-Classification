# Bank Transaction Clustering & Classification

Proyek akhir kelas **Belajar Machine Learning untuk Pemula (BMLP)** — Dicoding Indonesia, dikerjakan oleh **Muhammad Ilham** sebagai bagian dari **Program Studi Independent Bersertifikat - Mandiri | ASAH 2026 led by Dicoding supported by GoTo** jalur **Next-Gen AI Engineer**.

## Deskripsi

Proyek ini membangun pipeline machine learning dua tahap (unsupervised → supervised) pada dataset transaksi perbankan:

1. **Clustering** — mengelompokkan transaksi ke dalam segmen perilaku menggunakan **K-Means**, untuk menemukan pola tersembunyi pada data nasabah.
2. **Klasifikasi** — melatih model **Decision Tree** (dan Random Forest sebagai pembanding) untuk memprediksi segmen/cluster suatu transaksi berdasarkan fitur-fiturnya, sehingga data baru bisa langsung diklasifikasikan tanpa perlu clustering ulang.

## Dataset

Dataset berisi **2.512 sampel** transaksi perbankan dengan fitur meliputi nominal transaksi, tipe transaksi, lokasi, channel, usia dan pekerjaan nasabah, durasi transaksi, jumlah percobaan login, dan saldo akun.

## Struktur Proyek

| File                                                       | Deskripsi                                           |
| ---------------------------------------------------------- | --------------------------------------------------- |
| `[Clustering]_Submission_Akhir_BMLP_Muhammad_Ilham.ipynb`  | Notebook EDA, preprocessing, dan clustering K-Means |
| `[Klasifikasi]_Submission_Akhir_BMLP_Muhammad_Ilham.ipynb` | Notebook klasifikasi berdasarkan hasil cluster      |

## Metodologi

**Clustering:**

- Preprocessing: handling missing value, duplikat, feature encoding (`LabelEncoder`), scaling (`StandardScaler`)
- Penentuan jumlah cluster optimal dengan **Elbow Method** (`KElbowVisualizer`, metrik Silhouette Score)
- Training **K-Means** dengan `n_clusters=2`
- Evaluasi: **Silhouette Score = 0.576**
- Visualisasi cluster 2D menggunakan **PCA**
- Interpretasi karakteristik tiap cluster (sebelum & sesudah inverse transform)

**Klasifikasi:**

- Fitur kategorikal di-encode dengan One-Hot Encoding
- Split data train/test dengan `stratify` untuk menjaga proporsi kelas
- Model utama: **Decision Tree Classifier**
- Model pembanding: **Random Forest Classifier**
- Hyperparameter tuning dengan **GridSearchCV** (5-fold cross-validation)
- Evaluasi: accuracy, precision, recall, F1-score (`classification_report`)

## Hasil Utama

- Silhouette Score clustering: **0.576** (k=2)
- Model terbaik hasil tuning: Random Forest dengan `max_depth=10`, `n_estimators=300`

## Cara Menjalankan

1. Clone repository ini:

   ```bash
   git clone https://github.com/ilham0A/BMLP-Bank-Transaction-Clustering-Classification
   cd BMLP-Bank-Transaction-Clustering-Classification
   ```

2. Install dependency yang dibutuhkan:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib
   ```

3. Buka notebook secara berurutan:
   - Jalankan `[Clustering]_...ipynb` terlebih dahulu (menghasilkan `data_clustering.csv` dan model clustering)
   - Lanjutkan dengan `[Klasifikasi]_...ipynb` (menggunakan `data_clustering.csv` sebagai input)

4. Muat model yang sudah disimpan:
   ```python
   import joblib
   model = joblib.load("model_clustering.h5")
   decision_tree_model = joblib.load("decision_tree_model.h5")
   ```

## Tools & Library

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Yellowbrick, Joblib

## Lisensi

Proyek ini dibuat untuk keperluan submission akademik kelas Dicoding "Belajar Machine Learning untuk Pemula" dan diunggah dengan izin dari pihak Dicoding.

---

**Muhammad Ilham** — Informatics Student
