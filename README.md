# Saudi Employee Attrition Analysis & Predictive Modeling 👥📊

## Deskripsi Proyek
Proyek ini merupakan solusi *end-to-end Machine Learning* untuk menganalisis dan memprediksi tingkat pergantian karyawan (*Employee Attrition*) menggunakan dataset "Saudi Employee Attrition". Tujuan utama dari proyek ini adalah untuk mengidentifikasi pola penyebab karyawan keluar dan membangun model prediktif untuk membantu departemen HR dalam mengambil keputusan retensi yang tepat sasaran.

## Teknologi & Library
- **Environment:** Python (Jupyter Notebook / Google Colab)
- **Manipulasi Data:** `pandas`, `numpy`
- **Visualisasi:** `matplotlib`, `seaborn`
- **Machine Learning Pipeline:** `scikit-learn` (Decision Tree, Random Forest), `xgboost`
- **Penanganan Data Tidak Seimbang:** `imbalanced-learn` (SMOTE)

## Alur Kerja (Workflow) Sistem
1. **Exploratory Data Analysis (EDA):**
   - Menganalisis distribusi data dan mengeksplorasi hubungan antar variabel numerik menggunakan *Heatmap Correlation*.
   - Mendeteksi *outlier* pada 34 fitur fitur menggunakan *Boxplot*.
2. **Data Preprocessing & Cleaning:**
   - Membersihkan 85 baris data duplikat pada data latih.
   - Mengisi nilai kosong (*missing values*) menggunakan nilai *Median* (Imputasi).
   - Menangani *outlier* menggunakan metode **IQR (Interquartile Range) Capping** agar tidak mendistorsi model.
3. **Feature Engineering & Resampling:**
   - Mengatasi ketidakseimbangan kelas (*imbalanced data*) pada target prediksi menggunakan metode **SMOTE** (*Synthetic Minority Over-sampling Technique*).
   - Melakukan seleksi fitur menggunakan **SelectKBest** (F-statistic) untuk mengambil 15 fitur paling relevan dan mengurangi dimensi data.
4. **Pemodelan & Hyperparameter Tuning:**
   - Membandingkan tiga algoritma utama: Decision Tree, Random Forest, dan XGBoost (sebelum dan sesudah preprocessing).
   - Melakukan optimasi parameter (*Hyperparameter Tuning*) menggunakan **GridSearchCV** untuk mencari kombinasi parameter terbaik pada Decision Tree dan Random Forest.
5. **Evaluasi:**
   - Memvisualisasikan *Confusion Matrix*.
   - Mengukur metrik komprehensif seperti *Accuracy, Precision, Recall, F1-Score*, dan metrik *AUC-ROC*.

## Hasil dan Kesimpulan
- **Prapemrosesan:** Penggunaan teknik SMOTE dan penanganan *outlier* secara signifikan membantu model dalam mengenali pola kelas minoritas (karyawan yang keluar).
- **Performa Model:** Setelah melalui proses *Hyperparameter Tuning* menggunakan GridSearchCV, model **Random Forest** terpilih sebagai model terbaik dengan kombinasi parameter: `{'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}`.
- **Akurasi Akhir:** Model Random Forest berhasil mencapai skor performa terbaik di angka **~81.2%** (0.8119).

## Cara Menjalankan Notebook
1. Clone repositori ini.
2. Pastikan file dataset `Data Train Saudi Employee Attrition.csv` berada pada direktori yang tepat.
3. Buka file `employee_attrition_analysis.ipynb` menggunakan Jupyter Notebook atau Google Colab.
4. Jalankan setiap *cell* secara berurutan untuk mereplikasi proses analisis dan pelatihan model.
