Nama    : Ilham Notohusodo
NIM     : 24120500018   
Kasus A: Deteksi penipuan pembukaan rekening bank

Sebuah bank digital menerima ribuan pengajuan buka rekening online setiap hari. Sebagian kecil diajukan penipu, misalnya dengan identitas palsu atau untuk menampung uang hasil kejahatan. Tim risiko ingin setiap pengajuan baru diberi skor: seberapa mungkin pengajuan ini penipuan

Data: Bank Account Fraud (BAF) varian base, bulan 0-3. Satu baris = satu pengajuan, dengan satu kolom label penipuan
Tujuan: tangkap sebanyak mungkin penipu, dengan syarat nasabah jujur yang ikut tertandai paling banyak 5%. Setiap nasabah jujur yang tertandai berarti calon pelanggan hilang dan tambahan kerja verifikasi manual
Angka yang dilaporkan: persentase penipu yang tertangkap saat nasabah jujur yang salah tertandai dibatasi 5% (TPR pada FPR 5%)

Kasus A
Bank Account Fraud (BAF), unduh di Kaggle: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022. Pakai varian Base saja, baris dengan kolom month 0-3. Penjelasan kolom ada di datasheet pada repo resmi: https://github.com/feedzai/bank-account-fraud


# Deteksi Penipuan Pembukaan Rekening Bank (BAF Base, bulan 0-3)

Skor kemungkinan penipuan untuk tiap pengajuan rekening baru. Tujuan: tangkap penipu sebanyak mungkin dengan nasabah jujur tertandai maksimal 5% (metrik: **TPR pada FPR 5%**).

## Data
`Base.csv` (Kaggle BAF, via `kagglehub`), hanya `month` 0-3: 547.975 dari 1.000.000 baris. `device_fraud_count` dibuang karena konstan (selalu 0).

## Metode
- **Preprocessing** (satu `Pipeline`, jadi model menerima tabel mentah dan bebas kebocoran data): nilai `-1` pada kolom penanda data hilang (mis. `prev_address_months_count`, `bank_months_count`) diisi median data latih plus kolom penanda "hilang"; 5 kolom kategorikal di-encode ordinal dan dibaca sebagai kategori. `intended_balcon_amount` bernilai negatif kontinu (bukan `-1`), dibaca langsung.
- **Model:** `HistGradientBoostingClassifier` (scikit-learn), 29 fitur; `fraud_bool` dan `month` bukan fitur. Tiga konfigurasi dibandingkan; terpilih `sedang` (31 daun, learning rate 0,05, 215 iterasi, early stopping).

## Alasan memilih
Gradient boosting kuat untuk data tabular, tanpa scaling, mendukung kategori secara native, dan cepat untuk ratusan ribu baris. Penanda "hilang" dipertahankan karena data kosong bisa menjadi sinyal penipuan. Tanpa oversampling: threshold langsung dari FPR target.

## Cara membagi data
Berdasarkan **waktu**, bukan acak: **latih bulan 0-1** (260.060 baris), **validasi bulan 2** (136.979; 1.198 fraud), **uji bulan 3** (150.936). Di produksi model menilai pengajuan masa depan dan pola penipuan bergeser, jadi split acak membuat skor terlihat terlalu bagus. Konfigurasi dan threshold dipilih hanya di bulan 2; bulan 3 hanya mengecek threshold itu.

## Skor di data validasi (bulan 2)
**TPR pada FPR 5% = 53,0%** (FPR aktual 5,0%), ROC-AUC 0,882, PR-AUC 0,130, threshold 0,03008. Tiga konfigurasi memberi 52,6-53,0% (selisih di bawah kesalahan acak ±1,4 poin): setara.

<img src="hasil_model/roc_validasi.png" alt="Kurva ROC validasi: AUC 0,882; TPR 0,53 pada FPR 5%" width="230"><br><em>Kurva ROC validasi; titik merah = TPR 0,53 pada FPR 5%.</em>

## Hasil uji (bulan 3) dan keterbatasan
Dengan threshold yang sama, bulan 3 memberi TPR 68,2% tetapi FPR **8,6%** (melewati 5%): pola data bergeser antarbulan, jadi threshold perlu dihitung ulang berkala. Angka resmi tetap skor validasi.

## Memuat model (scikit-learn harus versi yang sama dengan saat melatih)
```python
m = joblib.load("model/baf_model.joblib")   # pipeline, threshold, feature_cols
skor = m["pipeline"].predict_proba(df_baru[m["feature_cols"]])[:, 1]
tandai = skor >= m["threshold"]
```
