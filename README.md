# Pengembangan Model Proxy Means Test Berbasis Multi-Output Machine Learning untuk Prediksi Pengeluaran Perkapita

**Penulis:** Fauzan Alfaraby Nirwan  
**NIM:** 222212605  
**Program Studi:** D-IV Komputasi Statistik, Peminatan Sains Data  
**Institusi:** Politeknik Statistika STIS

## Deskripsi Proyek

Proyek ini merupakan artefak penelitian skripsi mengenai pengembangan model *Proxy Means Test* (PMT) berbasis *machine learning* untuk memprediksi pengeluaran per kapita rumah tangga. Penelitian membandingkan pendekatan *single-output* dan *multi-output* menggunakan algoritma XGBoost dan Random Forest pada Provinsi D.I. Yogyakarta dan Jawa Tengah.

## Tujuan Penelitian

1. Menggambarkan pola pengeluaran per kapita dan karakteristik sosial ekonomi rumah tangga.
2. Mengembangkan dan membandingkan model PMT untuk pengeluaran total, makanan, dan nonmakanan.
3. Mengevaluasi kinerja model menggunakan RMSE, Kendall's tau-b, dan *inclusion error* pada kelompok 40% terbawah (IE40).
4. Menyajikan hasil analisis melalui *web story*.

## Data dan Metodologi

Penelitian menggunakan data Susenas tahun 2022–2025. Data tahun 2022–2024 digunakan sebagai data latih, sedangkan tahun 2025 digunakan sebagai data uji. Pemodelan dilakukan pada tingkat kabupaten/kota dengan karakteristik sosial ekonomi rumah tangga sebagai prediktor.

## Hasil Utama

- **D.I. Yogyakarta:** XGB Components menghasilkan RMSE 0,44679, Kendall's tau-b 0,5979, dan IE40 27,24%.
- **Jawa Tengah:** Multi-output Tree XGB menghasilkan RMSE terendah 0,43777 dan IE40 terendah 31,13%. Kendall's tau-b tertinggi sebesar 0,5392 diperoleh XGB Components.
- Secara umum, XGBoost menunjukkan kinerja yang lebih baik daripada Random Forest dalam penelitian ini. Keunggulan pendekatan multi-output berbeda menurut wilayah.

## Struktur Proyek

- `Dataset/`: data penelitian atau data dummy yang diizinkan untuk dibagikan.
- `Source Code/`: kode preprocessing, pemodelan, dan evaluasi.
- `docs/`: dokumentasi penelitian dan panduan penggunaan.
- `notebooks/`: notebook eksplorasi data dan analisis.
- `outputs/`: keluaran evaluasi yang aman dipublikasikan.

## Cara Penggunaan

1. Baca dokumentasi pada folder `docs/`.
2. Siapkan lingkungan Python sesuai `requirements.txt`.
3. Periksa struktur dataset dan konfigurasi pada kode.
4. Jalankan preprocessing, pemodelan, dan evaluasi sesuai petunjuk yang tersedia.

## Keamanan dan Batasan Data

Repositori publik hanya boleh memuat data, kode, dan hasil yang diizinkan untuk dipublikasikan. Jangan mengunggah mikrodata Susenas yang dibatasi aksesnya, identitas responden, token, password, atau kredensial. Jika data penelitian tidak dapat dipublikasikan, gunakan data dummy sintetis minimal lima baris.

## Reproduksibilitas

Angka hasil penelitian harus merujuk pada kode final dan konfigurasi yang benar-benar digunakan dalam skripsi. Data dummy hanya untuk menguji struktur dan alur program, bukan untuk mereproduksi hasil penelitian.
