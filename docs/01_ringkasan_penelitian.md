# Ringkasan Penelitian

## 1. Identitas Penelitian

- **Judul:** Pengembangan Model Proxy Means Test Berbasis Multi-Output Machine Learning untuk Prediksi Pengeluaran Perkapita
- **Penulis:** Fauzan Alfaraby Nirwan
- **Program Studi:** D-IV Komputasi Statistik, Peminatan Sains Data
- **Institusi:** Politeknik Statistika STIS
- **Wilayah penelitian:** D.I. Yogyakarta dan Jawa Tengah
- **Periode data:** Susenas 2022–2025

## 2. Latar Belakang

Proxy Means Test (PMT) merupakan pendekatan untuk memperkirakan kesejahteraan rumah tangga berdasarkan karakteristik sosial ekonomi yang relatif mudah diamati. Pendekatan ini dapat digunakan untuk membantu pemeringkatan rumah tangga dalam penargetan program bantuan sosial.

Pengeluaran rumah tangga terdiri atas komponen makanan dan nonmakanan yang memiliki pola berbeda, tetapi berpotensi saling berkaitan. Oleh karena itu, penelitian membandingkan pemodelan pengeluaran menggunakan pendekatan single-output dan multi-output berbasis machine learning.

## 3. Tujuan Penelitian

1. Menggambarkan pola pengeluaran per kapita dan karakteristik sosial ekonomi rumah tangga di D.I. Yogyakarta dan Jawa Tengah.
2. Mengembangkan dan membandingkan model PMT single-output dan multi-output menggunakan XGBoost dan Random Forest.
3. Menyajikan hasil analisis melalui web story interaktif.

## 4. Data Penelitian

Penelitian menggunakan data Susenas modul KOR dan KP tahun 2022–2025. Data tahun 2022–2024 digunakan sebagai data latih, sedangkan tahun 2025 digunakan sebagai data uji.

Pemodelan dilakukan pada tingkat kabupaten/kota. Variabel prediktor mencakup karakteristik demografi, pendidikan, pekerjaan, kondisi tempat tinggal, fasilitas dasar, dan kepemilikan aset.

Nilai pengeluaran disesuaikan ke basis tahun 2025 menggunakan deflator berbasis Garis Kemiskinan sesuai prosedur penelitian.

## 5. Pendekatan Pemodelan

Algoritma yang digunakan adalah XGBoost dan Random Forest.

- **Single-output:** memodelkan pengeluaran total atau komponen makanan dan nonmakanan secara terpisah.
- **Multi-output:** memodelkan pengeluaran makanan dan nonmakanan secara bersama dalam satu kerangka pemodelan.

Kinerja model dievaluasi menggunakan Root Mean Squared Error (RMSE), Kendall's tau-b, dan inclusion error kelompok 40% terbawah (IE40).

## 6. Hasil Utama

### D.I. Yogyakarta

XGB Components menghasilkan RMSE sebesar 0,44679, Kendall's tau-b sebesar 0,5979, dan IE40 sebesar 27,24%.

### Jawa Tengah

Multi-output Tree XGB menghasilkan RMSE terendah sebesar 0,43777 dan IE40 terendah sebesar 31,13%. Kendall's tau-b tertinggi sebesar 0,5392 diperoleh XGB Components.

Secara umum, XGBoost memberikan kinerja lebih baik dibandingkan Random Forest. Namun, keunggulan pendekatan multi-output tidak konsisten pada seluruh wilayah.

## 7. Luaran Penelitian

Luaran penelitian meliputi kode pemodelan, dokumentasi metodologi, dataset dummy untuk demonstrasi, panduan penggunaan, hasil evaluasi yang diizinkan untuk dipublikasikan, serta web story untuk menyampaikan hasil penelitian.

## 8. Batasan Publikasi

Repositori publik tidak boleh memuat mikrodata Susenas yang dibatasi aksesnya, identitas responden, token, password, atau kredensial. Dataset dummy hanya digunakan untuk menguji struktur dan alur kode, bukan sebagai pengganti data penelitian atau dasar penarikan kesimpulan.
