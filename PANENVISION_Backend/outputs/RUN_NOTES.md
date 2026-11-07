# RUN NOTES - Pipeline Forecasting (run_pipeline_forecast.py)

## Keputusan desain

- Kerangka: forecasting 1-langkah rolling. Titik ramal akhir bulan t-1. Model pohon dilatih sekali pada latih 0..66; SARIMA/SARIMAX memakai rolling state update (append aktual, refit=False) di bulan uji.
- Fitur: 22 fitur forecasting (dilengkapi prod_lag1 dan ratio_lag1, semua lag/klimatologi dari masa lalu atau latih) + kab_id/kom_id untuk pohon. Tidak ada curah_hujan_mm, cuaca_ekstrem, ndvi_modis, ndvi_anomaly bulan berjalan di fitur mana pun. Exog SARIMAX memakai hujan dan NDVI t-1.
- Hasil Headline: LightGBM (+ Lag 1) adalah model terbaik mutlak (MASE Global 0,313, MASE Macro 0,236, MAE 306,17 ton), mengungguli SARIMAX+XGB (0,322) dan SARIMA (0,326).
- populasi_jiwa bulan t dipertahankan sebagai variabel known-in-advance: pada dataset ini ia proyeksi demografis deterministik antar titik tengah tahun, dapat dihitung pada titik ramal. Kandidat ablasi pertama bila reviewer menolak justifikasi ini.
- Surplus: rend dari data latih saja; konsumsi bulan uji adalah proyeksi (per kapita bulan kalender dari latih x populasi bulan uji), bukan konsumsi aktual. Galat surplus dihitung terhadap surplus aktual.
- Conformal: proper train 0..54, kalibrasi 55..66, skor absolut |y-p|/sigma dengan sigma = prod_seasmean proper train (clip 1), Mondrian per komoditas, kuantil koreksi sampel hingga ceil((n+1)(1-alpha))/n, alpha = 0,20, metode kuantil 'higher'. Prediksi uji dari model proper-train yang sama tanpa refit. Model titik headline adalah LightGBM latih penuh, terpisah dari model conformal.
- Cakupan empiris conformal pada uji: produksi 79.3%, surplus 77.5% (nominal 80%). Lebar rata-rata: produksi 1,257.4 ton, surplus 1,884.5 ton.
- Kuantil Mondrian produksi (n kalibrasi per komoditas, q): Jagung n=420 q=0.166, Kacang Hijau n=420 q=0.130, Kacang Tanah n=420 q=0.095, Kedelai n=420 q=0.121, Padi n=420 q=0.111, Ubi Jalar n=420 q=0.143, Ubi Kayu n=420 q=0.096.
- Conformal surplus dikalibrasi LANGSUNG pada galat surplus versi proyeksi di jendela kalibrasi yang sama (55..66): skor = |surplus aktual - (ramalan produksi proper-train x rend proper-train - konsumsi proyeksi)| / sigma surplus seri, dengan sigma surplus = rata-rata |surplus| proper train per seri (clip bawah 1). Interval uji = ramalan surplus uji plus minus q_surplus[komoditas] x sigma surplus. Ini menggantikan penurunan dari interval produksi pada run perdana, yang mengabaikan galat proyeksi konsumsi sehingga cakupan surplus hanya 46,5%.
- Kuantil Mondrian surplus (n kalibrasi per komoditas, q): Jagung n=420 q=0.234, Kacang Hijau n=420 q=0.409, Kacang Tanah n=420 q=0.545, Kedelai n=420 q=0.233, Padi n=420 q=0.184, Ubi Jalar n=420 q=0.611, Ubi Kayu n=420 q=0.344.
- Uji Diebold-Mariano utama: HAC Newey-West lag 3 atas selisih galat absolut yang dirata-rata per bulan uji. Keterbatasan: T hanya 17 bulan, sehingga estimasi varians jangka panjang dan nilai p HAC bersifat kasar; kolom DM iid pooled versi lama disertakan hanya sebagai pembanding dan cenderung terlalu signifikan karena mengabaikan korelasi panel dan waktu.
- Rolling-origin: fold origin 54/60/66, uji 12 bulan, penyebut MASE dihitung dari data latih fold itu sendiri (bukan penyebut split utama seperti kode lama).

## Parameter MILP (apa adanya, dari kode)

- Tarif variabel: Rp 2,500 per ton-km atas jarak haversine antar sentroid.
- Biaya impor luar provinsi: Rp 2,500,000 per ton (dirumuskan 4 x tarif x jarak maks 250 km).
- Biaya tetap pembukaan rute: TIDAK ADA. Variabel biner rute hanya mengikat batas muatan minimum/maksimum, tidak masuk fungsi objektif.
- Batas jarak rute: 250 km. Muatan minimum adaptif min(20 ton, pasokan, kebutuhan). Permintaan wajib terpenuhi penuh (alpha = 1,0) dengan impor sebagai penutup tak berkapasitas. Model satu periode per komoditas-bulan, tanpa stok antar bulan. Ini MILP deterministik berbasis skenario P10, bukan robust optimization formal.
- Solver: HiGHS (paket highspy) lewat PuLP 4.2. Kode lama memakai CBC lewat API PuLP lama; pada VM ini PuLP 4.2 tidak menyediakan CBC/solvers bawaan (terverifikasi kosong), sehingga mesin solver diganti tanpa mengubah formulasi matematis. Optimum MIP tidak bergantung mesin solver.

## Catatan jalan

- Fallback rolling SARIMA (fit/append/forecast gagal): 0 kejadian. Exception pertama tercatat: tidak ada.
- Fallback rolling SARIMAX (fit/append/forecast gagal): 0 kejadian. Exception pertama tercatat: tidak ada. Catatan perbaikan: pada run perdana SARIMAX mencatat 221 fallback karena konstruktor statsmodels menolak exog ber-NaN (MissingDataError: exog contains inf or nans) akibat rain_lag1/ndvi_lag1 NaN pada baris pertama tiap seri; sudah diperbaiki dengan pengisian rata-rata kolom latih.
- Status MILP: 119 instans, semua Optimal: True. Total rute 1419.
