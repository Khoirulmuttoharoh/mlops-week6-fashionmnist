# Praktikum MLOps Minggu 6 — Model Experimentation & Tracking

**Tujuan:** membangun alur eksperimen end-to-end dengan MLflow, mengikuti delapan butir tugas pada slide.

## Stack dan dataset

- Python + PyTorch + torchvision
- Fashion-MNIST: 60.000 citra latih dan 10.000 citra uji, grayscale 28×28, 10 kelas
- MLflow Tracking dengan backend SQLite dan penyimpanan artefak lokal
- Optuna untuk minimal 20 trial
- DVC untuk versioning dataset
- Git untuk versioning kode

> Catatan evaluasi: script tuning yang disediakan mencatat metrik test agar contoh praktikum mudah dijalankan. Untuk eksperimen ilmiah yang ketat, gunakan validation split untuk memilih hyperparameter dan buka test set hanya sekali pada evaluasi final. Jika dosen meminta pemisahan train/validation/test yang ketat, sesuaikan script sebelum menjalankan eksperimen final.

## 0. Persiapan

Gunakan Python 3.10 atau 3.11 untuk kompatibilitas library yang lebih luas. Jalankan perintah dari folder proyek.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika PowerShell memblokir aktivasi, gunakan Command Prompt:
```bat
.venv\Scripts\activate.bat
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Inisialisasi Git:
```bash
git init
git add .
git commit -m "Initial MLOps experiment project"
```

## 1. Jalankan MLflow Tracking Server lokal

Buka terminal pertama dan jalankan:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Biarkan terminal ini tetap berjalan. Buka `http://127.0.0.1:5000` di browser. Buka terminal kedua di folder proyek dan atur URI tracking.

PowerShell:
```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

Linux/macOS:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Server lokal menggunakan SQLite untuk metadata dan `./mlartifacts` untuk artefak. Jangan commit database atau artefak ke Git.

## 2. Unduh dataset dan catat versi

Dataset diunduh otomatis saat menjalankan training pertama. Untuk mengunduh lebih dahulu:

```bash
python -c "from src.common import get_datasets,dataset_hash; get_datasets(); print('SHA256:', dataset_hash())"
```

Catat SHA-256 yang tampil. Untuk versioning dengan DVC:

```bash
dvc init   # jalankan satu kali saja per repositori
dvc add data/fashion_mnist
git add data/fashion_mnist.dvc .gitignore .dvc/config
git commit -m "Version Fashion-MNIST dataset with DVC"
```

Folder data akan diabaikan oleh Git, sedangkan file `.dvc` menyimpan pointer/hash. Untuk penyimpanan bersama, konfigurasi remote DVC (misalnya storage kampus/S3) sesuai fasilitas yang tersedia; jangan menulis kredensial ke Git.

## 3. Latih baseline CNN

```bash
python -m src.experiment train --name baseline-cnn --seed 42 --epochs 5 --batch-size 128 --lr 0.001 --dropout 0.25
```

Buka MLflow UI → eksperimen `fashion-mnist-cnn` (jika belum muncul, script akan membuat eksperimen default). Untuk melihat run lebih rapi, script dapat diubah menambahkan `mlflow.set_experiment("fashion-mnist-cnn")` sebelum `start_run`.

Setiap run mencatat parameter, seed, hash dataset, device, hipotesis, waktu training, metrik per epoch (`train_loss`, `test_accuracy`, `test_macro_f1`), JSON ringkasan, dan model PyTorch.

**Hipotesis baseline:** CNN kecil mampu belajar pola dasar Fashion-MNIST; hasil baseline menjadi acuan untuk tuning dan ablation.

## 4. Hyperparameter tuning — minimal 20 trial

```bash
python -m src.tune
```

Optuna menjalankan 20 trial; setiap trial memiliki konfigurasi dan MLflow run. Parameter yang dicari: learning rate, batch size, dropout, optimizer, dan augmentation. Karena setiap trial melatih CNN beberapa epoch, lama eksekusi bergantung pada CPU/GPU.

Di MLflow UI, pilih run `optuna-20-trials` dan bandingkan nested runs. Catat trial terbaik serta parameter dan macro-F1-nya. Untuk pelaporan final, salin parameter terbaik dari output terminal ke `src/evaluate_seeds.py` dan `src/ablation.py`.

## 5. Evaluasi konfigurasi terbaik dengan 3 seed

Edit dictionary `best` pada `src/evaluate_seeds.py` memakai parameter terbaik dari langkah 4, lalu jalankan:

```bash
python -m src.evaluate_seeds --epochs 5
```

Script menggunakan seed `17`, `42`, dan `2026`, mencatat metrik setiap run, kemudian menghitung mean dan sample standard deviation:

\[
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i,\qquad
s = \sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}}
\]

Laporkan sebagai `mean ± std` untuk accuracy dan macro-F1. Jangan mengisi laporan dengan angka contoh; gunakan angka hasil run Anda sendiri.

## 6. Ablation study pada minimal dua komponen

Edit `best_config` di `src/ablation.py` dengan konfigurasi terbaik, lalu:

```bash
python -m src.ablation
```

Eksperimen membandingkan:
1. `ablation-full`: konfigurasi lengkap;
2. `ablation-no-augmentation`: augmentasi dimatikan;
3. `ablation-no-dropout`: dropout dihilangkan.

Variabel lain dijaga tetap sama. Bandingkan `final_test_macro_f1`, `final_test_accuracy`, dan waktu training. Kesimpulan harus berdasarkan hasil aktual; jangan berasumsi augmentasi/dropout pasti meningkatkan metrik.

## 7. Registrasikan model terbaik, beri alias `@champion`, lalu prediksi

Di MLflow UI, pilih run terbaik yang memiliki artefak model dan salin **Run ID**. Jalankan:

```bash
python -m src.register_model --run-id ISI_RUN_ID_DI_SINI
python -m src.predict_champion
```

Script mendaftarkan model sebagai `fashion-mnist-cnn`, memasang alias `champion` ke versi terdaftar, lalu memuat model dengan URI:

```text
models:/fashion-mnist-cnn@champion
```

Pastikan run ID yang dipilih benar-benar memiliki artefak `model`. Untuk praktik yang lebih kuat, pilih kandidat berdasarkan validation metric dan periksa hasil final sebelum menjadikannya champion.

## 8. Versioning dataset dengan DVC dan hash per run

Pastikan langkah 2 telah dijalankan. Pada setiap run, kode menghitung SHA-256 berkas dataset dan menyimpannya sebagai tag `dataset_sha256` serta parameter `dataset_hash` di MLflow. Dengan demikian, run dapat ditautkan ke versi data yang digunakan.

Periksa:
```bash
dvc status
git status
```

Jika menggunakan DVC remote yang telah dikonfigurasi:
```bash
dvc push
```

## Deliverable yang dikumpulkan

1. Repositori Git dengan source code, `requirements.txt`, `params.yaml`, README, dan pointer DVC.
2. Screenshot MLflow UI:
   - daftar eksperimen/run;
   - metrik per epoch atau chart;
   - compare view baseline vs trial;
   - hasil tiga seed;
   - ablation study;
   - model registry dan alias `champion`.
3. Laporan singkat berisi:
   - hipotesis tiap tahap;
   - konfigurasi baseline dan parameter terbaik;
   - tabel mean ± std dari tiga seed;
   - tabel ablation;
   - kesimpulan berdasarkan hasil aktual;
   - keterbatasan eksperimen dan perangkat yang digunakan.

### Template tabel laporan

| Eksperimen | Perubahan utama | Accuracy | Macro-F1 | Waktu (detik) |
|---|---|---:|---:|---:|
| Baseline | CNN default | isi hasil | isi hasil | isi hasil |
| Best tuning | Parameter terbaik | isi hasil | isi hasil | isi hasil |
| No augmentation | Augmentasi mati | isi hasil | isi hasil | isi hasil |
| No dropout | Dropout = 0 | isi hasil | isi hasil | isi hasil |

### Template kesimpulan

“Eksperimen dilakukan menggunakan Fashion-MNIST dan CNN sederhana. Baseline memperoleh [isi hasil]. Dari 20 trial Optuna, konfigurasi terbaik menggunakan [isi parameter] dengan macro-F1 [isi hasil]. Pengujian dengan tiga seed menghasilkan macro-F1 sebesar [mean] ± [std]. Pada ablation study, penghilangan [komponen] mengubah macro-F1 sebesar [selisih] dibanding konfigurasi lengkap. Model yang dipilih kemudian diregistrasikan pada MLflow dan dipanggil melalui alias `@champion`. Hasil ini berlaku pada konfigurasi dan lingkungan eksperimen yang digunakan.”

## Catatan teknis penting

- Training model CNN di CPU mungkin memerlukan waktu; bila tersedia, gunakan GPU.
- Jangan mengklaim hasil eksperimen sebelum seluruh run selesai.
- Versi PyTorch/torchvision harus kompatibel dengan sistem. Jika instalasi gagal, ikuti instruksi instalasi resmi PyTorch untuk OS dan akselerator yang digunakan.
- Script contoh menyederhanakan struktur praktikum. Untuk evaluasi final yang bebas test leakage, sebaiknya pisahkan validation set, lakukan tuning hanya pada validation/CV, dan evaluasi test sekali pada model final.
