# Deteksi Tumor Otak Menggunakan CNN + Transfer Learning (PyTorch)

Proyek ini bertujuan membangun model deep learning untuk **klasifikasi citra MRI otak** menjadi 2 kelas:

- `No Tumor` 🧠
- `Yes Tumor` 🧠❌

Model yang digunakan adalah **ResNet18 (transfer learning)** dari pustaka `torchvision.models`.

---

## Dataset

Dataset yang digunakan dalam proyek ini berasal dari Kaggle:

👉 [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data)

Silakan unduh dataset dari link tersebut, lalu ekstrak ke folder:

```
/content/brain_tumor_dataset/
```

Struktur folder setelah ekstrak:

```
brain_tumor_dataset/
├── no/
├── yes/
```

Dataset ini berisi gambar MRI otak dengan dua kelas:
- **no** : Tidak terdapat tumor.
- **yes** : Terdapat tumor.

---

## Langkah-langkah

### 1️⃣ Setup Environment

- Platform: **Google Colab**
- Library yang digunakan:
  - `torch`
  - `torchvision`
  - `sklearn`
  - `matplotlib`
  - `numpy`

### 2️⃣ Split Dataset

Dataset awal di-split menjadi:

- `train` (80%)
- `val` (20%)

Struktur setelah split:

```
brain_tumor_dataset_train/
├── no/
└── yes/

brain_tumor_dataset_val/
├── no/
└── yes/
```

### 3️⃣ Data Augmentation dan Transformasi

Transformasi yang diterapkan:

- Resize ke **224x224** (ukuran standar untuk ResNet)
- Augmentasi **RandomHorizontalFlip** (hanya di train)
- Normalisasi sesuai mean/std imagenet

### 4️⃣ Model

Model yang digunakan:

- **ResNet18 pretrained**
- Fully-connected layer terakhir diubah menjadi **Linear(num_features, 2)**

### 5️⃣ Training

- Optimizer: `Adam` (lr = 0.001)
- Loss: `CrossEntropyLoss`
- Epochs: **5**
- Batch size: **32**

Training dilakukan selama beberapa epoch dengan **monitoring akurasi dan loss** untuk `train` dan `val`.

### 6️⃣ Visualisasi Prediksi
![download (8)](https://github.com/user-attachments/assets/f2abe4c0-bdcb-4fbf-928a-8d1371b6c2f5)

---

## Cara Menjalankan Proyek

1️⃣ Upload dataset ke Google Colab (folder `brain_tumor_dataset`)

2️⃣ Jalankan step by step notebook / script berikut:

### Bagian 1: Import Library dan Setup Device
```python
# ... (lihat bagian di notebook) ...
```

### Bagian 2,3,4....7 seterusnya lihat di folder terlampir
