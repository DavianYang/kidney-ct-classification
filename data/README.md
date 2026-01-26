# CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone

This directory contains the Kidney CT image dataset used for the experiments
conducted in this thesis.

⚠️ **Note:** The dataset is not included in this repository due to size and
licensing restrictions. It must be downloaded separately.

---

## Dataset Source

- **Title:** CT Kidney Dataset: Normal, Cyst, Tumor, and Stone
- **Task:** Multi-class image classification
- **Classes:** Normal, Cyst, Tumor, Stone
- **Imaging Modality:** Computed Tomography (CT)
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone

All usage of the dataset follows the terms and conditions specified by the
original authors.

---

## Dataset Organization

The dataset is organized into class-specific directories:

```
CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
└── CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
    ├── Cyst
    ├── Normal
    ├── Stone
    ├── Tumor
    └── kidneyData.csv
```

Each directory contains CT images corresponding to its respective class label.
A metadata file (`kidneyData.csv`) provided with the dataset may be used for
label verification or dataset statistics.

---

## Dataset Splitting

The original dataset does not provide predefined training, validation, or test
splits. Therefore, data partitioning is performed programmatically during
experimentation.

- Splits are created using fixed random seeds
- Class distributions are preserved where applicable
- All split parameters are configurable via experiment configuration files

---

## Image Characteristics and Preprocessing

- **Format:** JPG
- **Channels:** Grayscale CT images
- **Resolution:** Varies across samples

Preprocessing steps applied during training include:

- Resizing to the model input resolution (e.g., 224 × 224)
- Intensity normalization
- Conversion to three channels when required by pretrained CNN models

All preprocessing is performed dynamically during data loading.

---

## Reproducibility

- Dataset splits are deterministic
- Random seeds are explicitly controlled
- All experiments can be reproduced using the provided configuration files

---

## Citation

If this dataset is used in academic work, please cite the dataset according to
the instructions provided on the Kaggle dataset page.
