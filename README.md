# SENTI-MATRIX: Multidimensional Sentiment Analysis with Generative Transformer Models
*Python · PyTorch · Tranformners · HuggingFace · LoRA*

**Description:**  
SENTI-MATRIX evaluates and fine-tunes transformer models across multiple sentiment analysis tasks — intent-based, aspect-based, fine-grained, and emotion detection, using pretrained models, base model fine-tuning, and LoRA adapters for parameter-efficient learning.

---

## 📖 Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---
## Overview
SENTI-MATRIX addresses challenges in multidimensional sentiment analysis by combining:

- **Pretrained Model Evaluation** — Benchmarking models pretrained on task-specific datasets.
- **Base Model Fine-Tuning** — Fine-tuning transformers on your datasets.
- **LoRA Adapter Fine-Tuning** — Applying LoRA for parameter-efficient adaptation.

---

## 🏆 Key Results

The table summarizes the performance of transformer-based models across multiple sentiment analysis tasks. LoRA adapters provide competitive accuracy while significantly reducing trainable parameters and GPU memory usage.

---

### **Intent-Based Sentiment Analysis (2-Class)**

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|---------|----------|------------|------------|
| SST-2 | 🟢 **Pretrained DistilBERT** | 98.13 | 98.20 | 66M | 2.0GB |
|  | 🔵 **Fine-tuned DistilBERT** | 89.09 | 89.51 | 66M | 2.0GB |
|  | 🟣 **LoRA Adapter DistilBERT** | 89.40 | 89.84 | 758K | 1.0GB |
| SST-2 | 🟢 **Pretrained RoBERTa** | 97.51 | 97.60 | 124M | 3.8GB |
|  | 🔵 **Fine-tuned RoBERTa** | 89.81 | 90.65 | 124M | 3.8GB |
|  | 🟣 **LoRA Adapter RoBERTa** | 92.00 | 92.21 | 1.2M | 1.2GB |
| IMDb | 🟢 **Pretrained BERT** | 95.62 | 95.53 | 109M | 2.7GB |
|  | 🔵 **Fine-tuned BERT** | 94.64 | 94.59 | 109M | 2.7GB |
|  | 🟣 **LoRA Adapter BERT** | 92.58 | 92.52 | 38K | 1.8GB |

*🟢 Pretrained, 🔵 Fully Fine-Tuned, 🟣 LoRA Adapter (parameter-efficient)*

---

### **Intent-Based Sentiment Analysis (3-Class)**

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|---------|----------|------------|------------|
| Twitter | 🟢 Pretrained RoBERTa | 64.96 | 62.88 | -- | -- |
|  | 🔵 Fine-tuned RoBERTa | 81.39 | 81.12 | 125M | 2.9GB |
|  | 🟣 LoRA Adapter BERT | 85.13 | 84.93 | 3.3M | 2.3GB |

---

### **Aspect-Based Sentiment Analysis (3-Class)**

| Domain | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|--------|-------|---------|----------|------------|------------|
| Combined (Laptop + Restaurant) | 🟢 Pretrained DeBERTa | 75.84 | 71.84 | -- | -- |
|  | 🔵 Fine-tuned DeBERTa | 78.04 | 66.74 | 184M | 3.5GB |
|  | 🟣 LoRA Adapter DeBERTa | 79.73 | 73.75 | 813K | 2.5GB |
| Laptop | 🟢 Pretrained DeBERTa | 81.90 | 80.71 | -- | -- |
|  | 🔵 Fine-tuned DeBERTa | 80.17 | 78.13 | 184M | 4.2GB |
|  | 🟣 LoRA Adapter DeBERTa | 78.02 | 69.84 | 813K | 2.5GB |
| Restaurant | 🟢 Pretrained DeBERTa | 73.41 | 68.35 | -- | -- |
|  | 🔵 Fine-tuned DeBERTa | 80.06 | 71.14 | 184M | 4.2GB |
|  | 🟣 LoRA Adapter DeBERTa | 79.02 | 68.28 | 813K | 2.4GB |

---

### **Fine-Grained Sentiment Analysis (5-Class / 3-Class)**

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|---------|----------|------------|------------|
| E-Commerce (5-Class) | 🟢 Pretrained BERT | 56.87 | 47.82 | -- | -- |
|  | 🔵 Fine-tuned BERT | 66.27 | 50.44 | 167M | 3.2GB |
|  | 🟣 LoRA Adapter BERT | 67.73 | 50.31 | 298K | 1.6GB |
| E-Commerce (3-Class) | 🟢 Pretrained RoBERTa | 79.12 | 53.51 | -- | -- |
|  | 🔵 Fine-tuned RoBERTa | 83.50 | 66.89 | 124M | 2.9GB |
|  | 🟣 LoRA Adapter RoBERTa | 84.96 | 66.47 | 889K | 1.7GB |
| Yelp (5-Class) | 🟢 Pretrained BERT | 55.85 | 55.46 | -- | -- |
|  | 🔵 Fine-tuned BERT | 61.13 | 60.44 | 16.7M | 3.7GB |
|  | 🟣 LoRA Adapter BERT | 64.58 | 64.09 | 1.3M | 3.5GB |
| Yelp (3-Class) | 🟢 Pretrained RoBERTa | 68.93 | 58.05 | -- | -- |
|  | 🔵 Fine-tuned RoBERTa | 82.01 | 77.54 | 124M | 4.2GB |
|  | 🟣 LoRA Adapter RoBERTa | 83.87 | 79.23 | 740K | 4.9GB |

---

### **Emotion Detection (6-Class)**

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|---------|----------|------------|------------|
| Emotion Dataset | 🟢 Pretrained DistilBERT | 93.15 | 89.87 | -- | -- |
|  | 🔵 Fine-tuned DistilBERT | 93.59 | 90.72 | 66.9M | 2.4GB |
|  | 🟣 LoRA Adapter DistilBERT | 93.40 | 90.61 | 668K | 2.1GB |

*💡 LoRA adapters provide a **balance of high accuracy and computational efficiency**, updating only a small fraction of total parameters while reducing GPU memory usage.*
