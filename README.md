# SentiMatrix: Parameter-Efficient Fine-Tuning of Encoder-Based Transformers for Multidimensional Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Python · PyTorch · Transformers · HuggingFace · LoRA · AdaLoRA*

> **This work is currently under review at the** ***Machine Learning*** **journal (Springer).**

**Keywords:** Sentiment Analysis · Transformer Models · Parameter-Efficient Fine-Tuning · Low-Rank Adaptation · Multidimensional Evaluation

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Pretrained Models](#pretrained-models)
- [Key Results](#key-results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage and Reproduction](#usage-and-reproduction)
- [PEFT Methods](#peft-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

---

## Overview

SentiMatrix evaluates transformer-based models under three adaptation strategies within a single unified pipeline across four sentiment analysis paradigms:

| Paradigm | Task | Classes | Datasets |
|:---------|:-----|:-------:|:---------|
| Intent-Based | Binary polarity | 2 | SST-2, IMDb |
| Intent-Based | Multi-class polarity | 3 | Twitter US Airline |
| Aspect-Based | Aspect-level polarity | 3 | SemEval-2014 (Laptop + Restaurant) |
| Fine-Grained | Rating-scale sentiment | 3 & 5 | Amazon E-Commerce, Yelp Reviews |
| Emotion Detection | Discrete affective states | 6 | CARER Emotion |

The four adaptation conditions evaluated are:

1. **Pretrained Baseline Inference** — Task-specific HuggingFace checkpoints previously fine-tuned on the corresponding benchmarks, serving as informed upper-bound references rather than cold-start zero-shot systems.
2. **Full Fine-Tuning (FFT)** — All model parameters updated; serves as the performance upper bound.
3. **LoRA (Proposed)** — Low-rank adapter matrices injected into attention layers; only adapter weights trained.
4. **AdaLoRA (Comparator)** — SVD-based importance-aware rank allocation; benchmarked against LoRA.

---

## Datasets

All datasets are pre-processed and stored in the `Dataset/` directory. An 80%/10%/10% train/validation/test split is applied consistently across all tasks.

| Paradigm | Dataset | Classes | Total | Train (80%) | Val (10%) | Test (10%) |
|:---------|:--------|:-------:|------:|------------:|----------:|-----------:|
| Intent-Based | SST-2 | 2 | 9,613 | 7,690 | 961 | 962 |
| Intent-Based | IMDb Movie Reviews | 2 | 50,000 | 40,000 | 5,000 | 5,000 |
| Intent-Based | Twitter US Airline | 3 | 61,949 | 49,559 | 6,195 | 6,195 |
| Aspect-Based | SemEval-2014 ABSA | 3 | 5,914 | 4,731 | 591 | 592 |
| Fine-Grained | Amazon E-Commerce | 5 | 22,641 | 18,112 | 2,264 | 2,265 |
| Fine-Grained | Yelp Reviews | 5 | 85,000 | 68,000 | 8,500 | 8,500 |
| Emotion Detection | CARER Emotion | 6 | 75,000 | 60,000 | 7,500 | 7,500 |

**Dataset notes:**
- SST-2: Sentence-level movie review annotations (Stanford).
- IMDb: Document-level long-form reviews requiring discourse-level understanding. Many reviews exceed the 512-token limit, leading to truncation during preprocessing.
- Twitter US Airline: Informal social media text with Positive / Neutral / Negative labels.
- SemEval-2014 ABSA: Aspect-level annotations across Laptop and Restaurant domains.
- Amazon E-Commerce / Yelp: 5-star rating scales; also evaluated under collapsed 3-class schema (1–2 = Negative, 3 = Neutral, 4–5 = Positive).
- CARER: Six discrete affective states — *sadness, joy, love, anger, fear, surprise*.

---

## Pretrained Models

Model selection follows architectural alignment with task-specific characteristics (text length, register, label granularity). All pretrained models are task-specific HuggingFace checkpoints serving as informed upper-bound references.

| Dataset | Model Architecture | Base Version | Parameters | Layers | Heads |
|:--------|:-------------------|:-------------|:----------:|:------:|:-----:|
| SST-2 | DistilBERT | base-uncased | 66M | 6 | 12 |
| SST-2 | RoBERTa | base | 125M | 12 | 12 |
| IMDb | BERT | base-uncased | 110M | 12 | 12 |
| Twitter | RoBERTa | base-sentiment-latest | 125M | 12 | 12 |
| SemEval-2014 ABSA | DeBERTa-v3 | base | 184M | 12 | 12 |
| E-Commerce / Yelp | BERT-Multilingual | base-uncased | 110M | 12 | 12 |
| CARER Emotion | DistilBERT | base-uncased | 66.9M | 6 | 12 |

---

## Key Results

*Pretrained Baseline = task-specific pretrained checkpoint, no target-dataset training | FFT = full parameter update | LoRA = Low-Rank Adapter | AdaLoRA = Adaptive Low-Rank Adapter*

All efficiency metrics correspond to training-time efficiency. Inference-time efficiency depends on deployment configuration and is not directly measured in this study.

Best result per dataset in **bold**.

---

### Intent-Based Sentiment Analysis (2-Class)

| Dataset | Model | Accuracy | macro-F1 | Parameters | GPU Memory |
|:--------|:------|:--------:|:--------:|:----------:|:----------:|
| SST-2 | Pretrained Baseline (DistilBERT) | **98.13%** | **98.20%** | — | — |
| | FFT (DistilBERT) | 89.09% | 89.51% | 66M | 2.0 GB |
| | LoRA-DistilBERT | 89.40% | 89.84% | 758K | 1.0 GB |
| | AdaLoRA-DistilBERT | 88.77% | 88.89% | 924K | 1.2 GB |
| SST-2 | Pretrained Baseline (RoBERTa) | **97.51%** | **97.60%** | — | — |
| | FFT (RoBERTa) | 89.81% | 90.65% | 124M | 3.8 GB |
| | LoRA-RoBERTa | 92.00% | 92.21% | 1.2M | 1.2 GB |
| | AdaLoRA-RoBERTa | 92.72% | 92.90% | 1.8M | 1.2 GB |
| IMDb | Pretrained Baseline (BERT) | **95.62%** | **95.53%** | — | — |
| | FFT (BERT) | 94.64% | 94.59% | 110M | 2.7 GB |
| | LoRA-BERT | 92.58% | 92.52% | 38K | 1.8 GB |
| | AdaLoRA-BERT | 91.38% | 91.41% | 444K | 1.2 GB |

---

### Intent-Based Sentiment Analysis (3-Class)

| Dataset | Model | Accuracy | macro-F1 | Parameters | GPU Memory |
|:--------|:------|:--------:|:--------:|:----------:|:----------:|
| Twitter | Pretrained Baseline (RoBERTa) | 64.96% | 62.88% | — | — |
| | FFT (RoBERTa) | 81.39% | 81.12% | 125M | 2.9 GB |
| | LoRA-RoBERTa | **85.13%** | **84.93%** | 3.3M | 2.3 GB |
| | AdaLoRA-RoBERTa | 75.58% | 75.41% | 1.0M | 1.1 GB |

---

### Aspect-Based Sentiment Analysis (3-Class)

| Dataset | Model | Accuracy | macro-F1 | Parameters | GPU Memory |
|:--------|:------|:--------:|:--------:|:----------:|:----------:|
| Combined (Laptop + Restaurant) | Pretrained Baseline (DeBERTa-v3) | 75.84% | 71.84% | — | — |
| | FFT (DeBERTa-v3) | 78.04% | 66.74% | 184M | 3.5 GB |
| | LoRA-DeBERTa | **79.73%** | **73.75%** | 813K | 2.5 GB |
| | AdaLoRA-DeBERTa | 76.35% | 66.72% | 1.2M | 1.7 GB |
| Laptop | Pretrained Baseline (DeBERTa-v3) | **81.90%** | **80.71%** | — | — |
| | FFT (DeBERTa-v3) | 80.17% | 78.13% | 184M | 4.2 GB |
| | LoRA-DeBERTa | 78.02% | 69.84% | 813K | 2.5 GB |
| Restaurant | Pretrained Baseline (DeBERTa-v3) | 73.41% | 68.35% | — | — |
| | FFT (DeBERTa-v3) | **80.06%** | **71.14%** | 184M | 4.2 GB |
| | LoRA-DeBERTa | 79.02% | 68.28% | 813K | 2.4 GB |

> AdaLoRA results are reported on the combined Laptop + Restaurant domain only.

---

### Fine-Grained Sentiment Analysis (5-Class / 3-Class)

| Dataset | Model | Accuracy | macro-F1 | Parameters | GPU Memory |
|:--------|:------|:--------:|:--------:|:----------:|:----------:|
| E-Commerce (5-Class) | Pretrained Baseline (BERT) | 56.87% | 47.82% | — | — |
| | FFT (BERT) | 66.27% | 50.44% | 167M | 3.2 GB |
| | LoRA-BERT | **67.73%** | 50.31% | 298K | 1.6 GB |
| | AdaLoRA-mBERT | 65.30% | 42.01% | 446K | 1.4 GB |
| E-Commerce (3-Class) | Pretrained Baseline (RoBERTa) | 79.12% | 53.51% | — | — |
| | FFT (RoBERTa) | 83.50% | **66.89%** | 124M | 2.9 GB |
| | LoRA-RoBERTa | **84.96%** | 66.53% | 889K | 1.7 GB |
| | AdaLoRA-RoBERTa | 84.87% | 66.53% | 1.3M | 3.3 GB |
| Yelp (5-Class) | Pretrained Baseline (BERT) | 55.85% | 55.46% | — | — |
| | FFT (BERT) | 61.13% | 60.44% | 167M | 3.7 GB |
| | LoRA-BERT | **64.58%** | **64.09%** | 1.3M | 3.5 GB |
| | AdaLoRA-mBERT | 47.53% | 47.75% | 446K | 1.4 GB |
| Yelp (3-Class) | Pretrained Baseline (RoBERTa) | 69.01% | 58.29% | — | — |
| | FFT (RoBERTa) | 77.79% | **73.47%** | 124M | 4.2 GB |
| | LoRA-RoBERTa | **77.99%** | 71.27% | 740K | 4.9 GB |
| | AdaLoRA-RoBERTa | 71.66% | 68.48% | 1.3M | 3.3 GB |

> Collapsing 5-class to 3-class labels yields approximately 17% accuracy improvement at no additional computational cost.

---

### Emotion Detection (6-Class)

| Dataset | Model | Accuracy | macro-F1 | Parameters | GPU Memory |
|:--------|:------|:--------:|:--------:|:----------:|:----------:|
| CARER Emotion | Pretrained Baseline (DistilBERT) | 93.15% | 89.87% | — | — |
| | FFT (DistilBERT) | **93.59%** | **90.72%** | 66.9M | 2.4 GB |
| | LoRA-DistilBERT | 93.40% | 90.61% | 668K | 2.1 GB |
| | AdaLoRA-DistilBERT | 93.36% | 93.46% | 927K | 0.5 GB |

---

### SentiMatrix — Proposed Model Summary and State-of-the-Art Comparison

**SentiMatrix** is the proposed LoRA-based framework. The table below reports SentiMatrix results alongside efficiency gains relative to full fine-tuning across all benchmarks.

| Benchmark | Task | SentiMatrix Model | Accuracy | macro-F1 | Params | Param Reduction | SotA? |
|:----------|:-----|:------------------|:--------:|:--------:|:------:|:---------------:|:-----:|
| SST-2 | Intent (2-class) | LoRA-DistilBERT | 89.40% | 89.84% | 758K | −98.9% | — |
| SST-2 | Intent (2-class) | LoRA-RoBERTa | 92.00% | 92.21% | 1.2M | −99.0% | — |
| IMDb | Intent (2-class) | LoRA-BERT | 92.58% | 92.52% | 38K | −99.9% | — |
| Twitter | Intent (3-class) | LoRA-RoBERTa | **85.13%** | **84.93%** | 3.3M | −97.4% | ✓ New SotA |
| ABSA Combined | Aspect (3-class) | LoRA-DeBERTa | 79.73% | 73.75% | 813K | −99.6% | — |
| E-Commerce | Fine-Grained (5-class) | LoRA-BERT | 67.73% | 50.31% | 298K | −99.8% | — |
| Yelp | Fine-Grained (5-class) | LoRA-BERT | **64.58%** | **64.09%** | 1.3M | −92.2% | ✓ New SotA |
| CARER | Emotion (6-class) | LoRA-DistilBERT | 93.40% | 90.61% | 668K | −99.0% | ≈ SotA |

*SentiMatrix reduces trainable parameters by up to 99.7%, training time by up to 70%, and peak GPU memory by 15–30% relative to full fine-tuning, while matching or surpassing FFT on 5 out of 7 benchmarks. All efficiency figures reflect training-time metrics only.*

---

## Key Findings

- **LoRA sets new state-of-the-art** on Twitter 3-class (85.13%) and Yelp 5-class (64.58%), and approaches state-of-the-art on CARER emotion detection (93.40%), without task-specific modifications.
- **LoRA outperforms AdaLoRA** on most benchmarks across accuracy, training efficiency, and GPU memory; AdaLoRA shows complementary strengths only on precision-sensitive tasks such as emotion detection.
- **LoRA reduces trainable parameters by up to 99.7%**, training time by up to 70%, and peak GPU memory by 15–30% relative to FFT, with gains consistent across model sizes from 66M to 184M parameters.
- **LoRA's fixed low-rank constraint acts as an effective regularizer** under domain shift (Twitter informal text), label noise, and high label granularity (5-class), confirmed by ablation experiments.
- **FFT retains an advantage only on long-document tasks** (IMDb), where full parameter updates better capture long-range discourse structure truncated by the 512-token limit.
- **Label granularity reduction** (5-class → 3-class) yields approximately 17% accuracy improvement at no additional computational cost.
- **Pretrained baseline models remain competitive** on binary tasks with well-matched pretraining objectives (SST-2: 97–98%), but degrade substantially on domain-shifted, multi-class, and fine-grained tasks. These baselines serve as informed upper-bound references, not cold-start zero-shot systems.
- **All efficiency comparisons reflect training-time metrics only.** Inference-time latency and memory depend on deployment configuration, such as whether LoRA adapters are merged into the base model prior to serving.

---

## Repository Structure

```
SENTI-MATRIX/
│
├── Code/
│   ├── Intent Based Sentiment (2 classes) SST-2 Dataset - Models.ipynb
│   ├── Intent Based Sentiment (2 classes) SST-2 Dataset - Models - AdaLoRA.ipynb
│   ├── Intent Based Sentiment (2 classes) IMDB Dataset.ipynb
│   ├── Intent Based Sentiment (2 classes) IMDB Dataset - AdaLoRA.ipynb
│   ├── Aspect Based Sentiment (3 classes) Laptop + Restaurant ABSA Dataset.ipynb
│   ├── Aspect Based Sentiment (3 classes) Laptop + Restaurant ABSA Dataset - AdaLoRA.ipynb
│   ├── Aspect Based Sentiment (3 classes) Laptop ABSA Dataset.ipynb
│   ├── Aspect Based Sentiment (3 classes) Restaurant ABSA Dataset.ipynb
│   ├── Fine-Grained Sentiment Analysis (3 classes) E-Commerce Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (3 classes) E-Commerce Reviews - AdaLoRA.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) E-Commerce Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) E-Commerce Reviews - AdaLoRA.ipynb
│   ├── Fine-Grained Sentiment Analysis (3 classes) Yelp Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (3 classes) Yelp Reviews - AdaLoRA.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) Yelp Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) Yelp Reviews - AdaLoRA.ipynb
│   ├── Emotion Detection Emotion Dataset.ipynb
│   └── Emotion Detection Emotion Dataset - AdaLoRA.ipynb
│
├── Dataset/
│   ├── E-Commerce Reviews (5 Classes).csv
│   ├── E-Commerce Reviews (3 Classes).csv
│   ├── Emotion Dataset (Small).csv
│   ├── Laptop + Restaurant - ABSA.csv
│   ├── Laptop - ABSA.csv
│   ├── Restaurant - ABSA.csv
│   ├── Twitter Dataset (3 Classes).csv
│   ├── Yelp Reviews (3 Classes).csv
│   ├── Yelp Reviews (5 Classes).csv
│   └── IMDB Dataset (2 Classes).csv
│
├── LICENSE
└── README.md
```

Each notebook in `Code/` is self-contained and covers one task–dataset combination. Notebooks without an `-AdaLoRA` suffix run the pretrained baseline, full fine-tuning, and LoRA experiments. Notebooks with `-AdaLoRA` run the AdaLoRA variant for the same task and dataset.

---

## Installation

### System Requirements

- Python 3.10 or higher
- NVIDIA GPU with ≥8 GB VRAM (16 GB recommended for DeBERTa / large-batch training)
- CUDA-enabled environment
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/ELTE-DSED/senti-matrix.git
cd senti-matrix
```

### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install torch torchvision torchaudio
pip install transformers datasets peft evaluate
pip install pandas scikit-learn matplotlib tqdm
pip install jupyter notebook
```

---

## Usage and Reproduction

All experiments are implemented as self-contained Jupyter Notebooks in `Code/`. Each notebook follows the four-condition pipeline:

1. **Pretrained Baseline Evaluation** — Load task-specific pretrained model from Hugging Face Hub; evaluate without target-dataset training.
2. **Full Fine-Tuning (FFT)** — Update all model parameters on the target dataset.
3. **LoRA Adapter Training** — Train only injected low-rank adapter matrices; base weights frozen.
4. **AdaLoRA Adapter Training** — SVD-based importance-aware rank allocation; run via `-AdaLoRA` notebooks.

### Running a Notebook

```bash
jupyter notebook "Code/Emotion Detection Emotion Dataset.ipynb"
```

Run all cells in order. Each notebook loads its dataset from `Dataset/`, trains the model, and reports accuracy, macro-F1, parameter count, training time, and peak GPU memory.

### Training Configuration

All experiments use AdamW optimization with:
- Learning rate: 1×10⁻⁵ to 2×10⁻⁵ (FFT); higher for LoRA/AdaLoRA
- 80%/10%/10% train/validation/test split
- Early stopping on validation macro-F1 or loss depending on task
- Mixed-precision training (`fp16`)
- Dropout and weight decay for regularization

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

### AdaLoRA Configuration

```python
from peft import AdaLoraConfig, get_peft_model

adalora_config = AdaLoraConfig(
    task_type="SEQ_CLS",
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, adalora_config)
model.print_trainable_parameters()
```

---

## PEFT Methods

### LoRA — Low-Rank Adaptation

LoRA freezes the pretrained weight matrix **W₀ ∈ ℝ^(d×k)** and constrains weight updates to a low-rank decomposition:

> **W = W₀ + ΔW = W₀ + BA**

where **B ∈ ℝ^(d×r)**, **A ∈ ℝ^(r×k)**, and rank *r ≪ min(d, k)*. Only **B** and **A** are trained, reducing trainable parameters from *d×k* to *r(d+k)*. LoRA adapters are applied to the query and value projection matrices of all multi-head attention layers across BERT, RoBERTa, DistilBERT, and DeBERTa.

### AdaLoRA — Adaptive Low-Rank Adaptation

AdaLoRA parameterizes the weight update via Singular Value Decomposition (SVD):

> **W = W₀ + ΔW = W₀ + PΛQ**

where **P ∈ ℝ^(d×r)** and **Q ∈ ℝ^(r×k)** are singular vectors and **Λ = diag(λ₁, …, λᵣ)** contains learnable singular values. During training, singular values are pruned iteratively based on importance scores, concentrating the parameter budget on the most informative weight matrices. The same attention layers are targeted as in LoRA for a controlled like-for-like comparison.

### Comparison

| Feature | Full Fine-Tuning | LoRA | AdaLoRA |
|:--------|:----------------:|:----:|:-------:|
| Trainable Parameters | 100% (66M–184M) | < 3% (38K–3.3M) | < 2% (446K–1.8M) |
| GPU Memory | High | Low (−15–30%) | Low (−10–25%) |
| Training Time | Baseline | Up to −70% | Moderate reduction |
| Rank Allocation | N/A | Fixed | Adaptive (SVD importance) |
| Storage per Task | ~500 MB+ | ~5–10 MB | ~5–10 MB |
| Best Overall Performance | — | 5/7 benchmarks | 2/7 benchmarks |

---

## Evaluation Metrics

### Performance Metrics

| Metric | Description |
|:-------|:------------|
| Accuracy | Overall percentage of correct predictions. |
| Precision | Proportion of positive identifications that were correct (macro-averaged). |
| Recall | Proportion of actual positives correctly identified (macro-averaged). |
| macro-F1 | Harmonic mean of Precision and Recall, macro-averaged across all classes with equal weight per class. |
| Confidence Score | Mean softmax probability assigned to the ground-truth class over all test instances. Captures predictive calibration beyond hard-decision accuracy. |
| Similarity Score | Cosine similarity between the model's predicted probability distribution and the one-hot representation of the ground-truth class. Measures prediction alignment with the target class, incorporating the entire probability distribution rather than only the maximum predicted probability. Not reported for AdaLoRA due to anomalous negative values produced by SVD-based adaptive rank pruning. |

### Efficiency Metrics

| Metric | Description |
|:-------|:------------|
| Trainable Parameters | Number of model parameters updated during training (M = millions, K = thousands). |
| Training Time | Total wall-clock time in seconds from initiation to convergence, measured on identical hardware. All efficiency figures are training-time only; inference-time efficiency depends on deployment configuration. |
| Peak GPU Memory | Maximum VRAM consumption in GB during training (parameters + optimizer states + gradients + activations). |

---

## Citation

If you use SentiMatrix in your research, please cite:

```bibtex
@article{arafat2026sentimatrix,
  title     = {SentiMatrix: Parameter-Efficient Fine-Tuning of Encoder-Based Transformers for Multidimensional Sentiment Analysis},
  author    = {Arafat, Md. Easin and
               Akmal, Muhammad Usman and
               Abosinnee, Ali S. and
               Orosz, Tam{\'a}s},
  journal   = {Machine Learning},
  publisher = {Springer},
  year      = {2026},
  note      = {Under review}
}
```

---

## Acknowledgements

The authors gratefully acknowledge the support of the Faculty of Excellence fellowship programme, Faculty of Informatics, Eötvös Loránd University, Budapest, Hungary (Job no. E19019/19). The authors also acknowledge the open-source communities behind the HuggingFace Transformers library, PyTorch, and the benchmark dataset providers whose resources made this research possible.

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Md. Easin Arafat, Muhammad Usman Akmal, Ali S. Abosinnee, Tamás Orosz

---

## Contact

**Md. Easin Arafat** *(Corresponding Author)*
Doctoral Fellow, Department of Data Science and Engineering
Faculty of Informatics, Eötvös Loránd University (ELTE), Budapest, Hungary
arafatmdeasin@inf.elte.hu

**Muhammad Usman Akmal**
Researcher, Department of Data Science and Engineering
Faculty of Informatics, Eötvös Loránd University (ELTE), Budapest, Hungary
ikhjls@inf.elte.hu

**Ali S. Abosinnee**
Department of Data Science and Engineering, Faculty of Informatics, Eötvös Loránd University (ELTE), Budapest, Hungary
Department of Computer Technical Engineering, College of Technical Engineering, The Islamic University, Najaf, Iraq
abosinnee.ali@inf.elte.hu

**Tamás Orosz**
Associate Professor, Department of Data Science and Engineering
Faculty of Informatics, Eötvös Loránd University (ELTE), Budapest, Hungary
orosztamas@inf.elte.hu

---

*Department of Data Science and Engineering, Eötvös Loránd University (ELTE), Budapest, Hungary*
