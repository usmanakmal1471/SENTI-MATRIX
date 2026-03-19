# SENTI-MATRIX: Multidimensional Sentiment Analysis with Generative Transformer Models
*Python · PyTorch · Tranformners · HuggingFace · LoRA*

**Description:**  
SENTI-MATRIX evaluates and fine-tunes transformer models across multiple sentiment analysis tasks including intent based, aspect based, fine grained, and emotion detection using pretrained models, base model fine tuning, and LoRA adapters for parameter efficient learning.

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
## 📖 Overview
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
|---------|-------|----------|----------|------------|------------|
| SST-2   | 🟢 **Pretrained DistilBERT** | 98.13 | 98.20 | -- | -- |
|         | 🔵 **Fine-tuned DistilBERT** | 89.09 | 89.51 | 66M | 2.0GB |
|         | 🟣 **LoRA Adapter DistilBERT** | 89.40 | 89.84 | 758K | 1.0GB |
| SST-2   | 🟢 **Pretrained RoBERTa** | 97.51 | 97.60 | -- | -- |
|         | 🔵 **Fine-tuned RoBERTa** | 89.81 | 90.65 | 124M | 3.8GB |
|         | 🟣 **LoRA Adapter RoBERTa** | 92.00 | 92.21 | 1.2M | 1.2GB |
| IMDb    | 🟢 **Pretrained BERT** | 95.62 | 95.53 | -- | -- |
|         | 🔵 **Fine-tuned BERT** | 94.64 | 94.59 | 109M | 2.7GB |
|         | 🟣 **LoRA Adapter BERT** | 92.58 | 92.52 | 38K | 1.8GB |

*🟢 Pretrained, 🔵 Fully Fine-Tuned, 🟣 LoRA Adapter (parameter-efficient)*

---

### **Intent-Based Sentiment Analysis (3-Class)**

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|----------|----------|------------|------------|
| Twitter | 🟢 Pretrained RoBERTa | 64.96 | 62.88 | -- | -- |
|         | 🔵 Fine-tuned RoBERTa | 81.39 | 81.12 | 125M | 2.9GB |
|         | 🟣 LoRA Adapter BERT | 85.13 | 84.93 | 3.3M | 2.3GB |

---

### **Aspect-Based Sentiment Analysis (3-Class)**

| Domain | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|--------|-------|----------|----------|------------|------------|
| Combined (Laptop + Restaurant) | 🟢 Pretrained DeBERTa | 75.84 | 71.84 | -- | -- |
|         | 🔵 Fine-tuned DeBERTa | 78.04 | 66.74 | 184M | 3.5GB |
|         | 🟣 LoRA Adapter DeBERTa | 79.73 | 73.75 | 813K | 2.5GB |
| Laptop  | 🟢 Pretrained DeBERTa | 81.90 | 80.71 | -- | -- |
|         | 🔵 Fine-tuned DeBERTa | 80.17 | 78.13 | 184M | 4.2GB |
|         | 🟣 LoRA Adapter DeBERTa | 78.02 | 69.84 | 813K | 2.5GB |
| Restaurant | 🟢 Pretrained DeBERTa | 73.41 | 68.35 | -- | -- |
|         | 🔵 Fine-tuned DeBERTa | 80.06 | 71.14 | 184M | 4.2GB |
|         | 🟣 LoRA Adapter DeBERTa | 79.02 | 68.28 | 813K | 2.4GB |

---

### **Fine-Grained Sentiment Analysis (5-Class / 3-Class)**

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|----------|----------|------------|------------|
| E-Commerce (5-Class) | 🟢 Pretrained BERT | 56.87 | 47.82 | -- | -- |
|         | 🔵 Fine-tuned BERT | 66.27 | 50.44 | 167M | 3.2GB |
|         | 🟣 LoRA Adapter BERT | 67.73 | 50.31 | 298K | 1.6GB |
| E-Commerce (3-Class) | 🟢 Pretrained RoBERTa | 79.12 | 53.51 | -- | -- |
|         | 🔵 Fine-tuned RoBERTa | 83.50 | 66.89 | 124M | 2.9GB |
|         | 🟣 LoRA Adapter RoBERTa | 84.96 | 66.47 | 889K | 1.7GB |
| Yelp (5-Class) | 🟢 Pretrained BERT | 55.85 | 55.46 | -- | -- |
|         | 🔵 Fine-tuned BERT | 61.13 | 60.44 | 16.7M | 3.7GB |
|         | 🟣 LoRA Adapter BERT | 64.58 | 64.09 | 1.3M | 3.5GB |
| Yelp (3-Class) | 🟢 Pretrained RoBERTa | 68.93 | 58.05 | -- | -- |
|         | 🔵 Fine-tuned RoBERTa | 82.01 | 77.54 | 124M | 4.2GB |
|         | 🟣 LoRA Adapter RoBERTa | 83.87 | 79.23 | 740K | 4.9GB |

---

### **Emotion Detection (6-Class)**

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|----------|----------|------------|------------|
| Emotion Dataset | 🟢 Pretrained DistilBERT | 93.15 | 89.87 | -- | -- |
|         | 🔵 Fine-tuned DistilBERT | 93.59 | 90.72 | 66.9M | 2.4GB |
|         | 🟣 LoRA Adapter DistilBERT | 93.40 | 90.61 | 668K | 2.1GB |

*💡 LoRA adapters provide a **balance of high accuracy and computational efficiency**, updating only a small fraction of total parameters while reducing GPU memory usage.*

## 🔎 Key Findings

- **LoRA achieves competitive performance with minimal parameter updates**, often matching fully fine-tuned models while training less than 2% of total parameters.
- **Fine-tuning and LoRA significantly improve performance on multi-class and fine-grained tasks**, where pretrained models alone are insufficient.
- **Aspect-Based Sentiment Analysis benefits from task-specific adaptation**, particularly in combined domain datasets.
- **LoRA reduces GPU memory consumption while maintaining strong accuracy**, making it suitable for resource-efficient deployment.

---

## 📁 Repository Structure

```text
SENTI-MATRIX/
│
├── Code/                             
│   │
│   ├── Aspect Based Sentiment (3 classes) Laptop + Restaurant ABSA Dataset.ipynb
│   ├── Aspect Based Sentiment (3 classes) Laptop ABSA Dataset.ipynb
│   ├── Aspect Based Sentiment (3 classes) Restaurant ABSA Dataset.ipynb
│   │
│   ├── Emotion Detection Emotion Dataset.ipynb
│   │
│   ├── Fine-Grained Sentiment Analysis (3 classes) E-Commerce Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) E-Commerce Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) Yelp Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (3 classes) Yelp Reviews.ipynb
│   │
│   ├── Intent Based Sentiment (2 classes) IMDB Dataset.ipynb
│   └── Intent Based Sentiment (2 classes) SST-2 Dataset - Models.ipynb
│
├── Dataset/                          
│   │
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
└── README.md
```

## ⚙️ Installation

Follow the steps below to set up the environment required to reproduce the experiments in **SENTI-MATRIX**.

---

### 1. System Requirements

- Python 3.10 or higher  
- pip package manager  
- NVIDIA GPU (recommended for training)  
- CUDA-enabled environment (for GPU acceleration)  
- 8GB+ RAM (16GB recommended for large models)


### 2. Clone the Repository

```bash
git clone https://github.com/usmanakmal1471/SENTI-MATRIX.git
cd SENTI-MATRIX
```

### 3. Install Dependencies

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install required packages
pip install torch torchvision torchaudio
pip install transformers datasets peft evaluate
pip install pandas scikit-learn matplotlib tqdm
```
---

## 📊 Datasets

The **SENTI-MATRIX** framework utilizes a diverse collection of benchmark and domain-specific datasets to evaluate model robustness across different levels of linguistic granularity. All datasets are pre-processed and stored in the `Dataset/` directory.

| Task | Dataset | Classes | Description |
| :--- | :--- | :---: | :--- |
| **Intent-Based** | SST-2 / IMDb | 2 | Binary sentiment (Positive/Negative). |
| **Intent-Based** | Twitter | 3 | Multi-class (Positive/Neutral/Negative). |
| **Aspect-Based** | Laptop / Restaurant | 3 | Sentiment targeting specific entities (ABSA). |
| **Fine-Grained** | E-Commerce / Yelp | 3 & 5 | Granular rating scales (1–5 stars). |
| **Emotion** | Emotion Dataset | 6 | Sadness, Joy, Love, Anger, Fear, Surprise. |

---

## 🚀 Usage

### Running the Experiments
All experiments are contained within Jupyter Notebooks located in the `Code/` directory. Each notebook is self-contained and follows a three-stage evaluation pipeline:

1.  **Pretrained Model Evaluation:** Benchmarking zero-shot or task-specific models from the Hugging Face Hub.
2.  **Base Model Fine-Tuning:** Performing full-parameter fine-tuning on the target dataset.
3.  **LoRA Adapter Training:** Implementing Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation.

To reproduce the results, navigate to the `Code/` folder and run the desired notebook:

```bash
jupyter notebook "Code/Emotion Detection Emotion Dataset (Pretrained + Base Model + Adapter).ipynb"
```
### LoRA Implementation Snippet
The following configuration is used across the models to ensure a balance between performance and efficiency:

```python
from peft import LoraConfig, get_peft_model

# Standard LoRA configuration for Sequence Classification
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
## 🔧 PEFT Methods

We utilize **LoRA (Low-Rank Adaptation)** as the primary PEFT strategy. This method freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture.



| Feature | Full Fine-Tuning | LoRA Adapter |
| :--- | :--- | :--- |
| **Trainable Parameters** | 100% | < 2% |
| **GPU Memory Usage** | High | Low |
| **Storage Requirement** | ~500MB+ per task | ~5MB - 10MB per task |
| **Performance** | Baseline | Competitive / Superior |

## 📏 Evaluation Metrics

To rigorously assess the models, we categorize our evaluation into two distinct dimensions: **Performance** (Predictive Power) and **Efficiency** (Resource Utilization).

### 1. Performance Metrics
These metrics evaluate how accurately the model identifies sentiment and emotion across the datasets.

* **Accuracy:** Overall percentage of correct predictions.
* **Precision:** Ability of the classifier not to label a negative sample as positive.
* **Recall:** Ability of the classifier to find all the positive samples.
* **F1-Score (Weighted):** The harmonic mean of Precision and Recall, providing a balanced view for imbalanced classes.
* **Similarity Score:** Measures the semantic closeness between predicted labels and ground truth.
* **Confidence Score:** The probability/certainty level assigned by the model to its top prediction.

### 2. Efficiency Metrics
These metrics highlight the advantages of using **LoRA** and Parameter-Efficient Fine-Tuning over traditional methods.

* **Training Time:** The total time required to converge during the fine-tuning process.
* **Trainable Parameters:** The specific count of weights updated (e.g., 100% for Base Models vs. <2% for LoRA).
* **GPU Storage:** Peak VRAM consumption during training and inference, measured in Gigabytes (GB).

---

Markdown
## 📏 Evaluation Metrics

The evaluation of **SENTI-MATRIX** is categorized into two distinct dimensions: **Performance Metrics** to measure predictive power and **Efficiency Metrics** to measure resource utilization.

### 1. Performance Metrics
These metrics assess how accurately the model identifies sentiment and emotion across the four task types.

* **Accuracy:** Overall percentage of correct predictions.
* **Precision:** The ability of the classifier not to label a negative sample as positive.
* **Recall:** The ability of the classifier to find all the positive samples.
* **F1-Score (Weighted):** The harmonic mean of Precision and Recall, providing a balanced view for imbalanced classes.
* **Similarity Score:** Measures the semantic closeness between predicted labels and ground truth.
* **Confidence Score:** The probability/certainty level assigned by the model to its top prediction.

### 2. Efficiency Metrics
These metrics highlight the advantages of using **LoRA** and Parameter-Efficient Fine-Tuning (PEFT).

* **Training Time:** The total time (seconds/minutes) required for the model to converge.
* **Trainable Parameters:** The count of weights updated (e.g., ~100M for Base Models vs. <1M for LoRA).
* **GPU Storage:** Peak VRAM consumption during training and inference (measured in GB).

---

## 📝 Citation

If you use this work or the SENTI-MATRIX framework, please cite it as follows:

```bibtex
@article{akmal2026sentimatrix,
  title     = {SENTI-MATRIX: Multidimensional Sentiment Analysis with Generative Transformer Models},
  author    = {Akmal, Muhammad Usman and 
               Arafat, Md. Easin and 
               Abosinnee, Ali S. and 
               Orosz, Tam{\'a}s},
  journal   = {Research Paper / Thesis Submission},
  year      = {2026},
  note      = {Not Submitted Yet}
}
```
---

## 🙏 Acknowledgements

I would like to express my sincere gratitude to my supervisor, **Md. Easin Arafat**, **Ali S. Abosinnee** & **Professor Tamás Orosz** for their invaluable guidance, constant encouragement, and technical expertise throughout this research.

My deepest thanks go to my **Parents** for their endless support and prayers, which have been my strength during this journey.

Lastly, I want to thank the **Department of Data Science at Eötvös Loránd University (ELTE)** for providing the academic environment and technical support necessary to complete this work.

---

## 📬 Contact

**Muhammad Usman Akmal** Researcher, Faculty of Informatics, Eötvös Loránd University (ELTE)  
Budapest, Hungary  
📧 usman.hu1471@gmail.com

**Md. Easin Arafat** Doctoral Fellow, Data Science and Engineering Department  
Faculty of Informatics, Eötvös Loránd University (ELTE)  
Budapest, Hungary  
📧 arafatmdeasin@inf.elte.hu

---
                                                   *Made with ❤️ at ELTE, Budapest*
