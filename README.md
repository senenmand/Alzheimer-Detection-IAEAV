# 🧠 Alzheimer Detection — IAEAV

Automatic detection of Alzheimer's disease from Spanish speech recordings using a full pipeline of audio processing, feature extraction, and classification with Machine Learning and Deep Learning models.

---

## ℹ️ Data 

For this project, IAEAV Dataset has been used. The description of the dataset can be found here:

https://rua.ua.es/entities/publication/095803d3-b25e-4888-a36f-faf764a670eb

## 📋 Description

This project implements an end-to-end pipeline to distinguish between Alzheimer's patients and healthy controls from audio recordings in which participants perform structured cognitive tasks. The system combines:

- **Automatic transcription** with Whisper `large-v3`
- **Cognitive feature extraction** from transcriptions (tasks T2–T7)
- **Semantic embeddings** with BERT and Wav2Vec2
- **Audio features** (log-Mel spectrograms, acoustic features)
- **ML classifiers** (SVM, Random Forest, Logistic Regression, LDA, Naive Bayes, Voting/Stacking ensembles)
- **DL classifier** (CNN on log-Mel) with a hybrid CNN + ensemble approach

---

## 📁 Repository Structure

```
Alzheimer-Detection-IAEAV/
│
├── Extract-Information.ipynb
├── Feature_Selection.ipynb
├── ClassificationML.ipynb
├── ClassificationDL.ipynb
├── visualization.ipynb
├── get_all_info.ipynb
│
└── src/
    ├── audio_preprocessing.py
    ├── config.py
    ├── embedding_features.py
    ├── evaluate_transcription.py
    ├── features.py
    ├── get_dataset_info.py
    ├── models.py
    ├── transcriptor.py
    ├── utils.py
    └── utils_plot.py
```

---

## 🔄 Pipeline

```
Audio (.wav)
    │
    ▼
[1] ZIP Extraction (7-Zip)
    │
    ▼
[2] Automatic Transcription (Whisper large-v3-turbo)
    │
    ▼
[3] Cognitive Task Evaluation (T2–T7)
    │
    ├──► Audio & transcription features
    └──► Semantic embeddings (BERT + Wav2Vec2)
    │
    ▼
[4] Feature Selection (ElasticNet)
    │
    ├──► [5a] ML Classification
    │         SVM · SVM-RBF · Random Forest
    │         Logistic Regression · LDA · Naive Bayes
    │         Voting / Stacking Ensemble
    │
    └──► [5b] DL Classification
              CNN on log-Mel spectrograms
              + Hybrid approach (CNN + Ensemble for uncertain cases)
    │
    ▼
[6] Evaluation (5-fold CV at individual level)
    Accuracy · F1 · Recall · Precision · ECE · Temperature Scaling
```

---

## ⚙️ Installation

```
pip install requirements.txt
```

## 🧩 Models

### Machine Learning
- **SVM** (linear and RBF kernel)
- **Random Forest**
- **Logistic Regression**
- **Linear Discriminant Analysis (LDA)**
- **Naive Bayes**
- **VotingClassifier** (soft voting)
- **StackingClassifier** (meta-estimator: Logistic Regression)

### Deep Learning
- **AlzheimerCNN** — CNN on 3-second log-Mel spectrograms
- **Hybrid CNN + Ensemble** — CNN for high-confidence predictions, ML ensemble for uncertain cases (configurable threshold)
- **Temperature Scaling** — Post-hoc probability calibration

### Feature Extraction
- **Whisper large-v3-turbo** — Spanish speech transcription
- **BERT** — Semantic embeddings from transcriptions
- **Wav2Vec2** — Audio embeddings

---

## 🔧 Configuration

The `src/config.py` file allows you to configure:

- **HuggingFace token** (`AUTH_TOKEN_HF`) for private models
- **Dataset paths** `DATA_DIR`

This configuration comes from variables placed in .env

Adjust the paths to match the local location of your data before running.

---


