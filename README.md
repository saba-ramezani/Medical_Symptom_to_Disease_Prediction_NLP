# Medical Symptom to Disease Prediction (NLP)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![NLP](https://img.shields.io/badge/NLP-Transformers-green)](https://huggingface.co/transformers/)  

This project applies Natural Language Processing (NLP) and machine learning techniques to predict diseases from symptom descriptions.  
It explores traditional ML models (Naive Bayes, Logistic Regression, SVM), embedding-based retrieval (all-MiniLM-L6-v2, Bio_ClinicalBERT) , and fine-tuned transformer architectures like DistilBert BioBERT and Bio_ClinicalBERT.

---

## Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Results](#-results)
- [Future Work](#-future-work)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## Features
- Exploratory Data Analysis (EDA) with visualizations
- Symptom text preprocessing using spaCy
- Baseline models: Logistic Regression, Naive Bayes, SVM
- Embedding models: SBERT, Bio_ClinicalBERT
- Fine-tuned transformer models: DistilBERT, Bio_ClinicalBERT
- Performance comparison with Accuracy and Macro-F1
- Result plots for easy interpretation

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/medical-symptom-disease-prediction.git
cd medical-symptom-disease-prediction

pip install -r requirements.txt

Or install manually:

pip install scikit-learn pandas numpy seaborn matplotlib shap umap-learn transformers torch
pip install transformers datasets evaluate
pip install spacy
python -m spacy download en_core_web_sm
```

## Dataset

This project uses the Symptom2Disease dataset (available on Kaggle)
- "https://www.kaggle.com/datasets/niyarrbarman/symptom2disease"

Ensure the dataset is placed in your working directory:

url = "/content/drive/MyDrive/MedicalProjects/NLP/Disease_Prediction/Dataset/Symptom2Disease.csv"


### Dataset structure:

label → disease name

text → symptom description

## Workflow
### 1. Data Exploration

Visualize disease distribution
<img width="996" height="759" alt="download" src="https://github.com/user-attachments/assets/2cfcff73-d6ef-40a4-9450-2c66ae468fd4" />


Histogram of symptom text length
<img width="850" height="470" alt="image" src="https://github.com/user-attachments/assets/f1ceb2d0-c2c1-4af9-b519-12bf5a849bb2" />


Top frequent symptom words
<img width="986" height="513" alt="image" src="https://github.com/user-attachments/assets/12de3b35-4249-48b1-9f58-c9d150e60546" />


### 2. Preprocessing

Tokenization, lemmatization

Stopword & non-alphabet removal

### 3. Train/Test Split

Encode labels with LabelEncoder

Stratified 80/20 split

### 4. Baseline Models

TF-IDF + Logistic Regression

TF-IDF + Naive Bayes

TF-IDF + Linear SVM

### 5. Embedding Models

Sentence-BERT (all-MiniLM-L6-v2)

Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)

Similarity-based top-1 prediction

### 6. Transformer Fine-Tuning

DistilBERT (distilbert-base-uncased)

Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)

Training with Hugging Face Trainer API

### 7. Results & Visualization


## Results
Accuracy and Macro-F1 bar plots
<img width="1018" height="700" alt="image" src="https://github.com/user-attachments/assets/c9b6f4e3-b4c3-4140-a48a-7374f62ed9ee" />
<img width="1018" height="700" alt="image" src="https://github.com/user-attachments/assets/d8506145-517f-48bf-a1f6-acee1bcd3a5c" />



⚠️ Values shown are examples. Results will vary depending on dataset splits and training.

## Future Work

Extend dataset with more diverse medical cases

Explore few-shot / zero-shot learning for rare diseases

Add explainability tools (e.g., SHAP, LIME)

Deploy as an API or chatbot for medical triage

⚠️ Disclaimer

This project is intended for research and educational purposes only.
It is not a substitute for professional medical advice. Always consult a licensed physician for medical concerns.

## Author

Developed as part of a Medical NLP research project using classical ML and transformer-based methods.
