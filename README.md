# 20-Newsgroups-Classifier
A machine learning pipeline for classifying news articles into six categories using TF-IDF features, additional linguistic signals, feature selection, and a LinearSVC classifier. Includes train/dev/test splitting, hyperparameter tuning, and evaluation metrics.

## 1. Project Overview

This repository contains the Python code and data used for **Part 2** of the coursework.  
The goal is to train a supervised machine learning model that predicts one of six topic labels (`class-1` … `class-6`) for each news article.

The main steps implemented are:

- Download 
- Loading the dataset (`20_Newsgroups.csv`)
- Splitting into **train**, **development**, and **test** sets (stratified)
- Extracting three types of features:
  - TF–IDF (unigrams + bigrams)
  - Document length (number of tokens)
  - VADER sentiment score (compound)
- Combining all features into a single feature matrix
- Applying **SelectKBest** feature selection with different values of *k*
- Training a **LinearSVC** classifier
- Selecting the best *k* based on macro-F1 on the development set
- Evaluating the final model on the test set (accuracy, macro-precision, macro-recall, macro-F1)

---

## 2. Repository Structure

```
.
├── main.py                # Main pipeline script
├── 20_Newsgroups.csv      # Dataset
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

## 3. Requirements

Python version: **3.9+** (any recent version should work)

Required packages:

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `scipy`

You can install them using:

```bash
pip install -r requirements.txt
```
The script also uses some NLTK resources:
- `stopwords`
- `vader_lexicon`

## 4. How to Run the Code (Terminal Instructions)


## 5. Functionalities and Parameters
