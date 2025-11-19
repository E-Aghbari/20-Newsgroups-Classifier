# 20-Newsgroups-Classifier
A machine learning pipeline for classifying news articles into six categories using TF-IDF features, additional linguistic signals, feature selection, and a LinearSVC classifier. Includes train/dev/test splitting, hyperparameter tuning, and evaluation metrics.

## 1. Project Overview

This repository contains the Python code and data used for **Part 2** of the coursework.  
The goal is to train a supervised machine learning model that predicts one of six topic labels (`class-1` … `class-6`) for each news article.

The main steps implemented are:

- Download nltk packages
- Loading the dataset (`20_Newsgroups.csv`)
- Splitting into **train**, **development**, and **test** sets (stratified)
- Extracting three types of features:
  - **TF–IDF** (unigrams + bigrams)
  - **Document length** (number of tokens)
  - **VADER** sentiment score (compound)
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

Insatll dependencies:
```bash
pip install -r requirements.txt
```
The script also uses some NLTK resources:
- `stopwords`
- `vader_lexicon`

## 4. Dataset Format
The dataset (`20_Newgroups.csv`) must contain:
```
text, label
article 1, class-1
article n, class-4
```
Any other dataset must follow this format.

## 5. How to Run the Code (Terminal Instructions)
After installing dependencies, run the default dataset (`20_Newsgroups.csv`):
```bash
python main.py
```
Run a custom dataset:
```bash
python main.py --data path_to_dataset
```
This executes the full pipeline:

1. Loads the data

2. Splits into train/dev/test

3. Extracts all three feature types

4. Runs feature selection with different k-values

5. Trains a LinearSVC for each k

6. Selects the best model

7. Evaluates on the test set

Note: The dataset has to match the Dataset Format.
## 6. Functionalities and Parameters

### 6.1 Train-Development-Test Split

- The dataset is split into 80% train + dev and 20% test.
- From 80% split, 12.5% is reserved as the development set. 
- Final proportions:
    - 70% train
    - 10% development
    - 20% test

Parameters :
```py
test_size=0.2
dev_size=0.125
```

### 6.2 Feature Extraction
The model uses three features types:

A. TF-IDF vectors:
```py
ngram_range(1, 2)
stopwords = list(stopwords)
```

B. Document Length
```py
len(sentence.split())
```

C. Sentiment score
```
sia.polarity_scores(sentence)['compound]
```

### 6.3 Feature Selection

This is implemented to reduce dimensionality and improve generalisation.

Candidate values of *k* tried by default:
```py
candidate_k = [1000, 2000, 3000, 4000, 5000]
```

### 6.4 Development set
The development set is used for:
- Selecting the best value of *k* in SelectKBest  
- Comparing models fairly  
- Avoiding overfitting to the training data 

```py
if f1 > best_f1:
    best_f1 = f1
    best_k = k
    best_selector = selector
    best_clf = clf
```
Keeping track of best model according to f1 measure.

### 6.5 Classifier: LinearSVC

The model uses scikit-learn’s LinearSVC.

```py
clf = LinearSVC()
```

- `penalty` = l1
- `loss` = squared_hinge
- `max_iter` = 1000
- `C` (regularization strength) = 1.0

### 6.6 Evaluation Metrics
The final model is evaluated on the test set using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1


These metrics are printed in the terminal.
