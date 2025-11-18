import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# Load the 20 Newsgroup dataset
df = pd.read_csv('20_Newsgroups.csv')

# Extract raw text and labels (class-1 to class-6)
texts = df['text'].values
y = df['label'].values

# Define a custom set of stopwords for TF-IDF vectorisation
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.update({"'", '*', '&', '-', '=', ';', '@', '.', 'http', ':', 't.co', 'co',
                    '#', '!', '?', ')', '(', '{', '}', ','})


# Main pipeline funtion
def main():


    # Split the dataset into 20% test data
    X_full_train, X_test_text, y_full_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
        )

    # Split the remaining training portion into train and development set
    X_train_text, X_dev_text, y_train, y_dev = train_test_split(
        X_full_train, y_full_train, test_size = 0.125, random_state=42, stratify=y_full_train
        )


    # Feature 1: TF-IDF (unigrams + bigrams) fitted ONLY on training set
    tf = TfidfVectorizer(stop_words = list(stop_words), ngram_range=(1, 2))
    x_tf_train = tf.fit_transform(X_train_text)
    x_tf_test = tf.transform(X_test_text) # Tranform test and dev data 
    x_tf_dev = tf.transform(X_dev_text)

    # Feature 2: Document Length (in tokens)
    x_dl_train = np.array([len(sentence.split()) for sentence in X_train_text]).reshape(-1, 1)
    x_dl_test = np.array([len(sentence.split()) for sentence in X_test_text]).reshape(-1, 1)
    x_dl_dev = np.array([len(sentence.split()) for sentence in X_dev_text]).reshape(-1, 1)

    # Feature 3: Sentiment score (VADER compound)
    sia = SentimentIntensityAnalyzer()
    x_sc_train = np.array([sia.polarity_scores(sentence)['compound'] for sentence in X_train_text]).reshape(-1, 1)
    x_sc_test = np.array([sia.polarity_scores(sentence)['compound'] for sentence in X_test_text]).reshape(-1, 1)
    x_sc_dev = np.array([sia.polarity_scores(sentence)['compound'] for sentence in X_dev_text]).reshape(-1, 1)


    # Combine all feauters (sparse TF-IDF + dense numberic features (Doc Length + Sentiment scores))
    x_train_full = hstack([x_tf_train, x_dl_train, x_sc_train])
    x_test_full = hstack([x_tf_test, x_dl_test, x_sc_test])
    x_dev_full = hstack([x_tf_dev, x_dl_dev, x_sc_dev])


    # Hyperparameter Tuning: Select the best K using the dev set
    candidate_k = [1000, 2000, 3000]
    best_k = None
    best_f1 = 0.0
    best_selector = None
    best_clf = None

    for k in candidate_k:
        
        # Feature selection with K best features using f_classify
        selector = SelectKBest(k = k)
        x_train_sel = selector.fit_transform(x_train_full, y_train)
        x_dev_sel = selector.transform(x_dev_full)

        # Train LinearSVC classifier
        clf = LinearSVC()
        clf.fit(x_train_sel, y_train)

        # Evaluate on development set (Macro F1 score)
        y_dev_pred = clf.predict(x_dev_sel)
        _, _, f1, _ = precision_recall_fscore_support(
        y_dev, y_dev_pred, average='macro', zero_division=0
        )

        print(f"[DEV] k = {k} macro F1 = {f1:.4f}")

        # Keep track of best model
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_selector = selector
            best_clf = clf
    


    print(f"\nBest k on development set: {best_k} (macro F1 = {best_f1:.4f})")

    # Final evaluation on test set using best K and best model
    X_test_sel = best_selector.transform(x_test_full)
    y_test_pred = best_clf.predict(X_test_sel)

    # Evaluate the performance of the model
    acc = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='macro', zero_division=0
    )

    # Print results
    print("\n=== TEST RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    return best_clf


# Run the pipeline
if __name__ == '__main__':
    main()
