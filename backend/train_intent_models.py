"""
train_intent_models.py
-----------------------
Trains two intent-classification models on backend/data/intents.csv:

    1. Multinomial Naive Bayes  (fast, strong text-classification baseline)
    2. Support Vector Machine (linear kernel, via SGDClassifier/SVC)

Both share the same TF-IDF feature pipeline (built in nlp/preprocess.py) so
their predictions are directly comparable. At inference time chat_engine.py
loads BOTH models and combines them: SVM generally gets priority since it
tends to outperform Naive Bayes on short-text intent classification, but the
Naive Bayes score is used as a confidence cross-check / fallback whenever the
SVM's confidence is low - a lightweight ensembling strategy.

Run:
    python train_intent_models.py

Artifacts written to backend/models/:
    tfidf_vectorizer.pkl
    naive_bayes_model.pkl
    svm_model.pkl
    label_encoder.pkl
    metrics.json
"""

import json
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

from nlp.preprocess import preprocess

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    print(f"  {len(df)} labeled utterances across {df['intent'].nunique()} intents")

    print("Preprocessing text (clean -> tokenize -> stopwords -> lemmatize)...")
    df["clean_text"] = df["text"].apply(preprocess)

    # Encode string labels -> integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["intent"])

    # TF-IDF feature extraction. Unigrams keep the feature space dense enough
    # for a modestly-sized training set (bigrams are added only for phrases
    # that repeat often enough to be useful, via min_df).
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vectorizer.fit_transform(df["clean_text"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print("\nTraining Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_preds)
    nb_f1 = f1_score(y_test, nb_preds, average="weighted")
    print(f"  Naive Bayes  -> accuracy: {nb_acc:.3f}, weighted F1: {nb_f1:.3f}")

    print("\nTraining SVM (linear kernel, probability=True)...")
    svm_model = SVC(kernel="linear", C=2.0, probability=True, random_state=42, class_weight="balanced")
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds, average="weighted")
    print(f"  SVM          -> accuracy: {svm_acc:.3f}, weighted F1: {svm_f1:.3f}")

    print("\nClassification report (SVM):")
    print(classification_report(y_test, svm_preds, target_names=label_encoder.classes_, zero_division=0))

    # Evaluation above used a held-out split to report honest metrics.
    # For the model that actually ships with the app, refit on the FULL
    # labeled dataset so the deployed chatbot has maximum intent coverage.
    print("\nRefitting final models on the full dataset for deployment...")
    nb_model.fit(X, y)
    svm_model.fit(X, y)

    # Persist artifacts
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "naive_bayes_model.pkl"), "wb") as f:
        pickle.dump(nb_model, f)
    with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm_model, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    metrics = {
        "naive_bayes": {"accuracy": nb_acc, "f1_weighted": nb_f1},
        "svm": {"accuracy": svm_acc, "f1_weighted": svm_f1},
        "num_samples": len(df),
        "num_intents": int(df["intent"].nunique()),
        "intents": sorted(df["intent"].unique().tolist()),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved models + vectorizer + label encoder to {MODELS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
