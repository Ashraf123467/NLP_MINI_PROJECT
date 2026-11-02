"""
Smart Customer Feedback & Sentiment Analyzer - Core Engine (analyzer.py)

Module for:
1. Robust dataset loading and normalization for `sentiment-analysis.csv`.
2. Text preprocessing preserving negation cues and cleaning noise.
3. TF-IDF + Logistic Regression pipeline training and persistence.
4. Sentiment prediction with confidence scoring.
5. Explainable keyword-based complaint category detection.
"""

import os
import re
import csv
import io
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# -----------------------------------------------------------------------------
# Stop Words Customization (Preserving Sentiment Negation Words)
# -----------------------------------------------------------------------------
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'none', 'nor', 'nothing',
    'nowhere', "n't", 'hardly', 'scarcely', 'barely', 'without', 'cannot', "can't",
    "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"
}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - NEGATION_WORDS)

# -----------------------------------------------------------------------------
# Transparent Rule-Based Complaint/Feedback Categories
# -----------------------------------------------------------------------------
FEEDBACK_CATEGORIES = {
    "Customer Support": [
        "support", "service", "staff", "representative", "chat", "helpdesk",
        "agent", "call", "response", "assistance", "helpful", "rude", "unprofessional"
    ],
    "Product Quality": [
        "quality", "broken", "damaged", "break", "broke", "subpar", "poor",
        "defect", "durable", "material", "sound", "performance", "flaw"
    ],
    "Delivery & Shipping": [
        "delivery", "shipping", "shipped", "arrived", "late", "package",
        "packaging", "order", "received", "delayed", "flight", "transit"
    ],
    "Website & Usability": [
        "website", "app", "ui", "navigation", "layout", "speed", "slow",
        "confusing", "loading", "cluttered", "user-friendly", "interface", "bug"
    ],
    "Entertainment & Content": [
        "movie", "book", "song", "album", "music", "game", "plot", "band",
        "concert", "artist", "read", "track", "playlist", "ending", "story"
    ],
    "Hospitality & Dining": [
        "hotel", "restaurant", "food", "meal", "stay", "room", "cafe",
        "dish", "tasteless", "delicious", "luxury", "cruise", "park", "ride"
    ]
}


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Robustly loads the sentiment analysis dataset.
    Handles outer-quoted lines, standard CSVs, encoding, and trailing empty rows.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required dataset not found at path: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    if not raw_lines:
        raise ValueError(f"The dataset file at {filepath} is completely empty.")

    # Detect if the file has lines wrapped in outer quotation marks
    unwrapped_lines = []
    for line in raw_lines:
        # Check for outer enclosing quotes
        if line.startswith('"') and line.endswith('"') and len(line) >= 2:
            inner = line[1:-1].replace('""', '"')
            unwrapped_lines.append(inner)
        else:
            unwrapped_lines.append(line)

    # Parse with standard csv reader
    reader = csv.reader(unwrapped_lines, skipinitialspace=True)
    rows = list(reader)

    if not rows:
        raise ValueError(f"No parseable data found in {filepath}")

    raw_header = [col.strip() for col in rows[0]]
    data_rows = rows[1:]

    # Filter out empty or misaligned rows
    expected_len = len(raw_header)
    clean_data = []
    for r in data_rows:
        cleaned_cells = [c.strip() for c in r]
        if len(cleaned_cells) == expected_len:
            clean_data.append(cleaned_cells)

    if not clean_data:
        raise ValueError("No valid data rows could be parsed from the file.")

    df = pd.DataFrame(clean_data, columns=raw_header)

    # Standardize column naming
    rename_map = {}
    for col in df.columns:
        norm = col.lower().replace("/", "_").replace(" ", "_")
        if "text" in norm:
            rename_map[col] = "Text"
        elif "sentiment" in norm:
            rename_map[col] = "Sentiment"
        elif "source" in norm:
            rename_map[col] = "Source"
        elif "date" in norm or "time" in norm:
            rename_map[col] = "Date_Time"
        elif "user" in norm:
            rename_map[col] = "User_ID"
        elif "location" in norm:
            rename_map[col] = "Location"
        elif "confidence" in norm:
            rename_map[col] = "Confidence_Score"

    df = df.rename(columns=rename_map)

    # Normalize Sentiment values
    if "Sentiment" in df.columns:
        df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.capitalize()

    # Cast Confidence_Score to numeric if present
    if "Confidence_Score" in df.columns:
        df["Confidence_Score"] = pd.to_numeric(df["Confidence_Score"], errors="coerce")

    # Drop any rows where Text or Sentiment is empty/NaN
    df = df.dropna(subset=["Text", "Sentiment"])
    df = df[df["Text"].str.strip() != ""]

    return df


def preprocess_text(text: str, preserve_negations: bool = True) -> str:
    """
    Cleans raw text while preserving crucial sentiment negation indicators.
    - Lowercases text
    - Removes URLs
    - Normalizes common contractions (e.g. n't -> not)
    - Strips unwanted punctuation while keeping words intact
    - Collapses extra whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower().strip()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Normalize contractions that affect sentiment
    contractions = {
        r"won\'t": "will not",
        r"can\'t": "cannot",
        r"n\'t": " not",
        r"\'re": " are",
        r"\'s": " is",
        r"\'d": " would",
        r"\'ll": " will",
        r"\'t": " not",
        r"\'ve": " have",
        r"\'m": " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    # Remove punctuation except alphanumeric and spaces
    # We replace punctuation with single space to avoid merging separated words
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_pipeline() -> Pipeline:
    """
    Constructs a scikit-learn Pipeline with TF-IDF Vectorizer and Logistic Regression.
    Preserves negations by using a customized stop words list and bigrams.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=CUSTOM_STOP_WORDS,
        min_df=1,
        sublinear_tf=True
    )
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier)
    ])


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = None
) -> dict:
    """
    Trains the sentiment pipeline on the given dataset using stratified splitting.
    Saves the model to disk and returns comprehensive evaluation metrics.
    """
    if "Text" not in df.columns or "Sentiment" not in df.columns:
        raise KeyError("Dataset must contain 'Text' and 'Sentiment' columns.")

    # Preprocess text column
    cleaned_texts = df["Text"].apply(preprocess_text)
    labels = df["Sentiment"]

    unique_classes = labels.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Need at least 2 distinct classes to train, found: {unique_classes}")

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    labels_order = sorted(list(unique_classes))
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    clf_report = classification_report(
        y_test, y_pred, labels=labels_order, output_dict=True, zero_division=0
    )
    clf_report_text = classification_report(
        y_test, y_pred, labels=labels_order, zero_division=0
    )

    # Save pipeline if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pipeline, save_path)

    results = {
        "pipeline": pipeline,
        "classes": labels_order,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "precision_macro": macro_prec,
        "recall_macro": macro_rec,
        "f1_macro": macro_f1,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "classification_report_text": clf_report_text,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }
    return results


def load_model(model_path: str) -> Pipeline:
    """Loads a previously trained pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)


def predict_sentiment(text: str, pipeline: Pipeline) -> dict:
    """
    Predicts sentiment for a single piece of feedback.
    Returns predicted label, confidence score, class probabilities, and cleaned text.
    """
    clean = preprocess_text(text)
    if not clean:
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
            "probabilities": {},
            "cleaned_text": "",
            "warning": "Input text is empty or contains only punctuation."
        }

    probs = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    class_probs = {c: float(p) for c, p in zip(classes, probs)}

    pred_idx = np.argmax(probs)
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "probabilities": class_probs,
        "cleaned_text": clean,
        "warning": None
    }


def detect_feedback_category(text: str) -> dict:
    """
    Transparent, explainable rule-based keyword detector for feedback domain.
    Explains which keywords triggered the category assignment.
    """
    lower = text.lower()
    matched_categories = []

    for category, keywords in FEEDBACK_CATEGORIES.items():
        found = [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", lower)]
        if found:
            matched_categories.append((category, len(found), found))

    if matched_categories:
        # Sort by number of matched keywords
        matched_categories.sort(key=lambda x: x[1], reverse=True)
        top_cat, count, keywords_found = matched_categories[0]
        return {
            "category": top_cat,
            "matched_keywords": keywords_found,
            "match_count": count,
            "all_matches": {cat: kws for cat, _, kws in matched_categories},
            "is_rule_based": True,
            "note": "Rule-based keyword extraction (transparent, non-ML heuristic)."
        }
    else:
        return {
            "category": "General Experience",
            "matched_keywords": [],
            "match_count": 0,
            "all_matches": {},
            "is_rule_based": True,
            "note": "No domain-specific keywords detected; defaulted to General Experience."
        }


def get_top_features(pipeline: Pipeline, n_top: int = 10) -> dict:
    """
    Extracts the most influential TF-IDF terms for each sentiment class from the Logistic Regression weights.
    Excellent for academic viva explanations.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = classifier.classes_

    if len(classes) == 2:
        # Binary classification
        coefs = classifier.coef_[0]
        # Class 1 is classes[1] (typically Positive), Class 0 is classes[0] (typically Negative)
        top_class1_idx = np.argsort(coefs)[-n_top:][::-1]
        top_class0_idx = np.argsort(coefs)[:n_top]

        return {
            classes[1]: list(zip(feature_names[top_class1_idx], coefs[top_class1_idx])),
            classes[0]: list(zip(feature_names[top_class0_idx], np.abs(coefs[top_class0_idx])))
        }
    else:
        results = {}
        for idx, cls_name in enumerate(classes):
            coefs = classifier.coef_[idx]
            top_idx = np.argsort(coefs)[-n_top:][::-1]
            results[cls_name] = list(zip(feature_names[top_idx], coefs[top_idx]))
        return results

