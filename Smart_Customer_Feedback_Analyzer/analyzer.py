"""
Smart Customer Feedback & Sentiment Analyzer - Core Engine (analyzer.py)

Module for:
1. Robust dataset loading and normalization for `sentiment-analysis.csv`.
2. Multi-stage text preprocessing (Script validation, Filtration, Tokenization, Stop word removal, Stemming).
3. N-gram language modeling (Unigrams, Bigrams, Trigrams, MLE probability estimation).
4. TF-IDF + Logistic Regression pipeline training and persistence.
5. Sentiment prediction with confidence scoring.
6. Explainable keyword-based complaint category detection.
"""

import os
import re
import csv
import io
import math
import joblib
import pandas as pd
import numpy as np
from collections import Counter
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

# NLTK components for classic NLP operations
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# -----------------------------------------------------------------------------
# Stop Words Customization (Preserving Sentiment Negation Words)
# -----------------------------------------------------------------------------
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'none', 'nor', 'nothing',
    'nowhere', "n't", 'hardly', 'scarcely', 'barely', 'without', 'cannot', "can't",
    "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"
}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - NEGATION_WORDS)

# Porter stemmer instance
_STEMMER = PorterStemmer()

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
        if line.startswith('"') and line.endswith('"') and len(line) >= 2:
            inner = line[1:-1].replace('""', '"')
            unwrapped_lines.append(inner)
        else:
            unwrapped_lines.append(line)

    reader = csv.reader(unwrapped_lines, skipinitialspace=True)
    rows = list(reader)

    if not rows:
        raise ValueError(f"No parseable data found in {filepath}")

    raw_header = [col.strip() for col in rows[0]]
    data_rows = rows[1:]

    expected_len = len(raw_header)
    clean_data = []
    for r in data_rows:
        cleaned_cells = [c.strip() for c in r]
        if len(cleaned_cells) == expected_len:
            clean_data.append(cleaned_cells)

    if not clean_data:
        raise ValueError("No valid data rows could be parsed from the file.")

    df = pd.DataFrame(clean_data, columns=raw_header)

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

    if "Sentiment" in df.columns:
        df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.capitalize()

    if "Confidence_Score" in df.columns:
        df["Confidence_Score"] = pd.to_numeric(df["Confidence_Score"], errors="coerce")

    df = df.dropna(subset=["Text", "Sentiment"])
    df = df[df["Text"].str.strip() != ""]

    return df


# =============================================================================
# LINGUISTIC PREPROCESSING STAGES
# =============================================================================

def validate_script(text: str) -> dict:
    """
    Validates the text script and encoding.
    Ensures input conforms to standard English/Latin characters,
    flagging non-ASCII or multi-script anomalies.
    """
    if not isinstance(text, str):
        return {"is_valid": False, "non_ascii_chars": [], "script": "Unknown", "message": "Non-string input."}

    non_ascii = [c for c in text if ord(c) > 127]
    is_valid = len(non_ascii) == 0
    return {
        "is_valid": is_valid,
        "script": "Latin (Standard English)" if is_valid else "Mixed / Multi-Script",
        "non_ascii_count": len(non_ascii),
        "non_ascii_chars": list(set(non_ascii))[:10],
        "message": "Script conforms to standard English alphabet." if is_valid else f"Detected {len(non_ascii)} non-ASCII character(s)."
    }


def filter_noise(text: str) -> str:
    """
    Filtration stage: strips URLs, unwanted symbols, and noise while preserving words and contractions.
    """
    if not isinstance(text, str):
        return ""
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Expand contractions that contain sentiment negation
    contractions = {
        r"won\'t": "will not",
        r"can\'t": "cannot",
        r"n\'t": " not",
        r"\'re": " are",
        r"\'s": " is",
        r"\'d": " would",
        r"\'ll": " will",
        r"\'m": " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    # Remove unwanted punctuation symbols, leaving alphanumeric characters and spaces
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_feedback(text: str) -> dict:
    """
    Tokenization stage: splits customer feedback into sentence tokens and word tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return {"words": [], "sentences": [], "word_count": 0, "sentence_count": 0}

    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        words = re.findall(r"\b\w+\b", text)

    return {
        "words": words,
        "sentences": sentences,
        "word_count": len(words),
        "sentence_count": len(sentences)
    }


def remove_stopwords_with_negation(tokens: list, preserve_negations: bool = True) -> list:
    """
    Stop word removal: filters out high-frequency grammatical stop words
    while explicitly retaining sentiment negations (e.g. 'not', 'no', 'never').
    """
    try:
        base_stops = set(stopwords.words("english"))
    except Exception:
        base_stops = set(ENGLISH_STOP_WORDS)

    if preserve_negations:
        active_stops = base_stops - NEGATION_WORDS
    else:
        active_stops = base_stops

    return [t for t in tokens if t.lower() not in active_stops and re.match(r"^\w+$", t)]


def stem_tokens(tokens: list) -> list:
    """
    Stemming stage: applies the Porter Stemming algorithm to reduce words to their stems.
    """
    return [_STEMMER.stem(t) for t in tokens]


def get_preprocessing_pipeline_breakdown(text: str) -> dict:
    """
    Demonstrates the complete end-to-end preprocessing stages on customer feedback:
    1. Script Validation
    2. Filtration (Noise & URL removal)
    3. Tokenization (Word & Sentence)
    4. Stop Word Removal (with Negation Preservation)
    5. Stemming (Porter Stemmer)
    """
    # Stage 1: Script Validation
    script_info = validate_script(text)

    # Stage 2: Filtration
    filtered_text = filter_noise(text)

    # Stage 3: Tokenization
    tokens_info = tokenize_feedback(filtered_text)
    raw_words = tokens_info["words"]

    # Stage 4: Stop Word Removal
    clean_words_preserved = remove_stopwords_with_negation(raw_words, preserve_negations=True)
    clean_words_naive = remove_stopwords_with_negation(raw_words, preserve_negations=False)

    # Stage 5: Stemming
    stemmed_words = stem_tokens(clean_words_preserved)

    return {
        "original_text": text,
        "script_validation": script_info,
        "filtered_text": filtered_text,
        "sentences": tokens_info["sentences"],
        "word_tokens": raw_words,
        "stopwords_removed_preserved": clean_words_preserved,
        "stopwords_removed_naive": clean_words_naive,
        "stemmed_tokens": stemmed_words,
        "final_cleaned_string": " ".join(clean_words_preserved)
    }


def preprocess_text(text: str, preserve_negations: bool = True) -> str:
    """
    Production preprocessing function used by the sentiment model pipeline.
    Preserves negations, removes noise and extra spaces.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    filtered = filter_noise(text).lower()
    return filtered


# =============================================================================
# N-GRAM MODELING & PROBABILITY ESTIMATION
# =============================================================================

def extract_ngrams(tokens: list, n: int = 2) -> list:
    """
    Constructs N-Grams (Unigrams, Bigrams, Trigrams) from a token sequence.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def calculate_ngram_probabilities(corpus_texts: list, target_text: str, n: int = 2) -> dict:
    """
    Estimates sentence likelihood using an N-Gram Language Model with
    Maximum Likelihood Estimation (MLE) and Laplace Add-1 Smoothing.
    """
    # Tokenize entire corpus
    all_tokens = []
    for doc in corpus_texts:
        clean = filter_noise(doc).lower()
        all_tokens.extend(clean.split())

    vocab = set(all_tokens)
    vocab_size = len(vocab) if vocab else 1

    # Counts
    unigram_counts = Counter(all_tokens)
    bigram_counts = Counter(extract_ngrams(all_tokens, 2))

    # Target text tokens
    target_clean = filter_noise(target_text).lower()
    target_tokens = target_clean.split()
    target_bigrams = extract_ngrams(target_tokens, 2)

    if not target_bigrams:
        return {
            "target_tokens": target_tokens,
            "target_bigrams": [],
            "bigram_details": [],
            "sentence_log_prob": 0.0,
            "perplexity": 0.0,
            "vocab_size": vocab_size
        }

    bigram_details = []
    total_log_prob = 0.0

    for w1, w2 in target_bigrams:
        c_w1_w2 = bigram_counts.get((w1, w2), 0)
        c_w1 = unigram_counts.get(w1, 0)
        # Laplace smoothed probability: (count(w1, w2) + 1) / (count(w1) + V)
        prob = (c_w1_w2 + 1) / (c_w1 + vocab_size)
        total_log_prob += math.log(prob)
        bigram_details.append({
            "bigram": f"('{w1}', '{w2}')",
            "corpus_count": c_w1_w2,
            "context_count": c_w1,
            "laplace_prob": round(prob, 5)
        })

    # Perplexity calculation: 2^(-total_log_prob / N)
    avg_neg_log = -total_log_prob / len(target_bigrams)
    perplexity = round(math.exp(avg_neg_log), 2)

    return {
        "target_tokens": target_tokens,
        "target_bigrams": [f"{w1} {w2}" for w1, w2 in target_bigrams],
        "bigram_details": bigram_details,
        "sentence_log_prob": round(total_log_prob, 4),
        "perplexity": perplexity,
        "vocab_size": vocab_size
    }


def get_corpus_top_ngrams(df: pd.DataFrame, n: int = 2, top_k: int = 8) -> dict:
    """
    Extracts most frequent N-Grams across Positive vs Negative feedback in the dataset.
    """
    results = {}
    for sentiment in ["Positive", "Negative"]:
        subset = df[df["Sentiment"] == sentiment]
        all_tokens = []
        for text in subset["Text"]:
            cleaned = filter_noise(text).lower()
            tokens = [t for t in cleaned.split() if t not in CUSTOM_STOP_WORDS]
            all_tokens.extend(tokens)
        ngrams = extract_ngrams(all_tokens, n)
        counts = Counter([" ".join(ng) for ng in ngrams])
        results[sentiment] = counts.most_common(top_k)
    return results


# =============================================================================
# MACHINE LEARNING PIPELINE
# =============================================================================

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

    cleaned_texts = df["Text"].apply(preprocess_text)
    labels = df["Sentiment"]

    unique_classes = labels.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Need at least 2 distinct classes to train, found: {unique_classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

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
    Extracts the most influential TF-IDF terms for each sentiment class from Logistic Regression weights.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = classifier.classes_

    if len(classes) == 2:
        coefs = classifier.coef_[0]
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
