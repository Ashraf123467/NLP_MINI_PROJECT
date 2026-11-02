"""
Smart Customer Feedback & Sentiment Analyzer - Evaluation Module (evaluate.py)

Trains/evaluates the sentiment classification pipeline on `sentiment-analysis.csv`.
Generates:
1. outputs/evaluation_report.txt
2. outputs/confusion_matrix.png
Also verifies all linguistic components (Preprocessing, Morphology, N-Grams, POS/Chunking, NER, Word Generation).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import (
    load_dataset,
    train_model,
    get_top_features,
    get_preprocessing_pipeline_breakdown,
    calculate_ngram_probabilities,
    get_corpus_top_ngrams
)
from lab_features import (
    extract_pos_tags,
    extract_syntactic_chunks,
    analyze_entities,
    analyze_morphology,
    generate_word_forms
)


def run_evaluation(
    data_path: str = None,
    model_save_path: str = None,
    report_path: str = None,
    plot_path: str = None
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "sentiment-analysis.csv")
    if model_save_path is None:
        model_save_path = os.path.join(base_dir, "models", "sentiment_model.pkl")
    if report_path is None:
        report_path = os.path.join(base_dir, "outputs", "evaluation_report.txt")
    if plot_path is None:
        plot_path = os.path.join(base_dir, "outputs", "confusion_matrix.png")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    print(f"[EVALUATION] Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    total_records = len(df)
    sent_counts = df["Sentiment"].value_counts().to_dict()

    print(f"[EVALUATION] Training TF-IDF + Logistic Regression model...")
    results = train_model(df, test_size=0.2, random_state=42, save_path=model_save_path)

    # 1. Generate text evaluation report
    report_content = []
    report_content.append("=" * 75)
    report_content.append(" SMART CUSTOMER FEEDBACK & SENTIMENT ANALYZER")
    report_content.append(" Model Evaluation & Linguistic Capabilities Report")
    report_content.append("=" * 75)
    report_content.append("")
    report_content.append("1. DATASET SUMMARY")
    report_content.append(f"   - Dataset File: {os.path.basename(data_path)}")
    report_content.append(f"   - Total Verified Records: {total_records}")
    report_content.append(f"   - Positive Sentiments: {sent_counts.get('Positive', 0)} ({sent_counts.get('Positive', 0)/total_records*100:.1f}%)")
    report_content.append(f"   - Negative Sentiments: {sent_counts.get('Negative', 0)} ({sent_counts.get('Negative', 0)/total_records*100:.1f}%)")
    report_content.append(f"   - Unique Feedback Texts: {df['Text'].nunique()}")
    report_content.append(f"   - Verified Sources: {df['Source'].nunique()} ({', '.join(df['Source'].unique()[:6])}...)")
    report_content.append(f"   - Verified Locations: {df['Location'].nunique()} ({', '.join(df['Location'].unique()[:6])}...)")
    report_content.append("")
    report_content.append("2. EXPERIMENTAL SETUP & PIPELINE")
    report_content.append("   - Feature Extraction: TF-IDF Vectorizer (ngram_range=(1, 2), sublinear_tf=True)")
    report_content.append("   - Stop Words: Custom list (Preserves sentiment negations: 'not', 'no', 'never', etc.)")
    report_content.append("   - Classifier: Logistic Regression (C=1.0, max_iter=1000, class_weight='balanced')")
    report_content.append("   - Split Strategy: Stratified Train-Test Split (80% Train, 20% Test)")
    report_content.append(f"   - Training Samples: {results['train_size']}")
    report_content.append(f"   - Testing Samples: {results['test_size']}")
    report_content.append("")
    report_content.append("3. CLASSIFICATION METRICS (TEST SET)")
    report_content.append(f"   - Accuracy:            {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    report_content.append(f"   - Weighted Precision:  {results['precision_weighted']:.4f}")
    report_content.append(f"   - Weighted Recall:     {results['recall_weighted']:.4f}")
    report_content.append(f"   - Weighted F1-Score:   {results['f1_weighted']:.4f}")
    report_content.append(f"   - Macro F1-Score:      {results['f1_macro']:.4f}")
    report_content.append("")
    report_content.append("4. DETAILED CLASSIFICATION REPORT")
    report_content.append(results["classification_report_text"])
    report_content.append("")
    report_content.append("5. CONFUSION MATRIX")
    report_content.append(f"   Labels: {results['classes']}")
    cm = results["confusion_matrix"]
    report_content.append(f"   Matrix:\n{cm}")
    report_content.append(f"   - True Negative:  {cm[0][0]}")
    report_content.append(f"   - False Positive: {cm[0][1] if cm.shape[1] > 1 else 0}")
    report_content.append(f"   - False Negative: {cm[1][0] if cm.shape[0] > 1 else 0}")
    report_content.append(f"   - True Positive:  {cm[1][1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0}")
    report_content.append("")
    report_content.append("6. TOP INFLUENTIAL N-GRAM FEATURES (FEATURE IMPORTANCE)")
    top_features = get_top_features(results["pipeline"], n_top=8)
    for cls_name, feats in top_features.items():
        report_content.append(f"   Class '{cls_name}':")
        for term, weight in feats:
            report_content.append(f"      * {term:<22} (Weight: {weight:+.4f})")
    report_content.append("")
    report_content.append("7. LINGUISTIC PIPELINE VERIFICATION")
    sample_text = "I loved the movie! The customer service was not good."
    prep = get_preprocessing_pipeline_breakdown(sample_text)
    pos = extract_pos_tags(sample_text)
    chunks = extract_syntactic_chunks(sample_text)
    ngram_eval = calculate_ngram_probabilities(df["Text"].tolist(), sample_text, n=2)
    ner_eval = analyze_entities(sample_text)
    morph_eval = analyze_morphology("unhappiness")
    word_gen_eval = generate_word_forms("study")

    report_content.append(f"   - Sample Text: '{sample_text}'")
    report_content.append(f"   - Preprocessing: Tokens={len(prep['word_tokens'])}, Stemmed={prep['stemmed_tokens']}")
    report_content.append(f"   - POS Tags Extracted: {len(pos)} tokens tagged")
    report_content.append(f"   - Syntactic Chunks Extracted: {[c['phrase'] for c in chunks['chunks']]}")
    report_content.append(f"   - N-Gram Perplexity: {ngram_eval['perplexity']} (Vocab: {ngram_eval['vocab_size']})")
    report_content.append(f"   - Named Entities: {len(ner_eval['entities'])} detected")
    report_content.append(f"   - Morphology: Root of 'unhappiness' -> '{morph_eval['root_candidate']}' ({morph_eval['morphology_type']})")
    report_content.append(f"   - Word Generation for 'study': {list(word_gen_eval['forms'].values())[:4]}")
    report_content.append("=" * 75)

    report_str = "\n".join(report_content)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[EVALUATION] Evaluation report written to: {report_path}")

    # 2. Generate Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        results["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=results["classes"],
        yticklabels=results["classes"],
        cbar=True,
        linewidths=1.5,
        linecolor="#f0f0f0"
    )
    plt.title("Confusion Matrix — Sentiment Classification Pipeline", fontsize=12, pad=15, fontweight="bold")
    plt.xlabel("Predicted Sentiment Label", fontsize=10, labelpad=10)
    plt.ylabel("Actual Sentiment Label", fontsize=10, labelpad=10)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[EVALUATION] Confusion matrix heatmap saved to: {plot_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
