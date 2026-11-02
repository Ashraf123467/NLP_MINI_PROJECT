"""
Smart Customer Feedback & Sentiment Analyzer - Streamlit Application (app.py)

Interactive customer feedback intelligence platform integrating:
1. Dashboard Overview of real customer feedback (sentiment-analysis.csv).
2. Single Feedback Analysis (Sentiment, Probabilities, Pipeline Stages, POS & Chunking, N-Grams, NER).
3. Batch Feedback Analysis (CSV upload, prediction, summary, download).
4. Entity & Phrase Extractor (Named Entity Recognition & Syntactic Noun/Verb Chunks).
5. Lexical & Word Analyzer (Morphological Decomposition & Word Generation Engine).
6. Model Evaluation & N-Gram Insights (Confusion Matrix, Metrics, Feature Importance).
7. About Project & Architecture (Technical Architecture, Methodology & Viva FAQ).
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Ensure local imports work reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from analyzer import (
    load_dataset,
    preprocess_text,
    get_preprocessing_pipeline_breakdown,
    calculate_ngram_probabilities,
    get_corpus_top_ngrams,
    train_model,
    load_model,
    predict_sentiment,
    detect_feedback_category,
    get_top_features
)
from lab_features import (
    extract_pos_tags,
    extract_syntactic_chunks,
    analyze_entities,
    extract_entities_spacy,
    extract_entities_fallback,
    analyze_morphology,
    generate_word_forms,
    SPACY_AVAILABLE,
    NLTK_AVAILABLE
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Feedback & Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Sentiment Badges */
    .badge-pos {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    .badge-neg {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    .badge-tag {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
        margin: 2px;
    }
    .viva-card {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "sentiment-analysis.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "project_logo.png")
CM_PLOT_PATH = os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")
REPORT_PATH = os.path.join(BASE_DIR, "outputs", "evaluation_report.txt")

# -----------------------------------------------------------------------------
# Cache Loading Functions
# -----------------------------------------------------------------------------
@st.cache_data
def get_dataset(filepath: str):
    return load_dataset(filepath)

@st.cache_resource
def get_or_train_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            return load_model(MODEL_PATH)
        except Exception:
            pass
    df = load_dataset(DATA_PATH)
    results = train_model(df, test_size=0.2, random_state=42, save_path=MODEL_PATH)
    return results["pipeline"]

# -----------------------------------------------------------------------------
# Sidebar Navigation (Professional Product Names Only)
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.title("💬 Smart Analyzer")

    st.markdown("### **Platform Navigation**")
    nav_option = st.radio(
        "Select Function:",
        [
            "📊 Dashboard Overview",
            "🔍 Single Feedback Analysis",
            "📁 Batch Feedback Analysis",
            "🏷️ Entity & Phrase Extractor",
            "🧬 Lexical & Word Analyzer",
            "📈 Model Evaluation & N-Gram Insights",
            "ℹ️ About Project & Architecture"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### **System Diagnostics**")
    st.markdown(f"• **Dataset:** `sentiment-analysis.csv`")
    st.markdown(f"• **Classifier:** TF-IDF + Logistic Regression")
    st.markdown(f"• **Statistical NER:** {'✅ Loaded (`en_core_web_sm`)' if SPACY_AVAILABLE else '⚠️ Regex Fallback'}")
    st.markdown(f"• **Linguistic Parser:** {'✅ NLTK & spaCy Ready' if NLTK_AVAILABLE else '⚠️ Standard Regex'}")
    st.caption("Smart Customer Feedback & Sentiment Analyzer")


# -----------------------------------------------------------------------------
# MODULE 1: DASHBOARD OVERVIEW
# -----------------------------------------------------------------------------
if nav_option == "📊 Dashboard Overview":
    st.title("📊 Customer Feedback Analytics Dashboard")
    st.markdown(
        "Real-time analytical insights derived strictly from the verified primary dataset: "
        "`sentiment-analysis.csv`. No synthetic or demo records used."
    )

    try:
        df = get_dataset(DATA_PATH)
    except FileNotFoundError:
        st.error(
            "CRITICAL ERROR: Primary dataset `sentiment-analysis.csv` is missing from `data/` directory. "
            "The system cannot proceed without the required dataset."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error parsing primary dataset: {e}")
        st.stop()

    total_records = len(df)
    pos_count = (df["Sentiment"] == "Positive").sum()
    neg_count = (df["Sentiment"] == "Negative").sum()
    pos_pct = (pos_count / total_records) * 100 if total_records else 0
    neg_pct = (neg_count / total_records) * 100 if total_records else 0

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Verified Reviews</div>
            <div class="metric-value">{total_records}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Positive Feedback</div>
            <div class="metric-value" style="color: #16a34a;">{pos_count} <span style="font-size: 16px;">({pos_pct:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Negative Feedback</div>
            <div class="metric-value" style="color: #dc2626;">{neg_count} <span style="font-size: 16px;">({neg_pct:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        avg_conf = df["Confidence_Score"].mean() if "Confidence_Score" in df.columns else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Review Confidence</div>
            <div class="metric-value" style="color: #4f46e5;">{avg_conf:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Visual Insights Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment Distribution")
        fig_sent, ax_sent = plt.subplots(figsize=(5, 3.5))
        colors = ["#22c55e", "#ef4444"]
        labels = ["Positive", "Negative"]
        counts = [pos_count, neg_count]
        wedges, texts, autotexts = ax_sent.pie(
            counts, labels=labels, autopct="%1.1f%%",
            startangle=140, colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        for t in texts:
            t.set_fontsize(11)
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
            at.set_color("white")
        ax_sent.set_title("Overall Sentiment Proportion", fontsize=12, pad=12)
        st.pyplot(fig_sent)
        plt.close(fig_sent)

    with col_right:
        st.subheader("Feedback by Source Channel")
        if "Source" in df.columns:
            source_counts = df["Source"].value_counts().head(8)
            fig_src, ax_src = plt.subplots(figsize=(6, 3.5))
            sns.barplot(x=source_counts.values, y=source_counts.index, palette="Blues_r", ax=ax_src)
            ax_src.set_title("Top Customer Review Platforms", fontsize=12, pad=12)
            ax_src.set_xlabel("Review Count")
            st.pyplot(fig_src)
            plt.close(fig_src)

    # Locations Breakdown
    if "Location" in df.columns:
        st.subheader("Geographical Distribution of Feedback")
        loc_counts = df["Location"].value_counts().head(10)
        st.bar_chart(loc_counts)

    # Frequent Bigram Expressions Across Feedback
    st.markdown("---")
    st.subheader("Most Frequent Customer Review Bigrams")
    st.markdown("Identifies dominant two-word co-occurrences in positive and negative consumer sentiments:")

    top_bg = get_corpus_top_ngrams(df, n=2, top_k=6)
    bg_col1, bg_col2 = st.columns(2)
    with bg_col1:
        st.markdown("##### 🟢 **Top Positive Phrases (Bigrams)**")
        if "Positive" in top_bg:
            pos_bg_df = pd.DataFrame(top_bg["Positive"], columns=["Bigram Sequence", "Corpus Frequency"])
            st.dataframe(pos_bg_df, use_container_width=True)
    with bg_col2:
        st.markdown("##### 🔴 **Top Negative Phrases (Bigrams)**")
        if "Negative" in top_bg:
            neg_bg_df = pd.DataFrame(top_bg["Negative"], columns=["Bigram Sequence", "Corpus Frequency"])
            st.dataframe(neg_bg_df, use_container_width=True)

    # Raw Dataset Explorer
    st.markdown("---")
    st.subheader("Dataset Records Explorer")
    st.markdown("Filter and inspect individual entries from `sentiment-analysis.csv`:")

    filt_col1, filt_col2 = st.columns([1, 3])
    with filt_col1:
        selected_sentiment = st.selectbox("Filter Sentiment:", ["All", "Positive", "Negative"])
    with filt_col2:
        search_kw = st.text_input("Search Text by Keyword:", placeholder="e.g. food, website, product, support...")

    filtered_df = df.copy()
    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df["Sentiment"] == selected_sentiment]
    if search_kw:
        filtered_df = filtered_df[filtered_df["Text"].str.contains(search_kw, case=False, na=False)]

    st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** records:")
    st.dataframe(filtered_df, use_container_width=True)


# -----------------------------------------------------------------------------
# MODULE 2: SINGLE FEEDBACK ANALYSIS
# -----------------------------------------------------------------------------
elif nav_option == "🔍 Single Feedback Analysis":
    st.title("🔍 Single Feedback Linguistic & Sentiment Analysis")
    st.markdown(
        "Analyze customer feedback to classify sentiment, calculate prediction confidence, "
        "identify complaint domains, inspect preprocessing stages, and extract syntactic phrases and entities."
    )

    try:
        pipeline = get_or_train_pipeline()
        df_corpus = get_dataset(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading system assets: {e}")
        st.stop()

    # Pre-filled samples for demonstration
    st.markdown("##### **Quick Preset Samples (from dataset):**")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    sample_text = ""
    if s_col1.button("Sample: Positive Product"):
        sample_text = "I love this product! The quality is outstanding and arrived quickly in New York."
    if s_col2.button("Sample: Support Complaint"):
        sample_text = "I'm extremely disappointed with their customer support. No resolution provided."
    if s_col3.button("Sample: Website Review"):
        sample_text = "Their website is so confusing, slow, and poorly designed. Difficult to navigate."
    if s_col4.button("Sample: Dining Experience"):
        sample_text = "Just had the best meal of my life! Delicious food and top-notch service."

    user_input = st.text_area(
        "Enter Customer Feedback / Review Text:",
        value=sample_text,
        height=100,
        placeholder="Type or paste any customer review or feedback here..."
    )

    if st.button("🚀 Analyze Feedback", type="primary"):
        if not user_input.strip():
            st.warning("Please enter feedback text to analyze.")
        else:
            with st.spinner("Executing linguistic and sentiment analysis..."):
                pred_result = predict_sentiment(user_input, pipeline)
                category_result = detect_feedback_category(user_input)
                prep_stages = get_preprocessing_pipeline_breakdown(user_input)
                pos_tags = extract_pos_tags(user_input)
                chunks_info = extract_syntactic_chunks(user_input)
                ngram_prob_info = calculate_ngram_probabilities(df_corpus["Text"].tolist(), user_input, n=2)
                ner_result = analyze_entities(user_input)

            st.markdown("---")
            st.subheader("Analysis Results")

            # Sentiment & Domain Row
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("#### **Sentiment Classification**")
                pred = pred_result["prediction"]
                conf = pred_result["confidence"]

                if pred == "Positive":
                    st.markdown(f'<span class="badge-pos">● POSITIVE SENTIMENT</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-neg">● NEGATIVE SENTIMENT</span>', unsafe_allow_html=True)

                st.markdown(f"**Model Confidence:** `{conf:.2%}`")
                st.progress(conf)

                st.markdown("**Class Probabilities:**")
                for cls, prob in pred_result["probabilities"].items():
                    st.write(f"- **{cls}:** `{prob:.2%}`")

            with res_col2:
                st.markdown("#### **Feedback Domain & Scope**")
                st.info(f"🏷️ **Detected Domain:** **{category_result['category']}**")
                if category_result["matched_keywords"]:
                    st.caption(f"Matched keyword(s): `{', '.join(category_result['matched_keywords'])}`")
                st.caption(f"ℹ️ {category_result['note']}")

            st.markdown("---")

            # Detailed Linguistic Analysis Tabs
            tab_prep, tab_pos_chunks, tab_ngram, tab_ner = st.tabs([
                "🧹 Text Preprocessing Stages",
                "🏷️ POS Tagging & Phrase Chunks",
                "📊 N-Gram Model & Sentence Likelihood",
                "🎯 Named Entity Extraction"
            ])

            # TAB 1: PREPROCESSING STAGES
            with tab_prep:
                st.subheader("Linguistic Preprocessing Pipeline Breakdown")
                st.markdown(
                    "Demonstrates the step-by-step transformation from raw customer text into clean feature tokens:"
                )

                st.markdown("##### **1. Script Validation & Character Check**")
                s_valid = prep_stages["script_validation"]
                st.write(f"- **Script Classification:** `{s_valid['script']}`")
                st.write(f"- **Encoding Status:** `{s_valid['message']}`")

                st.markdown("##### **2. Noise & Punctuation Filtration**")
                st.write("Cleaned text after removing URLs and extraneous punctuation:")
                st.code(prep_stages["filtered_text"], language="text")

                st.markdown("##### **3. Tokenization**")
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    st.write(f"**Sentence Tokens ({len(prep_stages['sentences'])}):**")
                    st.write(prep_stages["sentences"])
                with p_c2:
                    st.write(f"**Word Tokens ({len(prep_stages['word_tokens'])}):**")
                    st.write(prep_stages["word_tokens"][:15])

                st.markdown("##### **4. Stop Word Removal (Negation Preservation)**")
                st.write("**Tokens with Negation Preserved (Used in Sentiment Analysis):**")
                st.write(prep_stages["stopwords_removed_preserved"])
                st.caption("Notice: Critical sentiment negations (e.g. 'not', 'no', 'never') are explicitly retained.")

                st.markdown("##### **5. Stemming (Porter Stemmer)**")
                st.write("**Stemmed Root Tokens:**")
                st.write(prep_stages["stemmed_tokens"])
                st.caption("Demonstrates word ending truncation to normalize morphological variants.")

            # TAB 2: POS TAGGING & SYNTACTIC CHUNKS
            with tab_pos_chunks:
                st.subheader("Part-of-Speech (POS) Tagging & Syntactic Chunks")
                st.markdown(
                    "Analyzes grammatical categories to separate target entities (Noun Phrases) from evaluative actions (Verb Phrases)."
                )

                col_pos, col_chk = st.columns([1, 1])

                with col_pos:
                    st.markdown("##### **POS Tagging Table**")
                    if pos_tags:
                        pos_df = pd.DataFrame(pos_tags)[["token", "pos_tag", "category", "description"]]
                        pos_df.columns = ["Token", "Penn Tag", "Category", "Linguistic Role"]
                        st.dataframe(pos_df, use_container_width=True, height=280)
                    else:
                        st.write("No tokens tagged.")

                with col_chk:
                    st.markdown("##### **Syntactic Chunks (Shallow Parsing)**")
                    st.caption(f"Parser Engine: {chunks_info.get('engine', 'NLTK')}")
                    if chunks_info["chunks"]:
                        chk_df = pd.DataFrame(chunks_info["chunks"])[["phrase", "type", "description"]]
                        chk_df.columns = ["Extracted Phrase", "Chunk Type", "Semantic Function"]
                        st.dataframe(chk_df, use_container_width=True, height=280)
                    else:
                        st.info("No multi-word phrase chunks detected in this input.")

            # TAB 3: N-GRAMS & LANGUAGE MODELING
            with tab_ngram:
                st.subheader("N-Gram Representation & Sequence Probability")
                st.markdown(
                    "Extracts unigram and bigram sequences and calculates statistical language model likelihood based on the corpus."
                )

                ng_c1, ng_c2 = st.columns(2)
                with ng_c1:
                    st.markdown("##### **Extracted Bigrams (2-Word Sequences)**")
                    st.write(ngram_prob_info["target_bigrams"] if ngram_prob_info["target_bigrams"] else "Insufficient tokens for bigrams.")

                    st.markdown("##### **Statistical Sequence Metrics**")
                    st.write(f"- **Vocabulary Size ($V$):** `{ngram_prob_info['vocab_size']}` unique words")
                    st.write(f"- **Log Sentence Likelihood:** `{ngram_prob_info['sentence_log_prob']}`")
                    st.write(f"- **Perplexity Score:** `{ngram_prob_info['perplexity']}` (Lower indicates more common phrasing)")

                with ng_c2:
                    st.markdown("##### **Laplace-Smoothed Bigram Probabilities**")
                    if ngram_prob_info["bigram_details"]:
                        st.dataframe(pd.DataFrame(ngram_prob_info["bigram_details"]), use_container_width=True)
                    else:
                        st.write("No bigram details available.")

            # TAB 4: NAMED ENTITY RECOGNITION
            with tab_ner:
                st.subheader("Named Entity Recognition (NER)")
                st.caption(f"Engine: {ner_result['engine']}")
                if ner_result["entities"]:
                    ent_df = pd.DataFrame(ner_result["entities"])[["text", "label", "description"]]
                    ent_df.columns = ["Entity Token", "Entity Type", "Linguistic Description"]
                    st.dataframe(ent_df, use_container_width=True)

                    if ner_result.get("html"):
                        st.markdown("##### **Visual Entity Annotations**")
                        st.components.v1.html(ner_result["html"], height=130, scrolling=True)
                else:
                    st.info("No specific named entities (persons, locations, dates, currency) detected in this text.")


# -----------------------------------------------------------------------------
# MODULE 3: BATCH FEEDBACK ANALYSIS
# -----------------------------------------------------------------------------
elif nav_option == "📁 Batch Feedback Analysis":
    st.title("📁 Batch Feedback CSV Analysis")
    st.markdown(
        "Upload any CSV file containing customer feedback to batch-classify sentiment, "
        "calculate confidence scores, and export the enriched results."
    )
    st.info(
        "Note: Batch analysis operates on your newly uploaded CSV and will not alter or replace "
        "the primary `sentiment-analysis.csv` dataset."
    )

    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully: {len(batch_df)} rows detected.")
            st.dataframe(batch_df.head(5), use_container_width=True)

            text_cols = [c for c in batch_df.columns if batch_df[c].dtype == "object"]
            if not text_cols:
                st.error("No text columns found in the uploaded CSV.")
                st.stop()

            default_col = text_cols[0]
            for c in text_cols:
                if any(k in c.lower() for k in ["text", "feedback", "review", "comment"]):
                    default_col = c
                    break

            selected_col = st.selectbox(
                "Select the column containing Feedback Text:",
                text_cols,
                index=text_cols.index(default_col)
            )

            if st.button("⚡ Run Batch Analysis", type="primary"):
                pipeline = get_or_train_pipeline()
                progress_bar = st.progress(0)
                status_text = st.empty()

                preds = []
                confs = []
                categories = []

                total = len(batch_df)
                for idx, row in batch_df.iterrows():
                    val = str(row[selected_col]) if pd.notna(row[selected_col]) else ""
                    res = predict_sentiment(val, pipeline)
                    cat_res = detect_feedback_category(val)

                    preds.append(res["prediction"])
                    confs.append(round(res["confidence"], 4))
                    categories.append(cat_res["category"])

                    if idx % 10 == 0 or idx == total - 1:
                        progress_bar.progress((idx + 1) / total)
                        status_text.text(f"Processed {idx + 1} of {total} records...")

                status_text.text("Batch processing complete!")

                batch_df["Predicted_Sentiment"] = preds
                batch_df["Confidence_Score"] = confs
                batch_df["Detected_Category"] = categories

                st.markdown("---")
                st.subheader("Batch Analysis Summary")
                b_pos = (batch_df["Predicted_Sentiment"] == "Positive").sum()
                b_neg = (batch_df["Predicted_Sentiment"] == "Negative").sum()
                b_total = len(batch_df)

                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("Total Analyzed", b_total)
                bc2.metric("Positive Sentiment", f"{b_pos} ({(b_pos/b_total)*100:.1f}%)")
                bc3.metric("Negative Sentiment", f"{b_neg} ({(b_neg/b_total)*100:.1f}%)")

                fig_b, ax_b = plt.subplots(figsize=(6, 2.8))
                sns.countplot(x="Predicted_Sentiment", data=batch_df, palette=["#22c55e", "#ef4444"], ax=ax_b)
                ax_b.set_title("Predicted Sentiment Breakdown", fontsize=11)
                st.pyplot(fig_b)
                plt.close(fig_b)

                st.subheader("Enriched Results Table")
                st.dataframe(batch_df, use_container_width=True)

                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Analyzed CSV",
                    data=csv_buffer.getvalue(),
                    file_name="analyzed_customer_feedback.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")


# -----------------------------------------------------------------------------
# MODULE 4: ENTITY & PHRASE EXTRACTOR
# -----------------------------------------------------------------------------
elif nav_option == "🏷️ Entity & Phrase Extractor":
    st.title("🏷️ Entity & Syntactic Phrase Extractor")
    st.markdown(
        "Demonstrates advanced information extraction from customer text: "
        "locating structured named entities and parsing grammatical phrase chunks."
    )

    st.markdown("##### **Select Example or Input Custom Text:**")
    ner_presets = [
        "Steve Jobs founded Apple in California, and in 2023 it was worth $3 trillion.",
        "I bought a $75 headphones set from Amazon in New York on 2023-07-02. Contact: support@amazon.com",
        "The customer service flight from London to Paris was delayed by 3 hours. Order ID: ORD-9821."
    ]
    preset_choice = st.selectbox("Select a Preset Example or Type Custom Text:", ["Custom Input"] + ner_presets)

    default_text = preset_choice if preset_choice != "Custom Input" else "I ordered headphones from Amazon in Berlin for $49 on 2023-06-15. User @audiophile789 was very helpful."
    text_input = st.text_area("Customer Text Input:", value=default_text, height=100)

    use_fallback_toggle = st.checkbox(
        "Use Rule-Based Regex Fallback Extractor",
        value=False,
        help="Demonstrates the difference between statistical NLP (spaCy) and rule-based regex heuristics."
    )

    if st.button("🔍 Extract Entities & Phrases", type="primary"):
        if not text_input.strip():
            st.warning("Please enter text to extract.")
        else:
            if use_fallback_toggle:
                ner_res = extract_entities_fallback(text_input)
            else:
                ner_res = extract_entities_spacy(text_input)

            chunks_res = extract_syntactic_chunks(text_input)

            st.markdown("---")
            c_left, c_right = st.columns([1, 1])

            with c_left:
                st.subheader("1. Named Entities (NER)")
                st.info(f"**Engine:** {ner_res['engine']}")
                if ner_res["entities"]:
                    table_data = []
                    for e in ner_res["entities"]:
                        table_data.append({
                            "Entity": e["text"],
                            "Type": e["label"],
                            "Span": f"[{e['start']}, {e['end']}]",
                            "Description": e["description"]
                        })
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                    if ner_res.get("html"):
                        st.markdown("##### **Visual Entity Annotations**")
                        st.components.v1.html(ner_res["html"], height=130, scrolling=True)
                else:
                    st.warning("No named entities detected.")

            with c_right:
                st.subheader("2. Syntactic Chunks (Shallow Parsing)")
                st.info(f"**Parser:** {chunks_res.get('engine', 'Syntactic Parser')}")
                if chunks_res["chunks"]:
                    st.dataframe(pd.DataFrame(chunks_res["chunks"])[["phrase", "type", "description"]], use_container_width=True)
                else:
                    st.warning("No syntactic chunks detected.")


# -----------------------------------------------------------------------------
# MODULE 5: LEXICAL & WORD ANALYZER
# -----------------------------------------------------------------------------
elif nav_option == "🧬 Lexical & Word Analyzer":
    st.title("🧬 Lexical & Morphological Word Analyzer")
    st.markdown(
        "Explore internal word structure (morphemic dissection) and linguistic word generation "
        "governed by standard English orthographic rules."
    )

    tab_analysis, tab_generation = st.tabs([
        "🔬 Morphological Analysis (Free vs. Bound Morphemes)",
        "⚙️ Linguistic Word Generation Engine"
    ])

    with tab_analysis:
        st.subheader("Morphological Decomposition")
        st.markdown(
            "Decomposes words into **Free Morphemes** (roots that stand alone) and **Bound Morphemes** (prefixes & suffixes)."
        )

        sample_words = ["unhappiness", "disconnect", "playing", "studying", "acceptable", "rebuild", "subpar", "taste"]
        selected_sample = st.selectbox("Select a sample word:", ["-- Choose a word --"] + sample_words)
        user_word = st.text_input("Or enter any English word:", value="" if selected_sample.startswith("--") else selected_sample)

        if st.button("Decompose Word", type="primary"):
            if not user_word.strip():
                st.warning("Please enter a word to analyze.")
            else:
                analysis = analyze_morphology(user_word)
                st.markdown("---")
                st.subheader(f"Decomposition of: `{analysis['original_word']}`")

                m1, m2, m3 = st.columns(3)
                m1.metric("Prefix (Bound)", analysis["prefix"] if analysis["prefix"] else "None")
                m2.metric("Root (Free Morpheme)", analysis["root_candidate"])
                m3.metric("Suffix (Bound)", analysis["suffix"] if analysis["suffix"] else "None")

                st.markdown(f"**Morphological Category:** `{analysis['morphology_type']}`")
                st.markdown(f"**Bound Morphemes Identified:** `{', '.join(analysis['bound_morphemes'])}`")
                st.info(f"**Linguistic Explanation:** {analysis['explanation']}")

    with tab_generation:
        st.subheader("Word Generation Engine")
        st.markdown(
            "Enter a base root word to systematically generate inflected and derived word forms "
            "applying English orthographic rules."
        )

        gen_samples = ["play", "study", "stop", "create", "watch", "happy", "connect", "taste"]
        selected_gen = st.selectbox("Select a base root sample:", gen_samples)
        gen_input = st.text_input("Or type a custom base word:", value=selected_gen)

        if st.button("⚡ Generate Word Forms", type="primary"):
            if not gen_input.strip():
                st.warning("Please enter a base word.")
            else:
                gen_result = generate_word_forms(gen_input)
                st.markdown("---")
                st.subheader(f"Generated Word Forms for: `{gen_result['base_word']}`")

                forms_list = []
                for form_name, word_form in gen_result["forms"].items():
                    forms_list.append({
                        "Grammatical / Morphological Category": form_name,
                        "Generated Form": word_form
                    })
                st.table(pd.DataFrame(forms_list))

                st.markdown("#### **Orthographic Rules Applied:**")
                for r in gen_result["rules_applied"]:
                    st.markdown(f"- ✅ {r}")


# -----------------------------------------------------------------------------
# MODULE 6: MODEL EVALUATION & N-GRAM INSIGHTS
# -----------------------------------------------------------------------------
elif nav_option == "📈 Model Evaluation & N-Gram Insights":
    st.title("📈 Machine Learning Performance & N-Gram Analysis")
    st.markdown(
        "Comprehensive performance metrics for the **TF-IDF + Logistic Regression** pipeline "
        "trained strictly on `sentiment-analysis.csv`."
    )

    try:
        df = get_dataset(DATA_PATH)
        eval_results = train_model(df, test_size=0.2, random_state=42, save_path=MODEL_PATH)
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{eval_results['accuracy']:.2%}")
    m2.metric("Weighted Precision", f"{eval_results['precision_weighted']:.2%}")
    m3.metric("Weighted Recall", f"{eval_results['recall_weighted']:.2%}")
    m4.metric("Weighted F1-Score", f"{eval_results['f1_weighted']:.2%}")

    st.markdown("---")

    col_cm, col_report = st.columns([1, 1])

    with col_cm:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            eval_results["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=eval_results["classes"],
            yticklabels=eval_results["classes"],
            cbar=False,
            ax=ax_cm
        )
        ax_cm.set_title("Test Confusion Matrix", fontsize=11, fontweight="bold")
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("Actual Label")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_report:
        st.subheader("Classification Report (Test Split)")
        clf_dict = eval_results["classification_report"]
        report_rows = []
        for key in ["Negative", "Positive", "accuracy", "macro avg", "weighted avg"]:
            if key in clf_dict:
                val = clf_dict[key]
                if isinstance(val, dict):
                    report_rows.append({
                        "Class / Metric": key,
                        "Precision": f"{val['precision']:.3f}",
                        "Recall": f"{val['recall']:.3f}",
                        "F1-Score": f"{val['f1-score']:.3f}",
                        "Support": val["support"]
                    })
                else:
                    report_rows.append({
                        "Class / Metric": "Accuracy",
                        "Precision": "-",
                        "Recall": "-",
                        "F1-Score": f"{val:.3f}",
                        "Support": eval_results["test_size"]
                    })
        st.table(pd.DataFrame(report_rows))

    st.markdown("---")
    st.subheader("Top Influential N-Gram Features (Feature Weights)")
    st.markdown(
        "Words and bigrams learned by the model with highest weights in driving Positive and Negative classifications:"
    )

    top_features = get_top_features(eval_results["pipeline"], n_top=8)
    feat_col1, feat_col2 = st.columns(2)

    with feat_col1:
        st.markdown("##### 🟢 **Top Positive Indicators**")
        if "Positive" in top_features:
            pos_df = pd.DataFrame(top_features["Positive"], columns=["N-Gram Feature", "Logistic Regression Weight"])
            st.dataframe(pos_df, use_container_width=True)

    with feat_col2:
        st.markdown("##### 🔴 **Top Negative Indicators**")
        if "Negative" in top_features:
            neg_df = pd.DataFrame(top_features["Negative"], columns=["N-Gram Feature", "Logistic Regression Weight"])
            st.dataframe(neg_df, use_container_width=True)

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as rf:
            rep_text = rf.read()
        st.download_button(
            label="📥 Download Full Academic Evaluation Report (.txt)",
            data=rep_text,
            file_name="sentiment_model_evaluation_report.txt",
            mime="text/plain"
        )


# -----------------------------------------------------------------------------
# MODULE 7: ABOUT PROJECT & ARCHITECTURE
# -----------------------------------------------------------------------------
elif nav_option == "ℹ️ About Project & Architecture":
    st.title("ℹ️ Project Architecture & Viva Examination Guide")
    st.markdown("Comprehensive overview prepared for B.Tech Computer Science & AI project evaluation.")

    viva_tab1, viva_tab2, viva_tab3 = st.tabs([
        "🏗️ Architecture & Modules",
        "🎓 Viva Q&A Guide",
        "⚠️ Limitations & Future Scope"
    ])

    with viva_tab1:
        st.subheader("System Architecture")
        st.markdown("""
        ```
        +-----------------------------------------------------------------------------+
        |                          Streamlit UI (app.py)                              |
        |  [Dashboard] [Single Predict] [Batch CSV] [Entity/Phrase] [Lexical Engine]  |
        +-----------------------------------------------------------------------------+
                                              |
                              +---------------+---------------+
                              |                               |
                              v                               v
        +------------------------------------+  +------------------------------------+
        |      Core Engine (analyzer.py)     |  |   Linguistic Engine (lab_features) |
        |  - Robust CSV Quote Unwrapper      |  |  - POS Tagging (Penn Treebank)     |
        |  - Script Validation & Filtration  |  |  - Phrase Chunking (NP & VP)       |
        |  - Tokenization & Stemming         |  |  - Statistical NER (spaCy)         |
        |  - Negation-Preserving Stopwords   |  |  - Free vs Bound Morphemes         |
        |  - N-Gram Language Model & MLE     |  |  - Orthographic Word Generation    |
        |  - TF-IDF + Logistic Regression    |  +------------------------------------+
        +------------------------------------+                |
                              |                               |
                              v                               v
        +------------------------------------+  +------------------------------------+
        | Primary Dataset & Models           |  | Generated Outputs & Artifacts      |
        | data/sentiment-analysis.csv        |  | outputs/evaluation_report.txt      |
        | models/sentiment_model.pkl         |  | outputs/confusion_matrix.png       |
        +------------------------------------+  +------------------------------------+
        ```
        """)

    with viva_tab2:
        st.subheader("Technical Questions & Answers for Project Viva")

        viva_qa = [
            (
                "Q1: Why is script validation and character filtration necessary before tokenization?",
                "Real-world customer feedback frequently contains mixed encodings, non-ASCII emojis, or corrupted characters. Script validation verifies that the text belongs to the expected language/alphabet, while filtration cleans noise and URLs without corrupting sentiment cues."
            ),
            (
                "Q2: Why must negation words be preserved during stop word removal?",
                "Traditional stop word lists strip words like 'not', 'no', 'never'. In sentiment analysis, removing 'not' turns 'the service was not good' into 'service good', inverting the sentiment polarity. We protect negation tokens and model them as bigrams."
            ),
            (
                "Q3: How do N-Gram models capture context in customer feedback?",
                "Unigrams evaluate single isolated words. Bigrams and trigrams capture sequential context (e.g. 'poor quality', 'love product', 'terrible service'). Our system estimates sequence probability using Laplace add-1 smoothed Maximum Likelihood Estimation."
            ),
            (
                "Q4: What is the distinction between POS tagging and syntactic chunking?",
                "POS tagging labels individual words with grammatical roles (Noun, Verb, Adjective). Chunking (shallow parsing) groups related words into non-overlapping grammatical phrases such as Noun Phrases (target products/services) and Verb Phrases (actions/experiences)."
            ),
            (
                "Q5: How does Named Entity Recognition (NER) differ from simple phrase chunking?",
                "Chunking groups words syntactically based on grammar (e.g. any noun phrase), while NER identifies and classifies real-world entities into predefined semantic classes such as Persons, Organizations, Locations, Dates, and Monetary amounts."
            ),
            (
                "Q6: What is the difference between Free and Bound Morphemes?",
                "A free morpheme can stand alone as an independent meaningful word (e.g. 'play', 'service'). A bound morpheme cannot stand alone and must attach to a free morpheme (e.g. prefix 'un-', suffix '-ing', '-ed', '-ness')."
            )
        ]

        for q, a in viva_qa:
            st.markdown(f"""
            <div class="viva-card">
                <b>{q}</b><br>
                <span style="color: #334155;">{a}</span>
            </div>
            """, unsafe_allow_html=True)

    with viva_tab3:
        st.subheader("Project Limitations & Academic Disclaimers")
        st.markdown("""
        - **Dataset Size:** The primary dataset `sentiment-analysis.csv` contains 96 curated feedback entries. While it achieves high accuracy on its stratified split, real-world generalization across varied informal vernaculars requires continuous learning on larger datasets.
        - **Binary Sentiment Scope:** The dataset contains strictly `Positive` and `Negative` classes. As instructed, no artificial `Neutral` class was fabricated.
        - **Domain Categorization:** Category detection is transparently implemented as an explainable keyword-based rule system rather than an unverified supervised model.
        - **Morphology Scope:** The word generation engine applies regular English orthographic patterns and is designed for academic demonstration.
        """)
