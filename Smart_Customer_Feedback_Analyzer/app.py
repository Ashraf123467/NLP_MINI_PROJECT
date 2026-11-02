"""
Smart Customer Feedback & Sentiment Analyzer - Streamlit Application (app.py)

Interactive web application for:
1. Dashboard Overview of real dataset (sentiment-analysis.csv).
2. Single Feedback Analysis (Sentiment, Probabilities, NER, Categorization, Cleaned Text).
3. Batch Feedback Analysis (CSV upload, prediction, summary, download).
4. NLP Lab Experiment 05: Named Entity Recognition (spaCy & Regex Fallback).
5. NLP Lab Experiment 06: Morphological Analysis & Word Generation.
6. Model Evaluation & Performance (Confusion Matrix, Metrics, Feature Importance).
7. About Project & B.Tech Academic Viva Guide.
"""

import os
import sys
import io
import time
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
    train_model,
    load_model,
    predict_sentiment,
    detect_feedback_category,
    get_top_features
)
from lab_features import (
    analyze_entities,
    extract_entities_spacy,
    extract_entities_fallback,
    analyze_morphology,
    generate_word_forms,
    SPACY_AVAILABLE
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

# Custom CSS for modern academic styling
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
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 12px;
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
    # If model doesn't exist, train it using the required dataset
    df = load_dataset(DATA_PATH)
    results = train_model(df, test_size=0.2, random_state=42, save_path=MODEL_PATH)
    return results["pipeline"]

# -----------------------------------------------------------------------------
# Sidebar Navigation & Information
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.title("💬 Smart Analyzer")

    st.markdown("### **Navigation**")
    nav_option = st.radio(
        "Select Module:",
        [
            "📊 Dashboard Overview",
            "🔍 Single Feedback Analysis",
            "📁 Batch Feedback Analysis",
            "🏷️ Lab 05: Named Entity Recognition",
            "🧬 Lab 06: Morphology & Word Gen",
            "📈 Model Evaluation",
            "ℹ️ About Project & Viva Guide"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### **System Environment**")
    st.markdown(f"• **Primary Dataset:** `sentiment-analysis.csv`")
    st.markdown(f"• **ML Model:** TF-IDF + Logistic Regression")
    st.markdown(f"• **spaCy Status:** {'✅ Ready (`en_core_web_sm`)' if SPACY_AVAILABLE else '⚠️ Fallback active'}")
    st.markdown(f"• **Student Spec:** B.Tech CSE (Artificial Intelligence)")
    st.caption("College Mini Project Demonstration")


# -----------------------------------------------------------------------------
# MODULE 1: DASHBOARD OVERVIEW
# -----------------------------------------------------------------------------
if nav_option == "📊 Dashboard Overview":
    st.title("📊 Customer Feedback & Sentiment Dashboard")
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
            <div class="metric-label">Total Verified Records</div>
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
            <div class="metric-label">Avg Dataset Confidence</div>
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
    st.title("🔍 Single Feedback & Review Analysis")
    st.markdown(
        "Analyze customer feedback to classify sentiment, calculate prediction confidence, "
        "identify complaint categories, and extract named entities."
    )

    try:
        pipeline = get_or_train_pipeline()
    except Exception as e:
        st.error(f"Error loading trained sentiment pipeline: {e}")
        st.stop()

    # Pre-filled samples for quick college viva demonstration
    st.markdown("##### **Quick Sample Presets (from dataset):**")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    sample_text = ""
    if s_col1.button("Sample: Positive Product"):
        sample_text = "I love this product! The quality is outstanding and arrived quickly."
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
        placeholder="Type or paste any customer feedback here..."
    )

    if st.button("🚀 Analyze Feedback", type="primary"):
        if not user_input.strip():
            st.warning("Please enter feedback text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                pred_result = predict_sentiment(user_input, pipeline)
                category_result = detect_feedback_category(user_input)
                ner_result = analyze_entities(user_input)

            st.markdown("---")
            st.subheader("Analysis Results")

            r_col1, r_col2 = st.columns([1, 1])

            with r_col1:
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

                st.markdown("#### **Rule-Based Feedback Domain**")
                st.info(f"🏷️ **Category:** **{category_result['category']}**")
                if category_result["matched_keywords"]:
                    st.caption(f"Matched keyword(s): `{', '.join(category_result['matched_keywords'])}`")
                st.caption(f"ℹ️ {category_result['note']}")

            with r_col2:
                st.markdown("#### **Text Preprocessing Breakdown**")
                st.write("**Original Text:**")
                st.code(user_input, language="text")

                st.write("**Cleaned Feature Text (Negations Preserved):**")
                st.code(pred_result["cleaned_text"], language="text")
                st.caption("Lowercased, URLs & punctuation removed, contractions expanded, negations retained.")

                st.markdown("#### **Extracted Entities (NER)**")
                st.caption(f"Engine: {ner_result['engine']}")
                if ner_result["entities"]:
                    ent_df = pd.DataFrame(ner_result["entities"])[["text", "label", "description"]]
                    ent_df.columns = ["Entity", "Type", "Linguistic Description"]
                    st.dataframe(ent_df, use_container_width=True)
                else:
                    st.write("No named entities detected in this specific text.")


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

            # Default to a column with 'text' or 'feedback' or 'review' in name
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
                cleaned_texts = []

                total = len(batch_df)
                for idx, row in batch_df.iterrows():
                    val = str(row[selected_col]) if pd.notna(row[selected_col]) else ""
                    res = predict_sentiment(val, pipeline)
                    cat_res = detect_feedback_category(val)

                    preds.append(res["prediction"])
                    confs.append(round(res["confidence"], 4))
                    categories.append(cat_res["category"])
                    cleaned_texts.append(res["cleaned_text"])

                    if idx % 10 == 0 or idx == total - 1:
                        progress_bar.progress((idx + 1) / total)
                        status_text.text(f"Processed {idx + 1} of {total} records...")

                status_text.text("Batch processing complete!")

                batch_df["Predicted_Sentiment"] = preds
                batch_df["Confidence_Score"] = confs
                batch_df["Detected_Category"] = categories

                # Summary Statistics
                st.markdown("---")
                st.subheader("Batch Analysis Summary")
                b_pos = (batch_df["Predicted_Sentiment"] == "Positive").sum()
                b_neg = (batch_df["Predicted_Sentiment"] == "Negative").sum()
                b_total = len(batch_df)

                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("Total Analyzed", b_total)
                bc2.metric("Positive Sentiment", f"{b_pos} ({(b_pos/b_total)*100:.1f}%)")
                bc3.metric("Negative Sentiment", f"{b_neg} ({(b_neg/b_total)*100:.1f}%)")

                # Visualization of Batch Results
                fig_b, ax_b = plt.subplots(figsize=(6, 2.8))
                sns.countplot(x="Predicted_Sentiment", data=batch_df, palette=["#22c55e", "#ef4444"], ax=ax_b)
                ax_b.set_title("Predicted Sentiment Breakdown", fontsize=11)
                st.pyplot(fig_b)
                plt.close(fig_b)

                st.subheader("Enriched Results Table")
                st.dataframe(batch_df, use_container_width=True)

                # Export CSV
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
# MODULE 4: NLP LAB 05 — NAMED ENTITY RECOGNITION (NER)
# -----------------------------------------------------------------------------
elif nav_option == "🏷️ Lab 05: Named Entity Recognition":
    st.title("🏷️ NLP Lab Experiment 05 — Named Entity Recognition (NER)")
    st.markdown(
        "Demonstrates the extraction of real-world entities (Persons, Organizations, Locations, "
        "Dates, Monetary values) from unstructured customer feedback."
    )

    with st.expander("📖 Academic Background — Lab Exp 05", expanded=False):
        st.markdown("""
        **Objective:** To identify and classify named entities present in customer reviews into predefined categories.
        - **Model:** spaCy `en_core_web_sm` (Small English pipeline trained on OntoNotes 5).
        - **Statistical NER Architecture:** Transition-based parser using convolutional neural networks / embeddings.
        - **Entity Types Recognized:**
          - `PERSON`: Real or fictional people
          - `ORG`: Companies, agencies, institutions
          - `GPE`: Geopolitical entities (countries, cities, states)
          - `DATE`: Absolute or relative dates
          - `MONEY`: Monetary values including currency units
          - `PRODUCT`: Objects, vehicles, foods, etc.
        - **Fallback Heuristic:** Regular expression matching for dates, currency, emails, and order IDs.
        """)

    st.markdown("##### **Test Input:**")
    ner_presets = [
        "Steve Jobs founded Apple in California, and in 2023 it was worth $3 trillion.",
        "I bought a $75 headphones set from Amazon in New York on 2023-07-02. Contact: support@amazon.com",
        "The flight from London to Paris was delayed by 3 hours. Order ID: ORD-9821."
    ]
    preset_choice = st.selectbox("Select a Preset Example or Type Custom Text:", ["Custom Input"] + ner_presets)

    if preset_choice != "Custom Input":
        default_ner_text = preset_choice
    else:
        default_ner_text = "I ordered headphones from Amazon in Berlin for $49 on 2023-06-15. User @audiophile789 was very helpful."

    ner_input = st.text_area("Input Feedback Text:", value=default_ner_text, height=100)

    use_fallback_toggle = st.checkbox(
        "Compare with Regex Fallback Extractor (Laboratory Comparison)",
        value=False,
        help="Demonstrates the difference between statistical NLP (spaCy) and rule-based regex heuristics."
    )

    if st.button("🔍 Extract Entities", type="primary"):
        if not ner_input.strip():
            st.warning("Please enter text for entity recognition.")
        else:
            if use_fallback_toggle:
                result = extract_entities_fallback(ner_input)
            else:
                result = extract_entities_spacy(ner_input)

            st.markdown("---")
            st.subheader("Entity Extraction Results")
            st.info(f"**Extraction Engine:** {result['engine']} — {result['message']}")

            if result["entities"]:
                st.markdown("#### **Extracted Entities Table**")
                table_data = []
                for e in result["entities"]:
                    table_data.append({
                        "Entity Token": e["text"],
                        "Label / Category": e["label"],
                        "Character Span": f"[{e['start']}, {e['end']}]",
                        "Linguistic Explanation": e["description"]
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                if result.get("html"):
                    st.markdown("#### **Visual Entity Annotations (spaCy displaCy)**")
                    st.components.v1.html(result["html"], height=130, scrolling=True)
            else:
                st.warning("No named entities were detected in the given text.")


# -----------------------------------------------------------------------------
# MODULE 5: NLP LAB 06 — MORPHOLOGICAL ANALYSIS & WORD GENERATION
# -----------------------------------------------------------------------------
elif nav_option == "🧬 Lab 06: Morphology & Word Gen":
    st.title("🧬 NLP Lab Experiment 06 — Morphological Analysis & Word Generation")
    st.markdown(
        "Explore morphological decomposition, affix identification (inflectional vs derivational), "
        "and rule-governed word form generation."
    )

    tab_analysis, tab_generation = st.tabs([
        "🔬 Part A: Morphological Analysis",
        "⚙️ Part B: Word Generation Engine"
    ])

    # PART A: MORPHOLOGICAL ANALYSIS
    with tab_analysis:
        st.subheader("Morphological Analysis")
        st.markdown(
            "Enter a word to decompose it into its **prefix**, **root/stem candidate**, and **suffix**, "
            "and identify whether the morphological process is **Inflectional** or **Derivational**."
        )

        with st.expander("📖 Academic Notes: Inflectional vs. Derivational Morphology"):
            st.markdown("""
            - **Inflectional Morphology:** Adds grammatical information (tense, number, comparison) *without* changing the grammatical category or core lexical meaning of the base word.
              - Examples: `play` -> `playing` (progressive), `plays` (3rd person), `played` (past).
            - **Derivational Morphology:** Creates a new word or changes the part of speech / core concept.
              - Examples: `happy` (adj) -> `happiness` (noun), `connect` (verb) -> `disconnect` (opposite verb).
            """)

        sample_words = ["unhappiness", "disconnect", "playing", "studying", "acceptable", "rebuild", "subpar", "taste"]
        selected_sample = st.selectbox("Try a sample word:", ["-- Choose a word --"] + sample_words)
        user_word = st.text_input("Or enter any English word:", value="" if selected_sample.startswith("--") else selected_sample)

        if st.button("Decompose Word", type="primary"):
            if not user_word.strip():
                st.warning("Please enter a word to analyze.")
            else:
                analysis = analyze_morphology(user_word)
                st.markdown("---")
                st.subheader(f"Analysis for: `{analysis['original_word']}`")

                m1, m2, m3 = st.columns(3)
                m1.metric("Prefix", analysis["prefix"] if analysis["prefix"] else "None")
                m2.metric("Base / Root Candidate", analysis["root_candidate"])
                m3.metric("Suffix", analysis["suffix"] if analysis["suffix"] else "None")

                st.markdown(f"**Morphology Classification:** `{analysis['morphology_type']}`")
                st.info(f"**Linguistic Explanation:** {analysis['explanation']}")

    # PART B: WORD GENERATION
    with tab_generation:
        st.subheader("Linguistic Word Generation Engine")
        st.markdown(
            "Enter a base root word (e.g. `play`, `study`, `stop`, `connect`, `happy`) to systematically "
            "generate inflected and derived word forms applying English spelling rules."
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
                st.subheader(f"Generated Forms for Base Word: `{gen_result['base_word']}`")

                forms_list = []
                for form_name, word_form in gen_result["forms"].items():
                    forms_list.append({
                        "Grammatical / Morphological Form": form_name,
                        "Generated Word": word_form
                    })
                st.table(pd.DataFrame(forms_list))

                st.markdown("#### **Orthographic & Phonetic Spelling Rules Applied:**")
                for r in gen_result["rules_applied"]:
                    st.markdown(f"- ✅ {r}")


# -----------------------------------------------------------------------------
# MODULE 6: MODEL EVALUATION
# -----------------------------------------------------------------------------
elif nav_option == "📈 Model Evaluation":
    st.title("📈 Machine Learning Model Evaluation")
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

    # Top KPI Metrics
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
    st.subheader("Top Influential N-Gram Features (Feature Importance)")
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

    # Download Evaluation Report
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
# MODULE 7: ABOUT PROJECT & VIVA GUIDE
# -----------------------------------------------------------------------------
elif nav_option == "ℹ️ About Project & Viva Guide":
    st.title("ℹ️ Project Architecture & Academic Viva Guide")
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
        +-----------------------------------------------------------------+
        |                    Streamlit UI (app.py)                        |
        |  [Dashboard] [Single Predict] [Batch CSV] [NER Lab] [Morph Lab] |
        +-----------------------------------------------------------------+
                                      |
                      +---------------+---------------+
                      |                               |
                      v                               v
        +----------------------------+  +----------------------------+
        |   Sentiment Pipeline       |  |     NLP Lab Modules        |
        |      (analyzer.py)         |  |   (lab_features.py)        |
        |  - Robust CSV Loader       |  |  - Exp 05: spaCy NER       |
        |  - Negation-Aware Cleaner  |  |  - Regex Fallback Extractor|
        |  - TF-IDF Vectorizer       |  |  - Exp 06: Morphology      |
        |  - Logistic Regression     |  |  - Word Generation Rules   |
        +----------------------------+  +----------------------------+
                      |                               |
                      v                               v
        +----------------------------+  +----------------------------+
        | Primary Dataset & Models   |  | Generated Outputs          |
        | data/sentiment-analysis.csv|  | outputs/evaluation_report  |
        | models/sentiment_model.pkl |  | outputs/confusion_matrix   |
        +----------------------------+  +----------------------------+
        ```
        """)

        st.markdown("#### **Module Responsibilities:**")
        st.markdown("""
        1. **`analyzer.py`**:
           - Manages dataset ingestion from `data/sentiment-analysis.csv`, stripping outer quotation encapsulation and handling encoding.
           - Implements text cleaning that retains negation indicators (`not`, `no`, `never`, `n't`).
           - Trains and saves the TF-IDF + Logistic Regression Scikit-Learn Pipeline.
           - Provides transparent rule-based keyword categorization for customer feedback domains.
        2. **`lab_features.py`**:
           - Implements **NLP Lab Experiment 05** (Named Entity Recognition) using spaCy `en_core_web_sm` and a comparative Regex fallback.
           - Implements **NLP Lab Experiment 06** (Morphological Analysis and Word Generation), dissecting affixes and applying orthographic transformation rules.
        3. **`evaluate.py`**:
           - Generates `evaluation_report.txt` and `confusion_matrix.png` autonomously.
        4. **`app.py`**:
           - Full multi-tab Streamlit dashboard delivering interactive visualization and laboratory tooling.
        """)

    with viva_tab2:
        st.subheader("Frequently Asked Questions for Project Viva")

        viva_qa = [
            (
                "Q1: Why did you choose TF-IDF with Logistic Regression instead of Deep Learning (like BERT)?",
                "For intermediate-sized domain datasets (~100 to 1,000 samples), TF-IDF combined with L2-regularized Logistic Regression provides high interpretability, instant training, zero GPU overhead, and eliminates overfitting common to high-parameter transformers. Logistic Regression weights directly expose the most positive and negative n-grams."
            ),
            (
                "Q2: Why is it critical NOT to remove negation words in text preprocessing?",
                "Standard English stopword lists often include words like 'not', 'no', 'never'. If removed, 'the food was not good' becomes 'food good', which completely inverts the sentiment label. Our pipeline preserves negation cues and uses unigrams and bigrams (ngram_range=(1,2)) to capture phrases like 'not recommended' and 'not good'."
            ),
            (
                "Q3: What is the difference between Inflectional and Derivational Morphology (Lab 06)?",
                "Inflectional morphology modifies a word to express grammatical categories such as tense (-ed, -ing), number (-s, -es), or comparison (-er, -est) without altering the fundamental lexical category or meaning. Derivational morphology forms new words or shifts the syntactic category (e.g., verb 'connect' to noun 'connection', or adjective 'happy' to noun 'happiness' / opposite 'unhappy')."
            ),
            (
                "Q4: How does Named Entity Recognition (NER) assist customer feedback systems (Lab 05)?",
                "NER automatically extracts specific actionable metadata from unstructured text—such as product names, customer service agent names, competitor mentions, transaction amounts, locations, and dates—allowing organizations to route complaints directly to the responsible regional department."
            ),
            (
                "Q5: How did you handle the unusual structure in `sentiment-analysis.csv`?",
                "Inspection revealed that each CSV record was encapsulated in an outer set of quotation marks with internal escaped quotes. A naive reader parsed the whole row as a single column. We engineered a robust parser that unwraps outer quotes, unescapes inner quotations, strips whitespace, and parses all 7 original columns accurately."
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
        - **Complaint Category Detection:** The dataset does not have labeled complaint categories. Category detection is transparently implemented as an explainable keyword-based rule system rather than an unverified supervised model.
        - **Morphology Scope:** English contains irregular verbs (e.g., `go` -> `went`, `buy` -> `bought`). The morphological generator applies standard orthographic rules (CVC consonant doubling, sibilant pluralization, consonant+y modification) and is designed for academic demonstration.
        """)

