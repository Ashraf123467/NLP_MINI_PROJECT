# Smart Customer Feedback & Sentiment Analyzer

> **B.Tech Computer Science and Engineering (Artificial Intelligence)**  
> **Natural Language Processing Laboratory Project**  
> **Course Experiments Implemented:** Experiment 05 (Named Entity Recognition) & Experiment 06 (Morphological Analysis and Word Generation)

---

## 1. Abstract

In modern customer-centric businesses, consumer reviews and feedback arrive in vast volumes across multiple digital channels such as e-commerce platforms, app stores, social media, and customer support portals. Manually triaging and analyzing these reviews is labor-intensive, error-prone, and slow. 

The **Smart Customer Feedback & Sentiment Analyzer** is an end-to-end NLP and machine learning software system developed using Python, Scikit-Learn, spaCy, and Streamlit. The system takes raw, unstructured customer feedback and performs:
1. Sentiment classification (`Positive` vs. `Negative`) with confidence scoring.
2. Rule-based customer issue categorization.
3. Named Entity Recognition (Lab Exp 05) to extract brands, locations, dates, and amounts.
4. Morphological decomposition and rule-based word generation (Lab Exp 06).
5. Interactive visualization of real dataset patterns and batch CSV prediction.

The project is built around the verified primary dataset **`sentiment-analysis.csv`**, guaranteeing reproducibility and real-world analytical validity without synthetic or hardcoded data.

---

## 2. Problem Statement

Organizations struggle to process feedback efficiently due to:
- **Linguistic Ambiguity:** Reviews contain informal slang, contractions, and critical negation words (e.g., "not happy", "hardly useful"). Naive text preprocessing often strips negation words, leading to misclassification.
- **Unstructured Entity Information:** Reviews mention specific products, dates, transaction amounts, and regional stores, but standard classifiers treat text purely as bag-of-words without identifying key entities.
- **Data Encapsulation Noise:** Real-world datasets often suffer from formatting anomalies (e.g., lines wrapped in outer quotation marks, escaped inner quotes).
- **Lack of Interpretability:** Black-box deep learning models fail to explain *why* a customer review was flagged as negative, hindering viva demonstration and organizational trust.

---

## 3. Project Objectives

1. **Robust Ingestion:** Accurately parse and normalize the verified primary dataset `sentiment-analysis.csv`.
2. **Sentiment Classification:** Train an explainable machine learning pipeline using TF-IDF (unigrams + bigrams) and Logistic Regression with class-balanced weighting.
3. **Preserve Negation Semantics:** Ensure custom stopword filtering retains words like `not`, `no`, `never`, and `n't`.
4. **NLP Lab Experiment 05 Implementation:** Extract real-world named entities (`PERSON`, `ORG`, `GPE`, `DATE`, `MONEY`) using spaCy statistical models and comparative regex heuristics.
5. **NLP Lab Experiment 06 Implementation:** Decompose English words into prefixes, roots, and suffixes; distinguish inflectional from derivational morphology; and generate inflected forms using spelling rules.
6. **Interactive Dashboard:** Build a responsive Streamlit UI supporting single feedback inspection, batch CSV prediction, performance charts, and viva reference materials.

---

## 4. Existing System vs. Proposed System

| Dimension | Existing / Traditional Approaches | Proposed Smart Analyzer System |
| :--- | :--- | :--- |
| **Data Parsing** | Naive `pd.read_csv` fails on encapsulated quoted rows | Custom robust loader strips encapsulation and validates all 7 columns |
| **Stopword Handling** | Global stopword lists remove "not", "no", "never" | Custom stopword pruning preserves negations and leverages bigrams |
| **Sentiment Analysis** | Simple lexicon keyword counting or heavy black-box BERT | Explainable TF-IDF + Logistic Regression with feature weight inspection |
| **Entity Recognition** | Absent or unlinked to feedback analysis | Integrated spaCy statistical NER + fallback regex extractor (Lab 05) |
| **Morphological Study** | Neglected in standard web apps | Dedicated interactive morphology dissection and word generator (Lab 06) |
| **Batch Processing** | Manual one-by-one testing | High-throughput batch CSV upload with summary metrics and export |

---

## 5. Technologies Used

- **Programming Language:** Python 3.11
- **Machine Learning & NLP:**
  - `scikit-learn`: TfidfVectorizer, LogisticRegression, Pipeline, train_test_split, evaluation metrics
  - `spaCy` (`en_core_web_sm`): Statistical Named Entity Recognition pipeline
  - `numpy`: Numerical vector processing
- **Data Engineering & Analysis:**
  - `pandas`: DataFrames, aggregation, and batch processing
  - `csv` (Standard Library): Custom dialect detection and quote unwrapping
- **Visualization & UI:**
  - `streamlit`: Web dashboard, interactive widgets, file uploaders, progress tracking
  - `matplotlib` & `seaborn`: Confusion matrix heatmaps, source distributions, bar charts
- **Serialization:**
  - `joblib`: Machine learning pipeline persistence

---

## 6. Dataset Description (`sentiment-analysis.csv`)

The system is trained, tested, and visualized strictly using **`sentiment-analysis.csv`**:

- **File Name:** `sentiment-analysis.csv`
- **Total Valid Records:** 96 rows
- **Columns (7 total):**
  1. `Text`: The customer review or feedback string.
  2. `Sentiment`: Ground truth class label (`Positive` or `Negative`).
  3. `Source`: Platform origin (`Online Store`, `Spotify`, `Zomato`, `Website Review`, `Yelp Reviews`, `IMDb`, `TripAdvisor`, `Goodreads`, `Twitter`, etc.).
  4. `Date/Time`: Timestamp of the feedback (e.g. `2023-06-15 09:23:14`).
  5. `User ID`: User handle or identifier (e.g. `@user123`, `forumuser1`).
  6. `Location`: Customer geographic location (e.g. `New York`, `Sydney`, `London`, `Paris`, `Berlin`, `Toronto`, `Mumbai`, `Los Angeles`).
  7. `Confidence Score`: Dataset baseline confidence rating (range: `0.55` to `0.95`, mean: `0.79`).

### Class Distribution
- **Positive:** 53 records (55.2%)
- **Negative:** 43 records (44.8%)
- **Class Balance:** Naturally balanced; stratify splitting is applied to maintain the exact distribution across training (80%) and testing (20%) splits.

### Key Preprocessing Finding
The raw CSV file encapsulated entire rows in outer quotation marks (`"Text, Sentiment..."` and `"""I love this product!"", Positive..."`). The system's custom loader unwraps these outer enclosures and unescapes internal quotation marks before structured parsing.

---

## 7. System Architecture

```
+-------------------------------------------------------------------------+
|                       Streamlit Frontend (app.py)                       |
|   +-------------------+  +-------------------+  +--------------------+  |
|   | Dashboard Overview|  | Single Prediction |  | Batch CSV Analysis |  |
|   +-------------------+  +-------------------+  +--------------------+  |
|   +-------------------+  +-------------------+  +--------------------+  |
|   | Lab 05: NER Suite |  | Lab 06: Morphology|  | Model Evaluation   |  |
|   +-------------------+  +-------------------+  +--------------------+  |
+-------------------------------------------------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
+------------------------------------+  +---------------------------------+
|     Core Pipeline (analyzer.py)    |  |     Lab Suite (lab_features.py) |
|  - Custom Quote-Unwrapping Loader  |  |  - spaCy en_core_web_sm (NER)   |
|  - Negation-Preserving Cleaner     |  |  - Regex Heuristic Fallback     |
|  - TF-IDF Vectorizer (1-2 ngrams)  |  |  - Morphological Dissection     |
|  - Logistic Regression Classifier  |  |  - Linguistic Word Generation   |
+------------------------------------+  +---------------------------------+
                    |                               |
                    v                               v
+------------------------------------+  +---------------------------------+
| Primary Storage & Model Artifacts  |  | Evaluation & Visual Assets      |
|  - data/sentiment-analysis.csv     |  |  - outputs/evaluation_report.txt|
|  - models/sentiment_model.pkl      |  |  - outputs/confusion_matrix.png |
|                                    |  |  - assets/project_logo.png      |
+------------------------------------+  +---------------------------------+
```

---

## 8. Detailed Module Descriptions

### 1. `analyzer.py`
- **`load_dataset(filepath)`**: Unwraps line-level outer quotes, parses CSV fields, renames columns to canonical formats, normalizes sentiment strings, and validates data types.
- **`preprocess_text(text, preserve_negations=True)`**: Converts text to lowercase, strips URLs, expands contractions (`n't` -> `not`, `can't` -> `cannot`), removes noisy punctuation while keeping words distinct, and strips redundant whitespace. Negation words are preserved.
- **`train_model(df, test_size=0.2, random_state=42)`**: Executes stratified splitting, fits the `Pipeline([('tfidf', vectorizer), ('clf', classifier)])`, computes metrics, and serializes the pipeline using `joblib`.
- **`predict_sentiment(text, pipeline)`**: Computes class predictions, probability distributions, and confidence percentages.
- **`detect_feedback_category(text)`**: Explainable rule-based keyword matcher that maps customer feedback into domains: *Customer Support, Product Quality, Delivery & Shipping, Website & Usability, Entertainment & Content, Hospitality & Dining*.
- **`get_top_features(pipeline, n_top=10)`**: Inspects linear model weights to report the strongest positive and negative linguistic indicators.

### 2. `lab_features.py`
- **`extract_entities_spacy(text)`**: Invokes spaCy `en_core_web_sm` to extract entity spans, labels, and descriptions, and generates displaCy HTML for inline visualization.
- **`extract_entities_fallback(text)`**: Transparent rule-based regex fallback extracting currency, dates, emails, user handles, and order identifiers.
- **`analyze_morphology(word)`**: Strips known prefixes and suffixes, reconstructs the base stem candidate, and categorizes the process as Inflectional or Derivational.
- **`generate_word_forms(base_word)`**: Produces inflected and derived word forms applying English spelling rules (CVC doubling, consonant + y, silent 'e' drop, sibilant pluralization).

### 3. `evaluate.py`
- Standalone execution script that trains/evaluates the model on `data/sentiment-analysis.csv`.
- Generates `outputs/evaluation_report.txt` with formatted tables and metric summaries.
- Produces `outputs/confusion_matrix.png` using Seaborn.

### 4. `app.py`
- Responsive 7-page interactive Streamlit dashboard.
- Includes quick-test preset buttons, progress bars, chart generation, file uploaders, and academic viva cards.

---

## 9. NLP Concepts Implemented

### NLP Lab Experiment 05 — Named Entity Recognition (NER)
- **Concept:** Information extraction subtask that locates and classifies named entities in unstructured text into pre-defined categories.
- **Implementation:**
  - Employs spaCy's statistical transition-based NER model (`en_core_web_sm`).
  - Identifies entities: `PERSON` (customer/agent names), `ORG` (service providers like Amazon, Spotify), `GPE` (cities like London, Paris, New York), `DATE` (review dates), `MONEY` (prices and refunds).
  - Generates interactive visual markup with spaCy displaCy.
  - Demonstrates contrast with a Regex fallback heuristic for laboratory comparison.

### NLP Lab Experiment 06 — Morphological Analysis and Word Generation
- **Concept:** The study of the internal structure of words and the rules of word formation.
- **Implementation (Part A - Morphological Analysis):**
  - Identifies prefixes (`un-`, `dis-`, `re-`, `sub-`, etc.) and suffixes (`-ing`, `-ed`, `-ness`, `-tion`, `-ment`, etc.).
  - Distinguishes **Inflectional Morphology** (modifies tense or number without changing part of speech) from **Derivational Morphology** (creates new words or alters lexical category).
- **Implementation (Part B - Word Generation):**
  - Generates 3rd person singular (`plays`, `studies`, `watches`), past tense (`played`, `studied`, `stopped`), present participle (`playing`, `studying`, `stopping`), agent nouns (`player`, `stopper`), and derivations.
  - Implements formal orthographic rules:
    - **CVC Doubling:** Words with Consonant-Vowel-Consonant endings double the final consonant before vowel suffixes (`stop` -> `stopped`, `stopping`).
    - **Consonant + 'y':** Final 'y' preceded by a consonant converts to 'i' before '-es' and '-ed' (`study` -> `studies`, `studied`).
    - **Silent 'e' Drop:** Final silent 'e' is dropped before vowel suffixes (`create` -> `creating`, `created`).
    - **Sibilant Pluralization:** Words ending in sibilants (`s`, `ss`, `sh`, `ch`, `x`, `z`) append `-es` (`watch` -> `watches`).

---

## 10. Machine Learning Algorithm & Pipeline

- **Feature Extraction:** **TF-IDF Vectorizer** (Term Frequency - Inverse Document Frequency)
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
  - `ngram_range=(1, 2)`: Captures both individual words and bigram phrases (e.g., "not good", "terrible service").
  - `sublinear_tf=True`: Applies logarithmic sublinear scaling ($1 + \log(\text{TF})$) to prevent frequent words from dominating.
  - `stop_words`: Customized list ensuring that negation words (`not`, `no`, `never`, `hardly`, `n't`) are never removed.
- **Classification Algorithm:** **Logistic Regression**
  - Models the posterior probability of the positive class using the sigmoid function:
    $$P(Y = \text{Positive} \mid X) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
  - `C=1.0`: L2 regularization parameter preventing overfitting.
  - `class_weight='balanced'`: Compensates for minor class imbalance by inversely weighting class frequencies.
  - `random_state=42`: Assures exact academic reproducibility.

---

## 11. Installation Instructions

### Prerequisites
- Python 3.10 or 3.11 installed on your system.
- Git or extracted project archive.

### Setup Steps

1. **Navigate to the Project Directory:**
   ```bash
   cd Smart_Customer_Feedback_Analyzer
   ```

2. **Create a Dedicated Virtual Environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment:**
   - On Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - On Windows (Command Prompt):
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - On Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the spaCy Language Model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## 12. How to Run the Project

### A. Run Model Training & Evaluation Standalone
To execute the automated evaluation pipeline and generate `evaluation_report.txt` and `confusion_matrix.png`:
```bash
python evaluate.py
```

### B. Launch the Streamlit Web Application
To run the full interactive application:
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 13. Project Structure

```
Smart_Customer_Feedback_Analyzer/
│
├── app.py                      # Main Streamlit web application & dashboard
├── analyzer.py                 # Core engine: data loading, cleaning, pipeline training, inference
├── lab_features.py             # Lab Exp 05 (NER) & Lab Exp 06 (Morphology & Word Generation)
├── evaluate.py                 # Standalone model evaluation & report generator
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation & viva guide
│
├── data/
│   └── sentiment-analysis.csv  # Verified primary dataset (preserved exactly)
│
├── models/
│   └── sentiment_model.pkl     # Trained Scikit-Learn pipeline
│
├── outputs/
│   ├── evaluation_report.txt   # Detailed text evaluation report
│   └── confusion_matrix.png    # High-resolution confusion matrix heatmap
│
└── assets/
    └── project_logo.png        # UI banner and logo asset
```

---

## 14. Performance & Evaluation Results

Tested on an 80/20 stratified split of `sentiment-analysis.csv` (76 training rows, 20 test rows):

- **Test Accuracy:** **100.0%** (20/20 test samples correctly classified)
- **Weighted Precision:** **1.000**
- **Weighted Recall:** **1.000**
- **Weighted F1-Score:** **1.000**

### Confusion Matrix
```
               Predicted Negative   Predicted Positive
Actual Negative        9                    0
Actual Positive        0                   11
```

### Top Learned Features
- **Strongest Positive N-Grams:** `love`, `love product`, `song`, `friendly`, `store`, `amazing`, `outstanding`
- **Strongest Negative N-Grams:** `terrible`, `disappointed`, `service terrible`, `service`, `food restaurant`, `damaged`

*(Note: High test accuracy reflects distinct emotional vocabulary in this curated academic dataset. Real-world noisy social media text will exhibit lower accuracy, as discussed in the limitations).*

---

## 15. Screenshots & UI Progression

The Streamlit UI is organized into seven intuitive modules:
1. **Dashboard Overview:** Displays KPI metric cards (Total Records: 96, Positive: 53, Negative: 43), sentiment pie charts, platform bar charts, and a real-time record explorer.
2. **Single Feedback Analysis:** Provides text input with sample buttons, live sentiment badge, confidence bar, complaint domain classification, and entity extraction table.
3. **Batch CSV Analysis:** Allows file upload, column mapping, progress-tracked batch prediction, and CSV download of enriched results.
4. **Lab 05 NER Tool:** Interactive entity parsing with spaCy and displaCy rendering, plus a regex fallback comparator.
5. **Lab 06 Morphology Tool:** Morphological decomposition tab and word generation engine applying CVC doubling and consonant+y rules.
6. **Model Evaluation:** Interactive confusion matrix heatmap and top positive/negative feature importance tables.
7. **About & Viva Guide:** Architecture diagrams and answers to top viva examination questions.

---

## 16. Academic Limitations & Future Scope

### Limitations
1. **Curated Academic Dataset:** The primary dataset contains 96 rows with clear emotional lexicons. It does not encompass rare dialectal slang, sarcastic irony, or mixed sentiments within a single review.
2. **Binary Sentiment Scope:** The dataset strictly defines `Positive` and `Negative` sentiments; no `Neutral` class is fabricated.
3. **Keyword-Based Complaint Categories:** Category detection uses explainable keyword sets because the dataset does not have labeled multi-class complaint annotations.
4. **Morphology Rules:** Word generation focuses on regular English orthographic patterns and does not cover every irregular verb (e.g., `run` -> `ran`, `buy` -> `bought`).

### Future Scope
1. **Aspect-Based Sentiment Analysis (ABSA):** Decompose complex reviews to extract individual sentiments per aspect (e.g., "Food was great (+), but service was slow (-)").
2. **Transformer Integration:** Fine-tune small domain-adapted transformer models (DistilBERT or RoBERTa) on expanded e-commerce datasets.
3. **Active Learning Feedback Loop:** Enable administrators to correct model errors directly in the UI, incrementally retraining the classifier.
4. **Multilingual Support:** Extend morphological analysis and sentiment classification to regional and non-English languages.

---

## 17. Academic Viva Reference Questions & Answers

**Q1: What is the primary difference between Stemming and Lemmatization?**  
*Answer:* Stemming applies heuristic rule-based chopping of word ends (often producing non-words like `studi` from `studying`), whereas Lemmatization uses a vocabulary and morphological analysis to return the legitimate dictionary base form (lemma, e.g. `study`).

**Q2: Why use TF-IDF rather than simple Bag-of-Words (CountVectorizer)?**  
*Answer:* CountVectorizer only counts raw frequency, heavily weighting universally frequent non-discriminative words. TF-IDF downweights words that appear across all documents (high Document Frequency) and highlights distinctive words that appear frequently within a specific class.

**Q3: How does Logistic Regression perform binary classification?**  
*Answer:* It calculates a weighted linear combination of input features $z = \mathbf{w}^T \mathbf{x} + b$ and passes it through the sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$, mapping real-valued inputs into a probability interval $[0, 1]$.

**Q4: What is the purpose of stratification in `train_test_split`?**  
*Answer:* Stratification ensures that the proportion of classes in both training and testing subsets exactly matches the proportion in the original dataset, avoiding class-distribution skew in small datasets.

