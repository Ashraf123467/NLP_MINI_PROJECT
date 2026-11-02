# Smart Customer Feedback & Sentiment Analyzer

> **B.Tech Computer Science and Engineering (Artificial Intelligence)**  
> **Course:** Natural Language Processing (Laboratory & Mini Project)  
> **Curriculum Integration:** Complete Mapping to Laboratory Experiments 01 through 06

---

## 1. Abstract

In modern customer-centric businesses, consumer reviews and feedback arrive in vast volumes across multiple digital channels such as e-commerce platforms, app stores, social media, and customer support portals. Manually triaging and analyzing these reviews is labor-intensive, error-prone, and slow. 

The **Smart Customer Feedback & Sentiment Analyzer** is an end-to-end NLP and machine learning software system developed using Python, Scikit-Learn, spaCy, NLTK, and Streamlit. The system takes raw, unstructured customer feedback and performs:
1. Multi-stage text preprocessing (script validation, filtration, tokenization, negation-preserving stopword filtering, stemming).
2. Lexical & morphological decomposition (free vs. bound morphemes, inflectional vs. derivational morphology) and orthographic word generation.
3. Statistical N-Gram language modeling and perplexity/likelihood calculation.
4. Syntactic analysis (Part-of-Speech tagging and shallow phrase chunking for noun/verb phrases).
5. Information extraction via statistical Named Entity Recognition (NER) and regex heuristics.
6. Explainable sentiment classification (`Positive` vs. `Negative`) with confidence scoring.
7. Real dataset visualization and high-throughput batch CSV processing.

The project is built around the verified primary dataset **`sentiment-analysis.csv`**, guaranteeing reproducibility and real-world analytical validity without synthetic or hardcoded data.

---

## 2. Problem Statement

Organizations struggle to process feedback efficiently due to:
- **Linguistic Ambiguity & Negation Loss:** Reviews contain informal slang, contractions, and critical negation words (e.g., "not happy", "never arrived"). Naive text preprocessing often strips negation words, inverting the sentiment polarity.
- **Unstructured Entity Information:** Reviews mention specific products, dates, transaction amounts, and regional stores, but standard classifiers treat text purely as bag-of-words without identifying key entities.
- **Syntactic Ignorance:** Classifiers often ignore grammatical roles (distinguishing the *target product* represented by noun phrases from the *evaluation* represented by adjectives and verbs).
- **Data Encapsulation Noise:** Real-world datasets often suffer from formatting anomalies (e.g., rows wrapped in outer quotation marks and internal quotes).
- **Lack of Interpretability:** Black-box deep learning models fail to explain *why* a customer review was flagged as negative, hindering viva demonstration and organizational trust.

---

## 3. Academic Mapping: All Six NLP Laboratory Concepts

As part of the academic curriculum for B.Tech CSE (Artificial Intelligence), this mini project demonstrates practical application of all six NLP laboratory experiments. While the customer-facing Streamlit UI presents these capabilities under clean, professional engineering headings, their formal academic mapping is documented below:

| Exp No. | Exact Lab Manual Title | Project Feature / Module | Functions & Implementation Details | Role in Customer Feedback Intelligence | Viva-Friendly Summary |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **01** | **Preprocessing of Text (Tokenization, Filtration, Script validation, Stop Word Removal, Stemming)** | `analyzer.py`<br>`app.py` (Preprocessing Breakdown Tab) | `validate_script()`<br>`filter_noise()`<br>`tokenize_feedback()`<br>`remove_stopwords_with_negation()`<br>`stem_tokens()` | Validates Latin script, filters URLs/noise, splits into word/sentence tokens, retains negation cues ('not', 'no', 'never'), and normalizes morphological word variations. | Converts noisy raw reviews into clean, normalized linguistic tokens while preventing sentiment inversion. |
| **02** | **Morphological Analysis** | `lab_features.py`<br>`app.py` (Lexical & Word Analyzer) | `analyze_morphology()` | Analyzes words into Free Morphemes (independent roots) and Bound Morphemes (prefixes/suffixes); classifies morphological processes into Inflectional (grammatical) vs. Derivational (lexical shift). | Unpacks complex consumer terminology (e.g. 'unhappiness' $\rightarrow$ prefix `un-`, root `happy`, suffix `-ness`) to understand core complaint sentiment. |
| **03** | **NGram Model** | `analyzer.py`<br>`app.py` (N-Gram Insights & Probability Tab) | `extract_ngrams()`<br>`calculate_ngram_probabilities()`<br>`get_corpus_top_ngrams()` | Constructs unigrams, bigrams, and trigrams. Computes Maximum Likelihood Estimation (MLE) and Laplace add-1 smoothed sentence probability and perplexity against the review corpus. | Captures co-occurring phrase context ('not good', 'terrible service', 'love product') beyond isolated words. |
| **04** | **POS Tagging Chunking** | `lab_features.py`<br>`app.py` (POS Tagging & Chunks Tab) | `extract_pos_tags()`<br>`extract_syntactic_chunks()` | Assigns Penn Treebank POS tags (NN, VB, JJ, RB). Parses Noun Phrases (`NP: {<DT>?<JJ.*>*<NN.*>+}`) and Verb Phrases (`VP: {<VB.*><RB.*>*<VB.*>*}`). | Noun Phrases identify the customer's *target of evaluation* (e.g. 'the customer support'), while Verb Phrases capture their *experienced action* ('was disappointed'). |
| **05** | **Name Entity Recognition** | `lab_features.py`<br>`app.py` (Entity & Phrase Extractor) | `extract_entities_spacy()`<br>`extract_entities_fallback()`<br>`analyze_entities()` | Uses spaCy `en_core_web_sm` to detect PERSON, ORG, GPE, DATE, MONEY. Provides regex fallback for comparative demonstration and generates displaCy visual annotations. | Extracts concrete entities (store locations, purchase dates, order IDs, product brands) for targeted issue escalation. |
| **06** | **Virtual lab on Word Generator** | `lab_features.py`<br>`app.py` (Word Generation Tab) | `generate_word_forms()` | Generates inflected and derived word forms from base roots using formal orthographic rules: CVC consonant doubling (`stop` $\rightarrow$ `stopped`), consonant+y (`study` $\rightarrow$ `studies`), silent 'e' drop, and sibilant pluralization. | Generates linguistic variations of customer vocabulary to expand review matching coverage and normalize user queries. |

---

## 4. Existing System vs. Proposed System

| Dimension | Existing / Traditional Approaches | Proposed Smart Analyzer System |
| :--- | :--- | :--- |
| **Data Parsing** | Naive `pd.read_csv` fails on encapsulated quoted rows | Custom robust loader strips encapsulation and validates all 7 columns |
| **Preprocessing** | Global stopword lists remove "not", "no", "never" | Custom stopword pruning preserves negations and leverages bigrams |
| **N-Gram Modeling** | Ignored or limited to unigrams | Bigram/trigram statistical modeling with Laplace smoothing and perplexity |
| **Syntax & Grammar** | Words treated purely as flat bag-of-words | Full POS tagging and shallow parsing into Noun/Verb phrase chunks |
| **Entity Recognition** | Absent or unlinked to feedback analysis | Integrated spaCy statistical NER + fallback regex extractor (Lab 05) |
| **Morphological Study** | Neglected in standard web apps | Dedicated interactive morphology dissection and word generator (Labs 02 & 06) |
| **Batch Processing** | Manual one-by-one testing | High-throughput batch CSV upload with summary metrics and export |

---

## 5. Technologies Used

- **Programming Language:** Python 3.11
- **Machine Learning & NLP:**
  - `scikit-learn`: TfidfVectorizer, LogisticRegression, Pipeline, train_test_split, evaluation metrics
  - `spaCy` (`en_core_web_sm`): Statistical Named Entity Recognition pipeline
  - `nltk`: Tokenization, Penn Treebank POS tagging, RegexpParser phrase chunking, Porter stemmer, stopword filtering
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
+-----------------------------------------------------------------------------------+
|                           Streamlit Frontend (app.py)                             |
|  +--------------------+  +--------------------+  +------------------------------+ |
|  | Dashboard Analytics|  | Single Prediction  |  | Batch CSV Processing         | |
|  +--------------------+  +--------------------+  +------------------------------+ |
|  +--------------------+  +--------------------+  +------------------------------+ |
|  | Entity & Phrases   |  | Lexical & Word Lab |  | Performance & N-Gram Insights| |
|  +--------------------+  +--------------------+  +------------------------------+ |
+-----------------------------------------------------------------------------------+
                                          |
                          +---------------+---------------+
                          |                               |
                          v                               v
+------------------------------------+  +-----------------------------------------+
|      Core Engine (analyzer.py)     |  |       Linguistic Suite (lab_features.py)|
|  - Quote-Unwrapping CSV Loader     |  |  - Penn Treebank POS Tagging            |
|  - Multi-Stage Preprocessing:      |  |  - Syntactic Chunker (NP & VP Grammars) |
|    * Script Validation             |  |  - spaCy Statistical NER & Regex Fallback|
|    * Noise & Punctuation Filter    |  |  - Morpheme Analyzer (Free vs. Bound)   |
|    * Tokenization & Negation Stems |  |  - Orthographic Word Generation Engine  |
|  - Statistical N-Gram Language Model| +-----------------------------------------+
|  - TF-IDF + Logistic Regression     |                   |
+------------------------------------+                   |
                  |                                       v
                  v                     +-----------------------------------------+
+------------------------------------+  | Generated Outputs & Artifacts           |
| Primary Storage & Model Artifacts  |  |  - outputs/evaluation_report.txt        |
|  - data/sentiment-analysis.csv     |  |  - outputs/confusion_matrix.png         |
|  - models/sentiment_model.pkl      |  |  - assets/project_logo.png              |
+------------------------------------+  +-----------------------------------------+
```

---

## 8. Detailed Module Descriptions

### 1. `analyzer.py`
- **`load_dataset(filepath)`**: Robust parser handling outer-quoted rows and comma delimiters.
- **`validate_script(text)`**: Verifies Latin/English script conformance and flags multi-script/non-ASCII anomalies.
- **`filter_noise(text)`**: Removes URLs, non-alphanumeric noise, and expands contractions (`n't` -> `not`).
- **`tokenize_feedback(text)`**: Produces word tokens and sentence tokens.
- **`remove_stopwords_with_negation(tokens)`**: Removes common stopwords while preserving critical negation words (`not`, `no`, `never`).
- **`stem_tokens(tokens)`**: Implements Porter Stemming for vocabulary normalization.
- **`get_preprocessing_pipeline_breakdown(text)`**: Bundles all 5 preprocessing stages for transparent inspection.
- **`extract_ngrams(tokens, n)`**: Extracts unigrams, bigrams, and trigrams.
- **`calculate_ngram_probabilities(corpus, target_text)`**: Estimates sentence log-likelihood and perplexity with Laplace smoothing.
- **`get_corpus_top_ngrams(df, n=2)`**: Identifies top bigram phrases across positive and negative feedback in the dataset.
- **`train_model(df)`**: Stratified 80/20 train-test split, fitting `Pipeline([('tfidf', vectorizer), ('clf', classifier)])`.
- **`predict_sentiment(text, pipeline)`**: Computes class predictions and posterior probabilities.
- **`detect_feedback_category(text)`**: Rule-based keyword detector mapping feedback into domains.
- **`get_top_features(pipeline)`**: Extracts top positive and negative linear regression weights.

### 2. `lab_features.py`
- **`extract_pos_tags(text)`**: Penn Treebank POS tagging with high-level grammatical categorization (Noun, Verb, Adjective, Adverb).
- **`extract_syntactic_chunks(text)`**: Shallow syntactic chunking extracting Noun Phrases (target products) and Verb Phrases (actions).
- **`extract_entities_spacy(text)`**: Statistical NER using spaCy `en_core_web_sm` with displaCy HTML visualization.
- **`extract_entities_fallback(text)`**: Rule-based regex fallback for dates, money, emails, user handles, and order IDs.
- **`analyze_morphology(word)`**: Identifies Free Morphemes (roots) vs Bound Morphemes (prefixes/suffixes) and classifies Inflectional vs Derivational morphology.
- **`generate_word_forms(base_word)`**: Linguistic word generation applying CVC doubling, consonant+y, silent 'e', and sibilant pluralization rules.

### 3. `evaluate.py`
- Standalone evaluation script generating `outputs/evaluation_report.txt` and `outputs/confusion_matrix.png`.

### 4. `app.py`
- Professional 7-page interactive Streamlit dashboard.

---

## 9. Machine Learning Algorithm & Pipeline

- **Feature Extraction:** **TF-IDF Vectorizer** (Term Frequency - Inverse Document Frequency)
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
  - `ngram_range=(1, 2)`: Captures unigrams and contextual bigrams.
  - `sublinear_tf=True`: Logarithmic frequency scaling ($1 + \log(\text{TF})$).
  - `stop_words`: Customized list preserving negation words.
- **Classification Algorithm:** **Logistic Regression**
  $$P(Y = \text{Positive} \mid X) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
  - `C=1.0`: L2 regularization.
  - `class_weight='balanced'`: Balances class representation.
  - `random_state=42`: Assures academic reproducibility.

---

## 10. Installation & Execution Instructions

### Prerequisites
- Python 3.10 or 3.11 installed.

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

5. **Download the spaCy Language Model & NLTK Corpora:**
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"
   ```

### Running the Application

- **Run Standalone Model Evaluation:**
  ```bash
  python evaluate.py
  ```
- **Launch the Streamlit Web Application:**
  ```bash
  streamlit run app.py
  ```
  *Opens automatically at `http://localhost:8501`.*

---

## 11. Project Directory Structure

```
Smart_Customer_Feedback_Analyzer/
│
├── app.py                      # Main Streamlit web application
├── analyzer.py                 # Core engine (loading, preprocessing, n-grams, ML pipeline)
├── lab_features.py             # Linguistic features (POS, chunking, NER, morphology, word generation)
├── evaluate.py                 # Model evaluation and report generator
├── requirements.txt            # Python dependencies
├── README.md                   # Full documentation and curriculum mapping
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

## 12. Performance & Evaluation Results

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

---

## 13. Academic Viva Reference Questions & Answers

**Q1: What are the distinct stages in text preprocessing (Exp 01) and why are they ordered this way?**  
*Answer:* Preprocessing follows: (1) Script Validation (verifies expected character encoding), (2) Filtration (cleans URLs and non-alphanumeric noise), (3) Tokenization (segments text into words/sentences), (4) Stop Word Removal (removes non-informative words while preserving negations), and (5) Stemming (reduces words to root forms). Ordering is critical: punctuation must be filtered before tokenization, and tokens must exist before stop words or stemming can be applied.

**Q2: What is the difference between Free and Bound Morphemes (Exp 02)?**  
*Answer:* A Free Morpheme can stand alone as an independent meaningful word (e.g., 'cat', 'play', 'service'). A Bound Morpheme cannot stand alone and must attach to a free morpheme (e.g., prefix 'un-', suffix '-ed', '-ness').

**Q3: How do N-Gram Language Models work and what is Laplace Smoothing (Exp 03)?**  
*Answer:* An N-Gram model predicts the likelihood of word sequences based on conditional probability $P(w_n \mid w_{n-1})$. Maximum Likelihood Estimation counts occurrences in the corpus. When an unseen n-gram occurs, its probability is 0; Laplace add-1 smoothing adds 1 to all numerator counts and vocabulary size $V$ to the denominator to prevent zero-probability errors.

**Q4: How does Syntactic Chunking differ from POS Tagging (Exp 04)?**  
*Answer:* POS tagging assigns grammatical tags to single words (e.g. 'the'/DT, 'customer'/NN, 'service'/NN). Syntactic Chunking (shallow parsing) groups contiguous words into meaningful phrases based on grammar rules (e.g. Noun Phrase `[NP: the customer service]`), identifying the target subject of customer sentiment.

**Q5: What is the difference between Rule-Based and Statistical Named Entity Recognition (Exp 05)?**  
*Answer:* Rule-based NER matches predefined regular expression patterns (e.g., currency symbols for MONEY, `@` handles for USERS), but fails on ambiguous contexts. Statistical NER (like spaCy's transition-based model) uses machine learning context vectors to recognize entities based on how words are used in surrounding text.

**Q6: What orthographic rules govern regular English Word Generation (Exp 06)?**  
*Answer:* Standard rules include: (1) CVC Consonant Doubling for short-vowel monosyllabic bases before vowel suffixes (`stop` $\rightarrow$ `stopped`, `stopping`), (2) Consonant + 'y' changing to 'i' before '-es' and '-ed' (`study` $\rightarrow$ `studies`), (3) Dropping silent 'e' before vowel suffixes (`create` $\rightarrow$ `creating`), and (4) Sibilant pluralization appending '-es' to `s, ss, sh, ch, x, z`.
