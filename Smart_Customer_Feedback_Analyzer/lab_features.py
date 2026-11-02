"""
Smart Customer Feedback & Sentiment Analyzer - Linguistic Features (lab_features.py)

Implementations for:
1. Morphological Analysis (Free vs Bound morphemes, Root identification, Inflectional vs Derivational).
2. Part-of-Speech (POS) Tagging & Syntactic Chunking (Noun phrases, Verb phrases, Shallow parsing).
3. Named Entity Recognition (NER) (spaCy statistical pipeline & rule-based regex fallback).
4. Word Generation Engine (CVC doubling, consonant + y, silent 'e' drop, sibilant pluralization).
"""

import re
from typing import List, Dict, Any, Tuple

# Try loading spaCy
SPACY_AVAILABLE = False
nlp = None
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except Exception:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False

# Try loading NLTK components
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag, RegexpParser
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# =============================================================================
# PART-OF-SPEECH (POS) TAGGING & SYNTACTIC CHUNKING
# =============================================================================

PENN_TAG_EXPLANATIONS = {
    "CC": "Coordinating conjunction (and, but, or)",
    "CD": "Cardinal number (one, two, 123)",
    "DT": "Determiner (the, a, this, that)",
    "EX": "Existential there (there is, there are)",
    "FW": "Foreign word",
    "IN": "Preposition or subordinating conjunction (in, of, at, on)",
    "JJ": "Adjective (good, terrible, fast)",
    "JJR": "Adjective, comparative (better, faster)",
    "JJS": "Adjective, superlative (best, fastest)",
    "LS": "List item marker",
    "MD": "Modal auxiliary (can, could, will, should)",
    "NN": "Noun, singular or mass (service, product, store)",
    "NNS": "Noun, plural (services, products, reviews)",
    "NNP": "Proper noun, singular (Amazon, Paris, Twitter)",
    "NNPS": "Proper noun, plural (Americans, Airlines)",
    "PDT": "Predeterminer (all, both, half)",
    "POS": "Possessive ending ('s)",
    "PRP": "Personal pronoun (I, you, he, they, it)",
    "PRP$": "Possessive pronoun (my, your, their, its)",
    "RB": "Adverb (very, extremely, slowly, not)",
    "RBR": "Adverb, comparative (better, faster)",
    "RBS": "Adverb, superlative (best, fastest)",
    "RP": "Particle (give up, take off)",
    "TO": "to (to go, to do)",
    "UH": "Interjection (oh, wow, oops)",
    "VB": "Verb, base form (love, arrive, serve)",
    "VBD": "Verb, past tense (loved, arrived, served)",
    "VBG": "Verb, gerund / present participle (loving, arriving)",
    "VBN": "Verb, past participle (loved, arrived, served)",
    "VBP": "Verb, non-3rd person singular present (love, arrive)",
    "VBZ": "Verb, 3rd person singular present (loves, arrives)",
    "WDT": "Wh-determiner (which, that)",
    "WP": "Wh-pronoun (who, what)",
    "WP$": "Possessive wh-pronoun (whose)",
    "WRB": "Wh-adverb (where, when, why, how)"
}


def extract_pos_tags(text: str) -> List[Dict[str, str]]:
    """
    Assigns grammatical Part-of-Speech (POS) tags to each token.
    Uses NLTK pos_tag with spaCy fallback, returning token, tag, and linguistic category.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    tagged_tokens = []

    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
            nltk_tags = pos_tag(tokens)
            for word, tag in nltk_tags:
                desc = PENN_TAG_EXPLANATIONS.get(tag, "Grammatical marker")
                # Group into universal high-level category
                if tag.startswith("NN"):
                    category = "Noun"
                elif tag.startswith("VB"):
                    category = "Verb"
                elif tag.startswith("JJ"):
                    category = "Adjective"
                elif tag.startswith("RB"):
                    category = "Adverb"
                elif tag.startswith("PRP"):
                    category = "Pronoun"
                elif tag == "DT":
                    category = "Determiner"
                elif tag == "IN":
                    category = "Preposition"
                else:
                    category = "Other"

                tagged_tokens.append({
                    "token": word,
                    "pos_tag": tag,
                    "category": category,
                    "description": desc
                })
            return tagged_tokens
        except Exception:
            pass

    # Fallback to spaCy if NLTK tagger is unavailable
    if SPACY_AVAILABLE and nlp is not None:
        doc = nlp(text)
        for token in doc:
            tagged_tokens.append({
                "token": token.text,
                "pos_tag": token.tag_,
                "category": token.pos_.capitalize(),
                "description": PENN_TAG_EXPLANATIONS.get(token.tag_, spacy.explain(token.tag_) or "Part of speech")
            })
        return tagged_tokens

    # Minimal regex fallback
    words = re.findall(r"\b\w+\b", text)
    return [{"token": w, "pos_tag": "UNK", "category": "Word", "description": "Token"} for w in words]


def extract_syntactic_chunks(text: str) -> Dict[str, Any]:
    """
    Performs Shallow Parsing / Chunking on customer feedback.
    Extracts meaningful multi-word grammatical phrases:
    - Noun Phrases (NP): Target entities/subjects (e.g. 'the customer support', 'subpar quality')
    - Verb Phrases (VP): Customer actions/experiences (e.g. 'was disappointed', 'arrived damaged')
    """
    if not isinstance(text, str) or not text.strip():
        return {"chunks": [], "noun_phrases": [], "verb_phrases": []}

    chunks = []
    noun_phrases = []
    verb_phrases = []

    # Approach 1: NLTK RegexpParser (Classic Lab grammar)
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)

            # Grammar for Noun Phrases and Verb Phrases
            grammar = r"""
              NP: {<DT>?<JJ.*>*<NN.*>+}
              VP: {<VB.*><RB.*>*<VB.*>*}
            """
            cp = RegexpParser(grammar)
            tree = cp.parse(tagged)

            for subtree in tree.subtrees():
                if subtree.label() in ("NP", "VP"):
                    phrase_tokens = [w for w, t in subtree.leaves()]
                    phrase = " ".join(phrase_tokens)
                    label = subtree.label()
                    chunk_type = "Noun Phrase (Target Subject)" if label == "NP" else "Verb Phrase (Action/State)"
                    chunks.append({
                        "phrase": phrase,
                        "type": label,
                        "description": chunk_type,
                        "tokens": phrase_tokens
                    })
                    if label == "NP":
                        noun_phrases.append(phrase)
                    else:
                        verb_phrases.append(phrase)

            return {
                "chunks": chunks,
                "noun_phrases": noun_phrases,
                "verb_phrases": verb_phrases,
                "engine": "NLTK RegexpParser Chunking"
            }
        except Exception:
            pass

    # Approach 2: spaCy Noun Chunks
    if SPACY_AVAILABLE and nlp is not None:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            chunks.append({
                "phrase": chunk.text,
                "type": "NP",
                "description": "Noun Phrase (Target Subject)",
                "tokens": [t.text for t in chunk]
            })
            noun_phrases.append(chunk.text)

        return {
            "chunks": chunks,
            "noun_phrases": noun_phrases,
            "verb_phrases": verb_phrases,
            "engine": "spaCy Dependency Noun Chunks"
        }

    return {"chunks": [], "noun_phrases": [], "verb_phrases": [], "engine": "None"}


# =============================================================================
# NAMED ENTITY RECOGNITION (NER)
# =============================================================================

def extract_entities_spacy(text: str) -> Dict[str, Any]:
    """
    Extracts entities using spaCy's trained statistical pipeline (en_core_web_sm).
    Returns entity tokens, labels, explanations, and displaCy HTML.
    """
    if not SPACY_AVAILABLE or nlp is None:
        return {
            "success": False,
            "entities": [],
            "html": "",
            "engine": "spaCy (Unavailable)",
            "message": "spaCy model 'en_core_web_sm' is not loaded. Using fallback regex extractor."
        }

    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        explanation = spacy.explain(ent.label_) or "Named entity"
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "description": explanation
        })

    try:
        html = spacy.displacy.render(doc, style="ent", page=False)
    except Exception:
        html = ""

    return {
        "success": True,
        "entities": entities,
        "html": html,
        "engine": "spaCy en_core_web_sm (Statistical NLP)",
        "message": f"Successfully extracted {len(entities)} entities using spaCy."
    }


def extract_entities_fallback(text: str) -> Dict[str, Any]:
    """
    Lightweight rule-based fallback entity extractor using regular expressions.
    Used when spaCy is unavailable or for comparative lab demonstration.
    """
    entities = []

    # Currency / Money values
    for m in re.finditer(r"(\$|€|£|₹|USD|EUR|INR)\s?(\d+(?:\.\d{2})?)", text, re.IGNORECASE):
        entities.append({
            "text": m.group(0),
            "label": "MONEY",
            "start": m.start(),
            "end": m.end(),
            "description": "Monetary value or currency (Regex heuristic)"
        })

    # Dates
    for m in re.finditer(r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", text):
        entities.append({
            "text": m.group(0),
            "label": "DATE",
            "start": m.start(),
            "end": m.end(),
            "description": "Calendar date (Regex heuristic)"
        })

    # Email addresses
    for m in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
        entities.append({
            "text": m.group(0),
            "label": "EMAIL",
            "start": m.start(),
            "end": m.end(),
            "description": "Email address (Regex heuristic)"
        })

    # User IDs / Twitter handles
    for m in re.finditer(r"@[A-Za-z0-9_]{3,15}\b", text):
        entities.append({
            "text": m.group(0),
            "label": "USER_HANDLE",
            "start": m.start(),
            "end": m.end(),
            "description": "User account handle (Regex heuristic)"
        })

    # Order IDs / Tracking Codes
    for m in re.finditer(r"\b(?:order|id|tracking|ref)[#:\s-]*([A-Z0-9]{4,12})\b", text, re.IGNORECASE):
        entities.append({
            "text": m.group(0),
            "label": "ORDER_ID",
            "start": m.start(),
            "end": m.end(),
            "description": "Order or Tracking Identifier (Regex heuristic)"
        })

    return {
        "success": True,
        "entities": entities,
        "html": "",
        "engine": "Regex Pattern Matching (Fallback Heuristic)",
        "message": f"Identified {len(entities)} pattern-matched entities via regex rules."
    }


def analyze_entities(text: str, force_fallback: bool = False) -> Dict[str, Any]:
    """
    Unified entry point for entity extraction.
    Defaults to spaCy statistical NER; switches to fallback if requested or unavailable.
    """
    if force_fallback or not SPACY_AVAILABLE:
        return extract_entities_fallback(text)
    return extract_entities_spacy(text)


# =============================================================================
# MORPHOLOGICAL ANALYSIS
# =============================================================================

COMMON_PREFIXES = {
    "un": {"type": "Derivational", "meaning": "Not / Opposite of (e.g. unhappy, unlock)"},
    "dis": {"type": "Derivational", "meaning": "Not / Absence of / Reversal (e.g. disappoint, disconnect)"},
    "re": {"type": "Derivational", "meaning": "Again / Back (e.g. redo, rebuild, reload)"},
    "pre": {"type": "Derivational", "meaning": "Before (e.g. preview, prepay, preheat)"},
    "mis": {"type": "Derivational", "meaning": "Badly / Wrongly (e.g. mislead, misunderstand)"},
    "in": {"type": "Derivational", "meaning": "Not / Without (e.g. inactive, incomplete)"},
    "im": {"type": "Derivational", "meaning": "Not / Opposite (e.g. impossible, imperfect)"},
    "non": {"type": "Derivational", "meaning": "Not / Absence (e.g. nonstop, nonfunctional)"},
    "sub": {"type": "Derivational", "meaning": "Under / Below (e.g. subpar, substandard)"},
    "over": {"type": "Derivational", "meaning": "Too much / Excess (e.g. overcook, overpriced)"},
    "under": {"type": "Derivational", "meaning": "Below / Insufficient (e.g. underpaid, undercooked)"}
}

COMMON_SUFFIXES = {
    "ing": {"type": "Inflectional", "category": "Present Participle / Gerund", "example": "play -> playing"},
    "ed": {"type": "Inflectional", "category": "Past Tense / Past Participle", "example": "play -> played"},
    "es": {"type": "Inflectional", "category": "Plural / 3rd Person Present", "example": "watch -> watches"},
    "s": {"type": "Inflectional", "category": "Plural / 3rd Person Present", "example": "book -> books, play -> plays"},
    "er": {"type": "Derivational / Inflectional", "category": "Comparative / Agent Noun", "example": "fast -> faster, play -> player"},
    "est": {"type": "Inflectional", "category": "Superlative", "example": "fast -> fastest"},
    "ly": {"type": "Derivational", "category": "Adverbial Suffix", "example": "quick -> quickly, happy -> happily"},
    "tion": {"type": "Derivational", "category": "Action / State Noun", "example": "connect -> connection"},
    "ation": {"type": "Derivational", "category": "Action / State Noun", "example": "inform -> information"},
    "ment": {"type": "Derivational", "category": "Action / Result Noun", "example": "enjoy -> enjoyment"},
    "ness": {"type": "Derivational", "category": "State / Quality Noun", "example": "happy -> happiness"},
    "able": {"type": "Derivational", "category": "Capable of being", "example": "accept -> acceptable"},
    "ible": {"type": "Derivational", "category": "Capable of being", "example": "access -> accessible"},
    "ful": {"type": "Derivational", "category": "Full of (Adjective)", "example": "help -> helpful, wonder -> wonderful"},
    "less": {"type": "Derivational", "category": "Without (Adjective)", "example": "help -> helpless, taste -> tasteless"}
}


def analyze_morphology(word: str) -> Dict[str, Any]:
    """
    Performs morphological decomposition on an English word:
    - Identifies Free Morphemes (root/base that stands alone).
    - Identifies Bound Morphemes (prefixes and suffixes).
    - Classifies the morphological process (Inflectional vs Derivational).
    """
    word_clean = word.strip().lower()
    if not word_clean or not word_clean.isalpha():
        return {
            "original_word": word,
            "root_candidate": word_clean,
            "free_morpheme": word_clean,
            "bound_morphemes": [],
            "prefix": None,
            "suffix": None,
            "morphology_type": "None / Non-alphabetic",
            "explanation": "Please enter a single valid English word containing only alphabets."
        }

    detected_prefix = None
    detected_suffix = None
    prefix_info = None
    suffix_info = None
    working_stem = word_clean
    bound_morphemes = []

    # Check prefixes
    sorted_prefixes = sorted(COMMON_PREFIXES.keys(), key=len, reverse=True)
    for pfx in sorted_prefixes:
        if working_stem.startswith(pfx) and len(working_stem) > len(pfx) + 2:
            detected_prefix = pfx
            prefix_info = COMMON_PREFIXES[pfx]
            bound_morphemes.append(f"Prefix '{pfx}-' (Bound Morpheme, {prefix_info['type']})")
            working_stem = working_stem[len(pfx):]
            break

    # Check suffixes
    sorted_suffixes = sorted(COMMON_SUFFIXES.keys(), key=len, reverse=True)
    for sfx in sorted_suffixes:
        if working_stem.endswith(sfx) and len(working_stem) > len(sfx) + 2:
            detected_suffix = sfx
            suffix_info = COMMON_SUFFIXES[sfx]
            bound_morphemes.append(f"Suffix '-{sfx}' (Bound Morpheme, {suffix_info['type']})")
            candidate_stem = working_stem[:-len(sfx)]

            # Reconstruct root
            if candidate_stem.endswith("i") and len(candidate_stem) > 2:
                candidate_stem = candidate_stem[:-1] + "y"
            if len(candidate_stem) >= 4 and candidate_stem[-1] == candidate_stem[-2] and candidate_stem[-1] not in "ls":
                candidate_stem = candidate_stem[:-1]

            working_stem = candidate_stem
            break

    # Free vs Bound classification
    free_morpheme = working_stem

    morph_type = "Base Free Morpheme (Monomorphemic)"
    if detected_prefix and detected_suffix:
        morph_type = f"Combined ({prefix_info['type']} Prefix + {suffix_info['type']} Suffix)"
    elif detected_prefix:
        morph_type = f"Prefixation ({prefix_info['type']})"
    elif detected_suffix:
        morph_type = f"Suffixation ({suffix_info['type']})"

    explanation_lines = []
    explanation_lines.append(f"Free Morpheme (Root/Stem): '{free_morpheme}'")
    if detected_prefix:
        explanation_lines.append(f"Bound Prefix '{detected_prefix}-': {prefix_info['meaning']}")
    if detected_suffix:
        explanation_lines.append(f"Bound Suffix '-{detected_suffix}': {suffix_info['category']} ({suffix_info['example']})")
    if not detected_prefix and not detected_suffix:
        explanation_lines.append(f"'{word_clean}' consists of a single root/free morpheme.")

    return {
        "original_word": word,
        "root_candidate": free_morpheme,
        "free_morpheme": free_morpheme,
        "bound_morphemes": bound_morphemes if bound_morphemes else ["None"],
        "prefix": detected_prefix,
        "prefix_info": prefix_info,
        "suffix": detected_suffix,
        "suffix_info": suffix_info,
        "morphology_type": morph_type,
        "explanation": " | ".join(explanation_lines)
    }


# =============================================================================
# WORD GENERATION ENGINE
# =============================================================================

def generate_word_forms(base_word: str) -> Dict[str, Any]:
    """
    Generates inflectional and derivational forms from a base English root word.
    Applies standard linguistic orthographic rules:
    1. Consonant + y -> -ies (present), -ied (past), -ying (participle)
    2. Sibilant sounds (-s, -ss, -sh, -ch, -x, -z) -> -es
    3. Silent 'e' dropping before vowel suffixes (-ing, -ed)
    4. CVC Consonant Doubling for short vowel monosyllabic bases (e.g. stop -> stopped, stopping)
    """
    word = base_word.strip().lower()
    if not word or not word.isalpha():
        return {
            "base_word": base_word,
            "forms": {},
            "rules_applied": ["Input must be a single alphabetic word."]
        }

    rules_applied = []
    vowels = "aeiou"

    # 1. 3rd Person Present / Plural (-s / -es / -ies)
    if word.endswith("y") and len(word) > 1 and word[-2] not in vowels:
        form_3rd = word[:-1] + "ies"
        rules_applied.append("Consonant + 'y' Rule: replaced 'y' with 'ies' for 3rd person singular / plural.")
    elif word.endswith(("s", "ss", "sh", "ch", "x", "z", "o")):
        form_3rd = word + "es"
        rules_applied.append("Sibilant/Vowel 'o' Rule: added '-es' for smooth phonetic articulation.")
    else:
        form_3rd = word + "s"
        rules_applied.append("Standard Rule: added inflectional '-s'.")

    # 2. Past Tense / Participle (-ed / -ied / doubled consonant + ed)
    is_cvc = (
        len(word) >= 3
        and word[-1] not in vowels
        and word[-1] not in "wxy"
        and word[-2] in vowels
        and word[-3] not in vowels
    )

    if word.endswith("e"):
        form_past = word + "d"
        rules_applied.append("Silent 'e' Rule: added only '-d' for past tense.")
    elif word.endswith("y") and len(word) > 1 and word[-2] not in vowels:
        form_past = word[:-1] + "ied"
        rules_applied.append("Consonant + 'y' Rule: replaced 'y' with 'ied' for past tense.")
    elif is_cvc:
        form_past = word + word[-1] + "ed"
        rules_applied.append(f"CVC Consonant Doubling Rule: doubled final consonant '{word[-1]}' before '-ed'.")
    else:
        form_past = word + "ed"
        rules_applied.append("Standard Rule: added inflectional '-ed' for past tense.")

    # 3. Present Participle / Gerund (-ing)
    if word.endswith("ie"):
        form_ing = word[:-2] + "ying"
        rules_applied.append("'-ie' to '-ying' Rule (e.g. lie -> lying).")
    elif word.endswith("e") and not word.endswith(("ee", "oe", "ye")):
        form_ing = word[:-1] + "ing"
        rules_applied.append("Silent 'e' Dropping Rule: dropped 'e' before '-ing'.")
    elif is_cvc:
        form_ing = word + word[-1] + "ing"
        rules_applied.append(f"CVC Consonant Doubling Rule: doubled final consonant '{word[-1]}' before '-ing'.")
    else:
        form_ing = word + "ing"
        rules_applied.append("Standard Rule: appended inflectional '-ing'.")

    # 4. Agent Noun / Comparative (-er)
    if word.endswith("e"):
        form_er = word + "r"
    elif word.endswith("y") and len(word) > 1 and word[-2] not in vowels:
        form_er = word[:-1] + "ier"
    elif is_cvc:
        form_er = word + word[-1] + "er"
    else:
        form_er = word + "er"

    # 5. Derivational Forms
    derivations = {}
    if word.startswith(("b", "m", "p")):
        derivations["Negative Form (Derivational)"] = f"im{word} / un{word}"
    elif word.startswith(("l",)):
        derivations["Negative Form (Derivational)"] = f"il{word} / un{word}"
    elif word.startswith(("r",)):
        derivations["Negative Form (Derivational)"] = f"ir{word} / un{word}"
    else:
        derivations["Negative Form (Derivational)"] = f"un{word} / dis{word}"

    if word.endswith("y") and len(word) > 1 and word[-2] not in vowels:
        derivations["Adverbial Form (-ly)"] = word[:-1] + "ily"
    elif word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        derivations["Adverbial Form (-ly)"] = word[:-1] + "y"
    else:
        derivations["Adverbial Form (-ly)"] = word + "ly"

    if word.endswith("y") and len(word) > 1 and word[-2] not in vowels:
        derivations["State Noun (-ness)"] = word[:-1] + "iness"
    else:
        derivations["State Noun (-ness)"] = word + "ness"

    forms = {
        "Base / Root": word,
        "3rd Person Present / Plural (-s/-es/-ies)": form_3rd,
        "Past Tense / Participle (-ed/-ied)": form_past,
        "Present Participle / Gerund (-ing)": form_ing,
        "Agent Noun / Comparative (-er)": form_er,
        **derivations
    }

    return {
        "base_word": word,
        "forms": forms,
        "rules_applied": rules_applied
    }
