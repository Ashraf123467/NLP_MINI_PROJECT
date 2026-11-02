"""
Smart Customer Feedback & Sentiment Analyzer - Lab Features (lab_features.py)

Explicit implementations for:
1. NLP Lab Experiment 05: Named Entity Recognition (NER)
   - spaCy-powered NER extraction and visual rendering.
   - Transparent rule-based fallback NER (regex).
2. NLP Lab Experiment 06: Morphological Analysis and Word Generation
   - Morphological dissection: prefix, root/stem, suffix.
   - Classification into Inflectional vs. Derivational morphology.
   - Linguistic word generation applying standard English orthographic rules:
     * CVC consonant doubling (e.g. stop -> stopped, stopping)
     * Consonant + 'y' modification (e.g. study -> studies, studied)
     * Silent 'e' dropping (e.g. create -> created, creating)
     * Sibilant pluralization (-s, -es, -ies)
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
        # Model not found or could not load
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False


# =============================================================================
# NLP LAB EXPERIMENT 05 — NAMED ENTITY RECOGNITION (NER)
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

    # Generate displaCy HTML
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
    NOTE: Transparently labeled as regex pattern matching, NOT a statistical NER model.
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
    Defaults to spaCy statistical NER; gracefully switches to fallback if requested or unavailable.
    """
    if force_fallback or not SPACY_AVAILABLE:
        return extract_entities_fallback(text)
    return extract_entities_spacy(text)


# =============================================================================
# NLP LAB EXPERIMENT 06 — MORPHOLOGICAL ANALYSIS & WORD GENERATION
# =============================================================================

# Known morphological affixes for academic demonstration
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
    - Identifies prefix, stem/root candidate, and suffix.
    - Classifies the morphological process (Inflectional vs Derivational).
    - Explains morphological rules for academic viva.
    """
    word_clean = word.strip().lower()
    if not word_clean or not word_clean.isalpha():
        return {
            "original_word": word,
            "root": word_clean,
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

    # Check prefixes (longest match first)
    sorted_prefixes = sorted(COMMON_PREFIXES.keys(), key=len, reverse=True)
    for pfx in sorted_prefixes:
        if working_stem.startswith(pfx) and len(working_stem) > len(pfx) + 2:
            detected_prefix = pfx
            prefix_info = COMMON_PREFIXES[pfx]
            working_stem = working_stem[len(pfx):]
            break

    # Check suffixes (longest match first)
    sorted_suffixes = sorted(COMMON_SUFFIXES.keys(), key=len, reverse=True)
    for sfx in sorted_suffixes:
        if working_stem.endswith(sfx) and len(working_stem) > len(sfx) + 2:
            detected_suffix = sfx
            suffix_info = COMMON_SUFFIXES[sfx]
            candidate_stem = working_stem[:-len(sfx)]

            # Reconstruct potential root based on spelling rules
            # e.g., 'happi' from 'happiness' -> 'happy'
            if candidate_stem.endswith("i") and len(candidate_stem) > 2:
                candidate_stem = candidate_stem[:-1] + "y"
            # e.g., 'creati' from 'creating' -> 'create'
            elif sfx in ("ing", "ed", "able") and len(candidate_stem) >= 3:
                # Test if silent 'e' was dropped
                if not candidate_stem.endswith("e") and candidate_stem[-1] in "bcdfgptmnszv":
                    # Keep both or add e candidate
                    pass
            # e.g., doubled consonant: 'stopp' from 'stopped' -> 'stop'
            if len(candidate_stem) >= 4 and candidate_stem[-1] == candidate_stem[-2] and candidate_stem[-1] not in "ls":
                candidate_stem = candidate_stem[:-1]

            working_stem = candidate_stem
            break

    # Determine morphological category
    morph_type = "Base / Root Word"
    if detected_prefix and detected_suffix:
        morph_type = f"Combined ({prefix_info['type']} Prefix + {suffix_info['type']} Suffix)"
    elif detected_prefix:
        morph_type = f"Prefixation ({prefix_info['type']})"
    elif detected_suffix:
        morph_type = f"Suffixation ({suffix_info['type']})"

    explanation_lines = []
    if detected_prefix:
        explanation_lines.append(f"Prefix '{detected_prefix}-': {prefix_info['meaning']}")
    if detected_suffix:
        explanation_lines.append(f"Suffix '-{detected_suffix}': {suffix_info['category']} ({suffix_info['example']})")
    if not detected_prefix and not detected_suffix:
        explanation_lines.append(f"'{word_clean}' functions as a base/monomorphemic root or uses irregular morphology.")

    return {
        "original_word": word,
        "root_candidate": working_stem,
        "prefix": detected_prefix,
        "prefix_info": prefix_info,
        "suffix": detected_suffix,
        "suffix_info": suffix_info,
        "morphology_type": morph_type,
        "explanation": " | ".join(explanation_lines)
    }


def generate_word_forms(base_word: str) -> Dict[str, Any]:
    """
    Generates inflectional and derivational forms from a base English word.
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

    # --- 1. Plural / 3rd Person Singular Present (-s / -es / -ies) ---
    if word.endswith("y") and len(word) > 1 and word[-2] not in vowels:
        form_3rd = word[:-1] + "ies"
        rules_applied.append("Consonant + 'y' Rule: replaced 'y' with 'ies' for 3rd person singular / plural.")
    elif word.endswith(("s", "ss", "sh", "ch", "x", "z", "o")):
        form_3rd = word + "es"
        rules_applied.append("Sibilant/Vowel 'o' Rule: added '-es' for smooth phonetic articulation.")
    else:
        form_3rd = word + "s"
        rules_applied.append("Standard Rule: added inflectional '-s'.")

    # --- 2. Past Tense / Past Participle (-ed / -ied / doubled consonant + ed) ---
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

    # --- 3. Present Participle / Gerund (-ing) ---
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

    # --- 4. Agent Noun / Comparative (-er) ---
    if word.endswith("e"):
        form_er = word + "r"
    elif word.endswith("y") and len(word) > 1 and word[-2] not in vowels:
        form_er = word[:-1] + "ier"
    elif is_cvc:
        form_er = word + word[-1] + "er"
    else:
        form_er = word + "er"

    # --- 5. Derivational Forms (Negative, Adverb, Abstract Noun) ---
    derivations = {}
    # Negative prefix
    if word.startswith(("b", "m", "p")):
        derivations["Negative Form (Derivational)"] = f"im{word} / un{word}"
    elif word.startswith(("l",)):
        derivations["Negative Form (Derivational)"] = f"il{word} / un{word}"
    elif word.startswith(("r",)):
        derivations["Negative Form (Derivational)"] = f"ir{word} / un{word}"
    else:
        derivations["Negative Form (Derivational)"] = f"un{word} / dis{word}"

    # Adverb form (-ly)
    if word.endswith("y") and len(word) > 1 and word[-2] not in vowels:
        derivations["Adverbial Form (-ly)"] = word[:-1] + "ily"
    elif word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        derivations["Adverbial Form (-ly)"] = word[:-1] + "y"
    else:
        derivations["Adverbial Form (-ly)"] = word + "ly"

    # Quality / State Noun (-ness / -ment / -tion)
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

