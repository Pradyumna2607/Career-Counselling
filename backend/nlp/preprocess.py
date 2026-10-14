"""
preprocess.py
--------------
Text preprocessing utilities for the AI Career Counselling chatbot's NLP pipeline.

Pipeline stages performed here:
    1. Lowercasing
    2. Punctuation / special character removal
    3. Tokenization
    4. Stopword removal
    5. Lightweight suffix-stripping ("lemmatization")

Design note: this module is intentionally dependency-free (no NLTK corpus
downloads required) so the project runs out-of-the-box on any machine
(including offline / firewalled environments) without extra setup steps -
a common pain point students hit with NLTK's `punkt`/`wordnet` downloads.
The same clean_text/preprocess functions are used at both training time
(train_intent_models.py) and inference time (chat_engine.py) so features
stay consistent.
"""

import re

# A compact, curated English stopword list. Question/negation words that
# carry intent meaning (what, which, no, not, ...) are deliberately kept.
_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "this", "that", "these", "those",
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "do", "does", "did", "doing", "have", "has", "had", "having", "would",
    "could", "im", "ive", "id", "youre", "please", "kindly", "also",
}
_KEEP_WORDS = {"no", "not", "what", "which", "who", "how", "why", "when", "cant", "dont"}
_STOPWORDS = _STOPWORDS - _KEEP_WORDS

# Small suffix-stripping table for lightweight normalization
# (e.g. "engineering" -> "engineer", "designing" -> "design")
_SUFFIX_RULES = [
    ("ising", "ise"), ("izing", "ize"),
    ("ational", "ate"), ("tional", "tion"),
    ("ies", "y"), ("ing", ""), ("edly", ""), ("ed", ""),
    ("ness", ""), ("fulness", "ful"), ("ations", "ation"),
    ("ers", "er"), ("es", ""), ("s", ""),
]


def clean_text(text: str) -> str:
    """Lowercase and strip punctuation / digits / extra whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    """Simple, dependency-free whitespace tokenizer over cleaned text."""
    return text.split()


def remove_stopwords(tokens):
    return [t for t in tokens if t not in _STOPWORDS]


def _stem_word(word: str) -> str:
    if len(word) <= 5:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + replacement
    return word


def lemmatize(tokens):
    """Lightweight rule-based normalization (suffix stripping)."""
    return [_stem_word(t) for t in tokens]


def preprocess(text: str) -> str:
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> normalize
    Returns a single space-joined string ready for vectorization (TF-IDF / BoW).
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "I got 50 percentile in MHT-CET did I get admission in SB Jain in CSE branch?",
        "Suggest me a career, I love coding and mathematics!!",
        "What qualification do I need to become a data scientist?",
    ]
    for s in samples:
        print(f"{s!r} -> {preprocess(s)!r}")
