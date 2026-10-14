"""
entity_extractor.py
--------------------
Rule/dictionary-based Named Entity & keyword extraction for the career chatbot.

This is the second half of the "intent extraction pipeline" mentioned in the
project summary: once chat_engine.py has classified the *intent* of a
message (via Naive Bayes / SVM), this module pulls out the *entities* -
i.e. the concrete slots the response generator needs to fill in, such as:

    - career names   ("data scientist", "mechanical engineer" ...)
    - streams         (science / commerce / arts)
    - interest/skill keywords  (coding, biology, drawing, finance ...)
    - locations        (nagpur, maharashtra, mumbai ...)
    - percentile / score numbers  (for college cutoff style questions)

Entities are matched against the careers.csv knowledge base so that the
vocabulary always stays in sync with the recommendation engine's dataset.
"""

import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CAREERS_CSV = os.path.join(DATA_DIR, "careers.csv")

STREAMS = ["science", "commerce", "arts"]

# A small curated gazetteer of Indian cities/states commonly referenced in
# career/college queries. Extend as needed.
LOCATIONS = [
    "nagpur", "mumbai", "pune", "nashik", "maharashtra", "delhi",
    "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata",
    "surathkal", "karnataka", "gujarat", "indore",
]

# Common synonyms / abbreviations students actually type, mapped to the
# canonical career_name used in careers.csv
CAREER_ALIASES = {
    "medicine": "Doctor (MBBS)",
    "mbbs": "Doctor (MBBS)",
    "doctor": "Doctor (MBBS)",
    "ca": "Chartered Accountant",
    "chartered accountancy": "Chartered Accountant",
    "law": "Lawyer",
    "ias": "Civil Services Officer (IAS/IPS)",
    "ips": "Civil Services Officer (IAS/IPS)",
    "civil services": "Civil Services Officer (IAS/IPS)",
    "ai engineer": "AI/ML Engineer",
    "ml engineer": "AI/ML Engineer",
    "artificial intelligence engineer": "AI/ML Engineer",
    "software engineering": "Software Engineer",
    "software developer": "Software Engineer",
    "coding": "Software Engineer",
    "ux designer": "UX/UI Designer",
    "ui designer": "UX/UI Designer",
    "hr manager": "Human Resource Manager",
    "hr": "Human Resource Manager",
    "entrepreneur": "Entrepreneur/Startup Founder",
    "startup founder": "Entrepreneur/Startup Founder",
    "teaching": "Teacher/Professor",
    "professor": "Teacher/Professor",
    "teacher": "Teacher/Professor",
}


class EntityExtractor:
    def __init__(self, careers_csv: str = CAREERS_CSV):
        self.df = pd.read_csv(careers_csv)
        self.career_names = [c.lower() for c in self.df["career_name"].tolist()]

        # Build a bag of interest/skill keywords from the dataset itself so
        # the extractor's vocabulary is always derived from real data.
        keyword_pool = set()
        for col in ["interest_keywords", "required_skills"]:
            for row in self.df[col].fillna(""):
                for kw in row.split(","):
                    kw = kw.strip().lower()
                    if kw:
                        keyword_pool.add(kw)
        self.keyword_pool = sorted(keyword_pool, key=len, reverse=True)

    def extract(self, text: str) -> dict:
        text_l = text.lower()

        career_names = self._find_multi(text_l, self.career_names)
        # Check aliases too (e.g. "medicine" -> "Doctor (MBBS)")
        for alias, canonical in CAREER_ALIASES.items():
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_l) and canonical.lower() not in career_names:
                career_names.append(canonical.lower())

        entities = {
            "career_names": career_names,
            "streams": self._find_multi(text_l, STREAMS),
            "interest_keywords": self._find_multi(text_l, self.keyword_pool, limit=8),
            "locations": self._find_multi(text_l, LOCATIONS),
            "percentile": self._find_number(text_l, r"(\d{1,3}(?:\.\d+)?)\s*(?:percentile|%|percent)"),
            "cgpa": self._find_number(text_l, r"(\d{1,2}(?:\.\d+)?)\s*(?:cgpa|gpa)"),
        }
        return entities

    @staticmethod
    def _find_multi(text: str, vocabulary, limit: int = None):
        found = []
        for term in vocabulary:
            # word-boundary safe match for short terms, substring for phrases
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, text):
                found.append(term)
        # de-duplicate while preserving order, drop items that are substrings
        # of an already-found longer match (e.g. "design" inside "graphic designer")
        found = sorted(set(found), key=len, reverse=True)
        deduped = []
        for f in found:
            if not any(f in longer and f != longer for longer in deduped):
                deduped.append(f)
        if limit:
            deduped = deduped[:limit]
        return deduped

    @staticmethod
    def _find_number(text: str, pattern: str):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None


if __name__ == "__main__":
    ex = EntityExtractor()
    tests = [
        "I got 50 percentile in MHT-CET, did I get admission in SB Jain in CSE branch?",
        "I love coding and mathematics, suggest a career in technology",
        "Tell me about data scientist salary in Nagpur",
    ]
    for t in tests:
        print(t)
        print(ex.extract(t))
        print()
