"""
recommendation_engine.py
--------------------------
Ranked career recommendation engine.

Given free-text describing a user's interests/skills/stream (already
run through the NLP pipeline for intent + entity extraction), this module
scores every career in the knowledge base (careers.csv) against the user's
profile using TF-IDF + cosine similarity over each career's combined
"interest_keywords + required_skills + description" text, then re-ranks
using a small set of business rules (stream match boost, entity-keyword
overlap boost) before returning the top-N results.

This is intentionally a *content-based* recommender (no user-user
collaborative data is required), which is the right fit for a cold-start
chatbot where every conversation may be a first-time user.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.preprocess import preprocess

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CAREERS_CSV = os.path.join(DATA_DIR, "careers.csv")


class RecommendationEngine:
    def __init__(self, careers_csv: str = CAREERS_CSV):
        self.df = pd.read_csv(careers_csv)
        self.df["combined_text"] = (
            self.df["interest_keywords"].fillna("") + " " +
            self.df["required_skills"].fillna("") + " " +
            self.df["description"].fillna("") + " " +
            self.df["domain"].fillna("")
        )
        self.df["combined_clean"] = self.df["combined_text"].apply(preprocess)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.career_matrix = self.vectorizer.fit_transform(self.df["combined_clean"])

    def recommend(self, user_text: str, stream: str = None, top_n: int = 3) -> list:
        """
        Returns a ranked list of dicts:
            [{career_name, score, stream, domain, description, qualification,
              entrance_exams, avg_salary_lpa, growth_outlook, matched_keywords}, ...]
        sorted by descending relevance score.
        """
        if not user_text or not user_text.strip():
            return []

        clean_query = preprocess(user_text)
        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.career_matrix).flatten()

        scored = self.df.copy()
        scored["similarity"] = similarities

        # Rule-based re-ranking boosts on top of the raw cosine similarity:
        #  (a) +0.15 if the user's declared stream matches the career's stream
        #  (b) small boost proportional to raw keyword overlap count, which
        #      helps short queries where TF-IDF alone is noisy
        query_tokens = set(clean_query.split())

        def keyword_overlap(row):
            career_tokens = set(preprocess(row["interest_keywords"]).split())
            return len(query_tokens & career_tokens)

        scored["keyword_overlap"] = scored.apply(keyword_overlap, axis=1)

        def stream_bonus(row_stream):
            if not stream:
                return 0.0
            return 0.15 if stream.lower() in str(row_stream).lower() else 0.0

        scored["stream_bonus"] = scored["stream"].apply(stream_bonus)
        scored["final_score"] = (
            scored["similarity"] +
            (scored["keyword_overlap"] * 0.05) +
            scored["stream_bonus"]
        )

        ranked = scored.sort_values("final_score", ascending=False).head(top_n)

        results = []
        for _, row in ranked.iterrows():
            if row["final_score"] <= 0:
                continue
            results.append({
                "career_name": row["career_name"],
                "score": round(float(row["final_score"]), 4),
                "stream": row["stream"],
                "domain": row["domain"],
                "description": row["description"],
                "required_skills": row["required_skills"],
                "qualification": row["qualification"],
                "entrance_exams": row["entrance_exams"],
                "avg_salary_lpa": row["avg_salary_lpa"],
                "growth_outlook": row["growth_outlook"],
            })
        return results

    def lookup_career(self, career_name: str):
        """Exact/fuzzy single-career lookup used for detail/qualification/exam intents."""
        matches = self.df[self.df["career_name"].str.lower() == career_name.lower()]
        if matches.empty:
            matches = self.df[self.df["career_name"].str.lower().str.contains(career_name.lower())]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return row.to_dict()


if __name__ == "__main__":
    engine = RecommendationEngine()
    results = engine.recommend("I love coding, mathematics and solving logical problems", stream="science")
    for r in results:
        print(f"{r['career_name']:30s} score={r['score']}")
