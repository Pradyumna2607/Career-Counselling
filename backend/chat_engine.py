"""
chat_engine.py
--------------
The modular intent-extraction + response-generation pipeline described in the
project summary:

    User Message
        |
        v
    [1] Preprocessing (nlp/preprocess.py)
        |
        v
    [2] Intent Classification  --  Naive Bayes + SVM ensemble (backend/models/*.pkl)
        |
        v
    [3] Entity Extraction (nlp/entity_extractor.py)
        |
        v
    [4] Conversation Memory merge  (ConversationContext, this file)
        |    - resolves pronouns ("it", "that", "this one") against the last
        |      career discussed in the session
        |    - carries the student's stream/interests forward so they don't
        |      have to repeat themselves every turn
        v
    [5] Response Generator
            -> career_recommendation intents call the Ranked Recommendation
               Engine (nlp/recommendation_engine.py), using accumulated
               interests from earlier in the conversation too
            -> lookup intents (qualification/exam/salary/skills/details)
               query the careers knowledge base, falling back to the
               last-discussed career if none is named this turn
            -> small-talk / conversational intents (greeting/thanks/goodbye/
               how-are-you/jokes/reassurance/...) use varied templated
               responses so the bot feels less repetitive
        |
        v
    Bot Response (+ intent + confidence + updated context, for logging/analytics)

This mirrors the block/system diagrams from the original mini-project report
(Chat Client -> Support Bot -> NLP Tool -> DB Services -> Response) but with
a real trained ML intent classifier, a real content-based recommender, and
lightweight session memory in place of the earlier planned design.
"""

import os
import pickle
import random
import re

from nlp.preprocess import preprocess
from nlp.entity_extractor import EntityExtractor
from nlp.recommendation_engine import RecommendationEngine

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

CONFIDENCE_THRESHOLD = 0.28  # below this, fall back to a clarifying response

# Words that signal the user is referring back to something already
# discussed ("what about ITS salary", "tell me more about THAT one")
_REFERENCE_WORDS = {"it", "its", "that", "this", "this one", "that one", "same", "the same"}

# Intents that need a specific career resolved before they can answer
_CAREER_DEPENDENT_INTENTS = {
    "career_details", "qualification_query", "exam_query",
    "salary_query", "future_scope_query", "skills_query", "follow_up_more",
}


class ConversationContext:
    """
    A small, serializable memory object carried across turns within one
    chat session. Kept intentionally simple (plain dict under the hood) so
    it can be stored as JSON in ChatSession.context_json between requests -
    the Flask app is stateless per-request, so memory has to live somewhere
    durable rather than in Python globals.
    """

    def __init__(self, data: dict = None):
        data = data or {}
        self.last_career = data.get("last_career")
        self.last_stream = data.get("last_stream")
        self.interests = data.get("interests", [])          # accumulated across turns
        self.last_intent = data.get("last_intent")
        self.turn_count = data.get("turn_count", 0)

    def to_dict(self):
        return {
            "last_career": self.last_career,
            "last_stream": self.last_stream,
            "interests": self.interests,
            "last_intent": self.last_intent,
            "turn_count": self.turn_count,
        }

    def remember_career(self, career_name):
        if career_name:
            self.last_career = career_name

    def remember_stream(self, stream):
        if stream:
            self.last_stream = stream

    def remember_interests(self, keywords):
        for kw in keywords or []:
            if kw not in self.interests:
                self.interests.append(kw)
        # keep it bounded so the recommender query doesn't grow unbounded
        self.interests = self.interests[-15:]


class ChatEngine:
    def __init__(self):
        self._load_intent_models()
        self.entity_extractor = EntityExtractor()
        self.recommender = RecommendationEngine()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_intent_models(self):
        def _load(name):
            path = os.path.join(MODELS_DIR, name)
            with open(path, "rb") as f:
                return pickle.load(f)

        try:
            self.vectorizer = _load("tfidf_vectorizer.pkl")
            self.nb_model = _load("naive_bayes_model.pkl")
            self.svm_model = _load("svm_model.pkl")
            self.label_encoder = _load("label_encoder.pkl")
            self.models_ready = True
        except FileNotFoundError:
            self.models_ready = False
            print(
                "[chat_engine] WARNING: trained intent models not found. "
                "Run `python train_intent_models.py` first."
            )

    # ------------------------------------------------------------------ #
    # Intent classification (Naive Bayes + SVM ensemble)
    # ------------------------------------------------------------------ #
    def classify_intent(self, text: str):
        """
        Returns (intent_label, confidence, debug_info) using an ensemble of
        the SVM and Naive Bayes classifiers:
            - SVM's probability estimate is the primary signal (it typically
              generalizes better on short-text intent data).
            - If SVM confidence is low, Naive Bayes' prediction is checked;
              if NB agrees with a reasonably confident vote, we accept it
              too. Otherwise we report low confidence so the response
              generator can fall back to a clarifying message.
        """
        if not self.models_ready:
            return "unknown", 0.0, {}

        clean = preprocess(text)
        X = self.vectorizer.transform([clean])

        svm_probs = self.svm_model.predict_proba(X)[0]
        nb_probs = self.nb_model.predict_proba(X)[0]

        svm_idx = svm_probs.argmax()
        nb_idx = nb_probs.argmax()

        svm_label = self.label_encoder.inverse_transform([svm_idx])[0]
        nb_label = self.label_encoder.inverse_transform([nb_idx])[0]

        svm_conf = float(svm_probs[svm_idx])
        nb_conf = float(nb_probs[nb_idx])

        debug = {
            "svm_label": svm_label, "svm_conf": round(svm_conf, 3),
            "nb_label": nb_label, "nb_conf": round(nb_conf, 3),
        }

        # Ensemble decision
        if svm_conf >= CONFIDENCE_THRESHOLD:
            return svm_label, svm_conf, debug
        if nb_conf >= CONFIDENCE_THRESHOLD and nb_label == svm_label:
            return nb_label, nb_conf, debug
        if nb_conf > svm_conf:
            return nb_label, nb_conf, debug
        return svm_label, svm_conf, debug

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def get_response(self, text: str, user_stream: str = None, context: dict = None) -> dict:
        ctx = ConversationContext(context)
        ctx.turn_count += 1

        intent, confidence, debug = self.classify_intent(text)
        entities = self.entity_extractor.extract(text)

        # Resolve stream: this message > remembered context > user's profile default
        stream = (
            (entities["streams"][0] if entities["streams"] else None)
            or ctx.last_stream
            or user_stream
        )

        if confidence < CONFIDENCE_THRESHOLD:
            reply = self._clarify_with_memory(ctx)
            ctx.last_intent = "unknown"
            return self._package(reply, "unknown", confidence, entities, debug, ctx)

        ctx.remember_stream(stream)
        ctx.remember_interests(entities.get("interest_keywords"))

        handler = getattr(self, f"_handle_{intent}", self._handle_unknown)
        reply = handler(text, entities, stream, ctx)

        # Remember the career this turn resolved to, so the *next* turn can
        # refer back to it with "it"/"that"/etc.
        resolved_career = self._resolve_career(entities, text, ctx, remember=False)
        ctx.remember_career(resolved_career)
        ctx.last_intent = intent

        return self._package(reply, intent, confidence, entities, debug, ctx)

    def _package(self, reply, intent, confidence, entities, debug, ctx):
        return {
            "reply": reply,
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "entities": entities,
            "debug": debug,
            "context": ctx.to_dict(),
        }

    def _clarify_with_memory(self, ctx: ConversationContext) -> str:
        if ctx.last_career:
            return (
                f"I'm not quite sure what you meant. Did you want to know more about "
                f"{ctx.last_career} - like its skills, exams, or salary? Or tell me "
                "something new you're curious about."
            )
        return (
            "I'm not fully sure I understood that. Could you rephrase, or tell me "
            "which stream you're in and what subjects/activities you enjoy? "
            "For example: \"I like coding and maths, what career suits me?\""
        )

    # ------------------------------------------------------------------ #
    # Career resolution (with memory + pronoun handling)
    # ------------------------------------------------------------------ #
    def _mentions_reference_word(self, text: str) -> bool:
        text_l = f" {text.lower()} "
        return any(f" {w} " in text_l for w in _REFERENCE_WORDS)

    def _resolve_career(self, entities, text, ctx: ConversationContext, remember: bool = True):
        if entities["career_names"]:
            career = entities["career_names"][0]
            if remember:
                ctx.remember_career(career)
            return career

        # fallback: try matching any career name as a substring of the raw text
        for name in self.recommender.df["career_name"]:
            if name.lower() in text.lower():
                if remember:
                    ctx.remember_career(name)
                return name

        # Referring back to something already discussed ("its salary?", "tell me more")
        if ctx.last_career and (self._mentions_reference_word(text) or len(text.split()) <= 6):
            return ctx.last_career

        return None

    # ------------------------------------------------------------------ #
    # Intent handlers — conversational / small talk (varied phrasing)
    # ------------------------------------------------------------------ #
    def _handle_greeting(self, text, entities, stream, ctx):
        if ctx.turn_count > 1:
            options = [
                "Hey again! What would you like to explore next?",
                "Hi! Ready to keep going, or is there something new on your mind?",
            ]
        else:
            options = [
                "Hello! I'm your AI Career Counselling Assistant. Tell me your interests, "
                "favourite subjects, or the stream you're in (Science/Commerce/Arts), and "
                "I'll suggest career paths tailored to you.",
                "Hi there! I'm here to help you figure out career options. What subjects or "
                "activities do you enjoy? Or ask me directly about a career's qualifications, "
                "exams, or salary.",
                "Hey! Let's find a career path that fits you. Tell me a bit about your "
                "interests, strengths, or the stream you're studying.",
            ]
        return random.choice(options)

    def _handle_goodbye(self, text, entities, stream, ctx):
        return random.choice([
            "Goodbye! Wishing you clarity and success on your career journey. Come back anytime you need guidance.",
            "Take care! Feel free to come back whenever you want to explore more career options.",
            "Bye for now - good luck, and I'm here whenever you want to pick this back up.",
        ])

    def _handle_thanks(self, text, entities, stream, ctx):
        return random.choice([
            "You're welcome! Happy to help. Is there anything else about your career path I can clarify?",
            "Anytime! Want to dig into another career, or compare a couple of options?",
            "Glad that helped! Let me know if you'd like more details or a different suggestion.",
        ])

    def _handle_about_bot(self, text, entities, stream, ctx):
        return (
            "I'm an AI-powered career counselling chatbot. I use Natural Language "
            "Processing (intent classification with Naive Bayes & SVM, plus keyword/entity "
            "extraction) to understand what you're asking, and a ranked recommendation "
            "engine to match your interests and skills against a database of careers - "
            "covering required qualifications, entrance exams, expected salary, and future outlook. "
            "I also remember what we've talked about in this chat, so you don't have to repeat yourself."
        )

    def _handle_small_talk(self, text, entities, stream, ctx):
        t = text.lower()
        if any(p in t for p in ["how are you", "how're you", "how you doing"]):
            return random.choice([
                "I'm doing well, thanks for asking! More importantly - how are you feeling about your career decisions so far?",
                "Doing great and ready to help! What's on your mind about your future career?",
            ])
        if any(p in t for p in ["your name", "who are you called", "what should i call you"]):
            return "I'm PathWise, your AI career counselling assistant. What's on your mind today?"
        if any(p in t for p in ["joke", "funny"]):
            return random.choice([
                "Why did the career counsellor bring a ladder to work? Because they wanted to help people climb up! 😄 Now, what career are we exploring?",
                "Why don't scientists trust atoms? Because they make up everything - kind of like a good CV does! Speaking of which, want some career suggestions?",
            ])
        if any(p in t for p in ["are you human", "are you a bot", "are you real", "are you ai"]):
            return "I'm an AI assistant, built to help with career guidance using NLP and a recommendation engine. Not human, but genuinely useful for figuring out your next steps!"
        if any(p in t for p in ["who made you", "who built you", "who created you"]):
            return "I was built as a final-year project - an NLP-powered career counselling chatbot. Now, let's put me to work: what are you interested in?"
        return "Good question! I'm mainly here to help with career guidance though - want to tell me your interests or ask about a specific career?"

    def _handle_reassurance(self, text, entities, stream, ctx):
        return (
            "It's completely normal to feel unsure about this - most students do. "
            "Let's break it down step by step: tell me a couple of subjects or activities "
            "you actually enjoy, even small ones, and I'll show you real career paths that "
            "connect to them. We don't have to figure it all out at once."
        )

    def _handle_affirmation(self, text, entities, stream, ctx):
        if ctx.interests:
            return (
                f"Great! Based on what you've told me so far ({', '.join(ctx.interests[-5:])}), "
                "want me to pull together career recommendations now?"
            )
        return "Great! Tell me more about your interests, skills, or the stream you're in, and I'll get you tailored recommendations."

    def _handle_negation(self, text, entities, stream, ctx):
        return "No problem. Let me know whenever you'd like career suggestions or have a question about qualifications, exams, or salaries."

    def _handle_unknown(self, text, entities, stream, ctx):
        return (
            "I'm not fully sure I understood that. You can ask things like:\n"
            "- \"Suggest a career for someone who loves biology\"\n"
            "- \"What qualification do I need for data scientist?\"\n"
            "- \"What is the salary of a mechanical engineer?\""
        )

    # ------------------------------------------------------------------ #
    # Intent handlers — recommendation & lookups (memory-aware)
    # ------------------------------------------------------------------ #
    def _handle_career_recommendation(self, text, entities, stream, ctx):
        query_terms = list(entities.get("interest_keywords") or [])
        # Pull in interests remembered from earlier turns too, so a student
        # who said "I like biology" two messages ago and now says "what about
        # commerce careers" gets recommendations blending both.
        for kw in ctx.interests:
            if kw not in query_terms:
                query_terms.append(kw)

        query_text = text + " " + " ".join(query_terms)
        results = self.recommender.recommend(query_text, stream=stream, top_n=3)

        if not results:
            return (
                "I couldn't find a strong match yet. Could you tell me a bit more - "
                "e.g. subjects you enjoy, hobbies, or skills you're good at?"
            )

        intro = random.choice([
            "Based on what you've shared, here are your top career matches:",
            "Here's what lines up well with your interests:",
            "Nice - here are a few careers worth considering:",
        ])
        lines = [intro + "\n"]
        for i, r in enumerate(results, start=1):
            lines.append(
                f"{i}. {r['career_name']} ({r['domain']}) - match score {r['score']}\n"
                f"   {r['description']}\n"
                f"   Qualification: {r['qualification']}\n"
                f"   Avg. Salary: ₹{r['avg_salary_lpa']} LPA | Outlook: {r['growth_outlook']}\n"
            )
        lines.append(
            "Ask me things like \"tell me more about it\" or \"what exams for the first one\" - "
            "I'll remember which one we're talking about."
        )
        # Remember the top-ranked recommendation so a follow-up like "what
        # qualification do I need for it?" resolves without repeating the name.
        ctx.remember_career(results[0]["career_name"])
        return "\n".join(lines)

    def _handle_career_details(self, text, entities, stream, ctx):
        career = self._resolve_career(entities, text, ctx)
        if not career:
            return "Which career would you like details about? For example: \"tell me about data scientist\"."
        row = self.recommender.lookup_career(career)
        if not row:
            return f"I don't have information on '{career}' yet, but I can recommend similar careers if you share your interests."
        return (
            f"{row['career_name']} ({row['domain']}, {row['stream']} stream)\n"
            f"{row['description']}\n"
            f"Required skills: {row['required_skills']}\n"
            f"Qualification: {row['qualification']}\n"
            f"Entrance exams: {row['entrance_exams']}\n"
            f"Average salary: ₹{row['avg_salary_lpa']} LPA | Growth outlook: {row['growth_outlook']}"
        )

    def _handle_qualification_query(self, text, entities, stream, ctx):
        career = self._resolve_career(entities, text, ctx)
        if not career:
            return "Which career's qualification requirements would you like to know?"
        row = self.recommender.lookup_career(career)
        if not row:
            return f"I don't have qualification details for '{career}' yet."
        return f"To become a {row['career_name']}, you typically need: {row['qualification']}."

    def _handle_exam_query(self, text, entities, stream, ctx):
        career = self._resolve_career(entities, text, ctx)
        if not career:
            return "Which career's entrance exams would you like to know about?"
        row = self.recommender.lookup_career(career)
        if not row:
            return f"I don't have exam details for '{career}' yet."
        return f"Common entrance exams for {row['career_name']}: {row['entrance_exams']}."

    def _handle_salary_query(self, text, entities, stream, ctx):
        career = self._resolve_career(entities, text, ctx)
        if not career:
            return "Which career's salary would you like to know about?"
        row = self.recommender.lookup_career(career)
        if not row:
            return f"I don't have salary data for '{career}' yet."
        return f"The average salary for a {row['career_name']} is around ₹{row['avg_salary_lpa']} LPA in India (varies by experience & location)."

    def _handle_future_scope_query(self, text, entities, stream, ctx):
        career = self._resolve_career(entities, text, ctx)
        if not career:
            return "Which career's future scope would you like me to assess?"
        row = self.recommender.lookup_career(career)
        if not row:
            return f"I don't have outlook data for '{career}' yet."
        return f"The future growth outlook for {row['career_name']} is currently rated: {row['growth_outlook']}."

    def _handle_skills_query(self, text, entities, stream, ctx):
        career = self._resolve_career(entities, text, ctx)
        if not career:
            return "Which career's required skills would you like to know?"
        row = self.recommender.lookup_career(career)
        if not row:
            return f"I don't have skills data for '{career}' yet."
        return f"Key skills for {row['career_name']}: {row['required_skills']}."

    def _handle_follow_up_more(self, text, entities, stream, ctx):
        """'Tell me more' / 'what else' / 'go on' — expands on the last career discussed."""
        career = self._resolve_career(entities, text, ctx)
        if not career:
            if ctx.interests:
                return self._handle_career_recommendation(text, entities, stream, ctx)
            return "Sure - more about what? Tell me a career name, or your interests, and I'll go deeper."
        row = self.recommender.lookup_career(career)
        if not row:
            return f"I don't have more details on '{career}' yet."
        return (
            f"Here's a fuller picture of {row['career_name']}:\n"
            f"{row['description']}\n"
            f"Required skills: {row['required_skills']}\n"
            f"Qualification: {row['qualification']}\n"
            f"Entrance exams: {row['entrance_exams']}\n"
            f"Average salary: ₹{row['avg_salary_lpa']} LPA | Growth outlook: {row['growth_outlook']}\n"
            f"Anything specific you'd like me to expand on further?"
        )

    def _handle_college_query(self, text, entities, stream, ctx):
        loc = f" in {entities['locations'][0].title()}" if entities["locations"] else ""
        return (
            f"I can help point you toward career paths and the exams/cutoffs you'll need to target{loc}, "
            "but for live, year-specific college admission cutoffs I'd recommend checking the official "
            "institute admission portal or your state CET cell, since cutoffs change every year. "
            "Tell me your stream and percentile/score and I can suggest which category of colleges/branches "
            "are typically realistic."
        )

    def _handle_stream_query(self, text, entities, stream, ctx):
        return (
            "Choosing a stream comes down to your interests and strengths:\n"
            "- Science: best if you enjoy logic, experiments, or working with technology/medicine.\n"
            "- Commerce: best if you enjoy numbers, business, or economics.\n"
            "- Arts/Humanities: best if you enjoy writing, design, social sciences, or communication.\n"
            "Tell me a couple of subjects or activities you enjoy and I'll suggest a stream + careers."
        )


# Singleton instance used by the Flask app
_engine_instance = None


def get_engine() -> ChatEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ChatEngine()
    return _engine_instance
