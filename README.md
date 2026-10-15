
# PathWise — AI-Based Career Counselling Chatbot

A full-stack career counselling assistant that understands student questions with a
**Naive Bayes + SVM intent-classification ensemble**, extracts entities with a rule-based pipeline,
ranks careers with a **TF-IDF cosine-similarity recommender**, and keeps **per-session memory**
so follow-ups like *"what about its salary?"* just work.

[![CI](https://github.com/YOUR_USERNAME/PathWise-Career-Counselling/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/PathWise-Career-Counselling/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of contents
- [Features](#features)
- [Demo conversation](#demo-conversation)
- [Architecture](#architecture)
- [Model performance](#model-performance)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Extending](#extending)
- [Testing](#testing)
- [Limitations](#limitations)
- [Contributing, security, license](#contributing-security-license)

## Features
- 18 intents: recommendations, qualifications, exams, salary, skills, future scope, stream advice, small talk, reassurance and more
- Knowledge base of 30 careers (stream, domain, skills, qualification, exams, salary, growth outlook)
- Conversation memory: remembers last career, stream and accumulated interests
- User accounts (Werkzeug password hashing), saved chat sessions, thumbs up/down feedback
- Single Flask process serves both the REST API and the frontend

## Demo conversation
```
You: I love coding and mathematics, suggest a career for me
Bot: Here's what lines up well with your interests:
     1. Software Engineer (Technology) ...
You: what qualification do I need for it
Bot: To become a Software Engineer, you typically need: B.Tech/B.E. in Computer Science or IT.
You: what about its salary
Bot: The average salary for a Software Engineer is around ₹6.5 LPA in India ...
```

## Architecture

```
User message
  -> [1] Preprocess (clean, tokenize, stopwords, normalise)
  -> [2] Intent classification (TF-IDF -> SVM primary, Naive Bayes cross-check)
  -> [3] Entity extraction (careers, stream, skills, location, numbers)
  -> [4] Response generation
        - career_recommendation -> TF-IDF cosine similarity + stream/keyword re-ranking
        - qualification / exam / salary / skills / scope -> lookup in careers.csv
        - chit-chat, reassurance, stream advice -> templated responses
  -> Reply stored with intent + confidence
```

**Why content-based recommendation?** Every student is a cold-start user, so similarity between stated
interests and each career's profile works from the very first message.

## Model performance
Held-out evaluation on `backend/data/intents.csv` (260 labelled utterances, 18 intents, 39 test samples):

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Naive Bayes | 56.4% | 0.516 |
| SVM (linear) | 79.5% | 0.788 |

The test split is small, so treat these numbers as indicative. Adding more utterances per intent is the
most effective way to improve them.

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/PathWise-Career-Counselling.git
cd PathWise-Career-Counselling/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_intent_models.py   # generates backend/models/*.pkl
export SECRET_KEY="change-me"   # Windows: set SECRET_KEY=change-me
python app.py
```

Open http://127.0.0.1:5000, sign up and start chatting.

> The trained `.pkl` files are generated locally and not committed. Only load pickle files you trained yourself.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/signup` | Register |
| POST | `/api/auth/login` | Log in (session cookie) |
| POST | `/api/auth/logout` | Log out |
| GET | `/api/auth/me` | Current user |
| POST | `/api/chat/sessions` | New chat session |
| GET | `/api/chat/sessions` | List sessions |
| GET | `/api/chat/sessions/<id>/messages` | Session messages |
| DELETE | `/api/chat/sessions/<id>` | Delete session |
| POST | `/api/chat/message` | Send message `{text, session_id?}` -> bot reply, intent, confidence, entities |
| POST | `/api/feedback` | Thumbs up/down on a bot message |
| GET | `/api/careers` | Browse the career knowledge base |

## Project structure

```
PathWise-Career-Counselling/
├── backend/
│   ├── app.py  models_db.py  chat_engine.py  train_intent_models.py
│   ├── data/        # careers.csv, intents.csv
│   ├── models/      # generated .pkl files (git-ignored)
│   └── nlp/         # preprocess.py, entity_extractor.py, recommendation_engine.py
├── frontend/        # index.html, chat.html, static/css, static/js
├── tests/           # pytest suite
└── .github/         # CI, issue and PR templates
```

## Extending
- **More careers:** append rows to `backend/data/careers.csv` (same 11 columns, quote fields containing commas).
- **Better intents:** add labelled rows to `backend/data/intents.csv`, then rerun `python train_intent_models.py`.
- **Upgrade models:** `classify_intent()` and `RecommendationEngine` are decoupled from the routes, so a
  transformer classifier or sentence-embedding recommender can be dropped in.
- **Production:** run with gunicorn/waitress, set `SECRET_KEY`, keep `FLASK_DEBUG=0`, and restrict CORS origins in `app.py`.

## Testing
```bash
pip install -r backend/requirements-dev.txt
python backend/train_intent_models.py
pytest -q
```

## Limitations
- Career data is a curated sample focused on the Indian education system; verify salary and exam details before acting on them.
- Intent accuracy is limited by the small training set.
- This is a guidance tool, not a replacement for a human career counsellor.
- SQLite and permissive CORS are suitable for development; harden before public deployment.

## Contributing, security, license
See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [LICENSE](LICENSE) (MIT).
=======
# Career-Counselling
AI career counselling chatbot built with Flask and scikit-learn. Uses a Naive Bayes + SVM intent ensemble, rule-based entity extraction, a TF-IDF cosine-similarity career recommender and per-session memory for natural follow-ups. Covers 30 careers across 18 intents, with user accounts, chat history and feedback.
>>>>>>> 07af4f7a2e6f1dbc533737cdc4d707e84ebc1990
