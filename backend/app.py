"""
app.py
------
Flask backend for the AI-Based Career Counselling System.

Endpoints:
    POST /api/auth/signup         Register a new user
    POST /api/auth/login          Login, returns session token (Flask session)
    POST /api/auth/logout         Logout
    GET  /api/auth/me             Current logged-in user

    POST /api/chat/sessions       Create a new chat session ("New Chat")
    GET  /api/chat/sessions       List current user's chat sessions
    GET  /api/chat/sessions/<id>/messages   Get messages for a session
    POST /api/chat/message        Send a message, get bot reply (core NLP pipeline)
    DELETE /api/chat/sessions/<id> Delete a chat session

    POST /api/feedback            Submit thumbs up/down on a bot message

    GET  /api/careers             List all careers (for browsing/autocomplete)

    GET  /                        Serves the frontend (chat.html)
"""

import os
import json
from datetime import datetime

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS

from models_db import db, User, ChatSession, ChatMessage, Feedback
from chat_engine import get_engine

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = Flask(
    __name__,
    static_folder=os.path.join(FRONTEND_DIR, "static"),
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "career_counseling.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

CORS(app, supports_credentials=True)
db.init_app(app)

with app.app_context():
    db.create_all()


def _current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return db.session.get(User, uid)


# --------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def serve_frontend_files(path):
    full_path = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(full_path):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


# --------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------- #
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True)
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    role = data.get("role", "student")
    stream = data.get("stream")

    if not full_name or not email or not password:
        return jsonify({"error": "full_name, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "an account with this email already exists"}), 409

    user = User(full_name=full_name, email=email, role=role, stream=stream)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    session["user_id"] = user.id
    return jsonify({"message": "account created", "user": user.to_dict()}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "invalid email or password"}), 401

    session["user_id"] = user.id
    return jsonify({"message": "logged in", "user": user.to_dict()})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return jsonify({"message": "logged out"})


@app.route("/api/auth/me", methods=["GET"])
def me():
    user = _current_user()
    if not user:
        return jsonify({"user": None})
    return jsonify({"user": user.to_dict()})


# --------------------------------------------------------------------- #
# Chat sessions
# --------------------------------------------------------------------- #
@app.route("/api/chat/sessions", methods=["POST"])
def create_session():
    user = _current_user()
    if not user:
        return jsonify({"error": "login required"}), 401

    chat_session = ChatSession(user_id=user.id, title="New Chat")
    db.session.add(chat_session)
    db.session.commit()
    return jsonify({"session": chat_session.to_dict()}), 201


@app.route("/api/chat/sessions", methods=["GET"])
def list_sessions():
    user = _current_user()
    if not user:
        return jsonify({"error": "login required"}), 401

    sessions = (
        ChatSession.query.filter_by(user_id=user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return jsonify({"sessions": [s.to_dict() for s in sessions]})


@app.route("/api/chat/sessions/<int:session_id>", methods=["DELETE"])
def delete_session(session_id):
    user = _current_user()
    if not user:
        return jsonify({"error": "login required"}), 401

    chat_session = ChatSession.query.filter_by(id=session_id, user_id=user.id).first()
    if not chat_session:
        return jsonify({"error": "session not found"}), 404
    db.session.delete(chat_session)
    db.session.commit()
    return jsonify({"message": "deleted"})


@app.route("/api/chat/sessions/<int:session_id>/messages", methods=["GET"])
def get_messages(session_id):
    user = _current_user()
    if not user:
        return jsonify({"error": "login required"}), 401

    chat_session = ChatSession.query.filter_by(id=session_id, user_id=user.id).first()
    if not chat_session:
        return jsonify({"error": "session not found"}), 404

    messages = (
        ChatMessage.query.filter_by(session_id=session_id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
    return jsonify({"messages": [m.to_dict() for m in messages]})


# --------------------------------------------------------------------- #
# Core chat endpoint — runs the NLP pipeline
# --------------------------------------------------------------------- #
@app.route("/api/chat/message", methods=["POST"])
def send_message():
    user = _current_user()
    if not user:
        return jsonify({"error": "login required"}), 401

    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    session_id = data.get("session_id")

    if not text:
        return jsonify({"error": "message text is required"}), 400

    chat_session = None
    if session_id:
        chat_session = ChatSession.query.filter_by(id=session_id, user_id=user.id).first()
    if not chat_session:
        chat_session = ChatSession(user_id=user.id, title=text[:60])
        db.session.add(chat_session)
        db.session.commit()

    # Auto-title the session from the first user message
    if chat_session.title == "New Chat":
        chat_session.title = text[:60]

    user_msg = ChatMessage(session_id=chat_session.id, sender="user", text=text)
    db.session.add(user_msg)
    db.session.commit()

    engine = get_engine()

    try:
        prior_context = json.loads(chat_session.context_json or "{}")
    except (TypeError, ValueError):
        prior_context = {}

    result = engine.get_response(text, user_stream=user.stream, context=prior_context)

    # Persist the updated conversation memory (last career/stream/interests
    # discussed) so the next message in this session doesn't need repeats.
    chat_session.context_json = json.dumps(result.get("context", {}))
    db.session.add(chat_session)

    bot_msg = ChatMessage(
        session_id=chat_session.id,
        sender="bot",
        text=result["reply"],
        intent=result["intent"],
        confidence=result["confidence"],
    )
    db.session.add(bot_msg)
    db.session.commit()

    return jsonify({
        "session": chat_session.to_dict(),
        "user_message": user_msg.to_dict(),
        "bot_message": bot_msg.to_dict(),
        "intent": result["intent"],
        "confidence": result["confidence"],
        "entities": result["entities"],
    })


# --------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------- #
@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    user = _current_user()
    if not user:
        return jsonify({"error": "login required"}), 401

    data = request.get_json(force=True)
    message_id = data.get("message_id")
    rating = data.get("rating")

    if rating not in ("up", "down"):
        return jsonify({"error": "rating must be 'up' or 'down'"}), 400

    fb = Feedback(message_id=message_id, rating=rating)
    db.session.add(fb)
    db.session.commit()
    return jsonify({"message": "feedback recorded"}), 201


# --------------------------------------------------------------------- #
# Careers browse endpoint
# --------------------------------------------------------------------- #
@app.route("/api/careers", methods=["GET"])
def list_careers():
    engine = get_engine()
    careers = engine.recommender.df[[
        "career_name", "stream", "domain", "description",
        "qualification", "avg_salary_lpa", "growth_outlook",
    ]].to_dict(orient="records")
    return jsonify({"careers": careers})


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        port=int(os.environ.get("PORT", 5000)),
        use_reloader=False,
    )
