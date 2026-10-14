"""
models_db.py
------------
SQLAlchemy ORM models for the AI Career Counselling System.

Tables:
    users          - registered accounts (student / parent / teacher)
    chat_sessions   - one row per conversation thread ("New Chat")
    chat_messages   - individual user/bot messages within a session
    feedback        - thumbs up/down feedback on bot responses (used to
                      report system effectiveness / user satisfaction,
                      mirrored from the original report's evaluation goals)
"""

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(20), default="student")  # student/parent/teacher
    password_hash = db.Column(db.String(256), nullable=False)
    stream = db.Column(db.String(50), nullable=True)     # science/commerce/arts
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    sessions = db.relationship("ChatSession", backref="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role,
            "stream": self.stream,
        }


class ChatSession(db.Model):
    __tablename__ = "chat_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(200), default="New Chat")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Serialized JSON: remembers last discussed career, stream, and interests
    # so follow-up questions ("what about its salary?", "tell me more") don't
    # require the student to repeat themselves.
    context_json = db.Column(db.Text, nullable=True, default="{}")

    messages = db.relationship("ChatMessage", backref="session", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
        }


class ChatMessage(db.Model):
    __tablename__ = "chat_messages"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False)
    sender = db.Column(db.String(10), nullable=False)  # 'user' or 'bot'
    text = db.Column(db.Text, nullable=False)
    intent = db.Column(db.String(50), nullable=True)     # classified intent (bot messages: NA)
    confidence = db.Column(db.Float, nullable=True)      # classifier confidence
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "sender": self.sender,
            "text": self.text,
            "intent": self.intent,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class Feedback(db.Model):
    __tablename__ = "feedback"

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey("chat_messages.id"), nullable=False)
    rating = db.Column(db.String(10), nullable=False)  # 'up' or 'down'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
