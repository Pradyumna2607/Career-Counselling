"""Smoke and behaviour tests for the PathWise backend."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import app as app_module  # noqa: E402
from chat_engine import get_engine  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app_module.app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI="sqlite:///" + str(tmp_path / "test.db"),
    )
    with app_module.app.app_context():
        app_module.db.engine.dispose()
        app_module.db.create_all()
    with app_module.app.test_client() as c:
        yield c


def test_engine_models_loaded():
    assert get_engine().models_ready


def test_intent_greeting():
    intent, conf, _ = get_engine().classify_intent("hello there")
    assert intent == "greeting"
    assert 0.0 <= conf <= 1.0


def test_recommendation_and_memory_follow_up():
    engine = get_engine()
    first = engine.get_response("I love coding and mathematics, suggest a career for me")
    assert first["intent"] == "career_recommendation"
    follow = engine.get_response(
        "what qualification do I need for it", context=first["context"]
    )
    assert follow["reply"]
    assert "Sorry" not in follow["reply"][:10]


def test_careers_endpoint(client):
    r = client.get("/api/careers")
    assert r.status_code == 200
    assert len(r.get_json()["careers"]) >= 30


def test_chat_requires_login(client):
    r = client.post("/api/chat/message", json={"text": "hi"})
    assert r.status_code == 401
