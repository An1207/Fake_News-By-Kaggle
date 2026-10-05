from contextlib import ExitStack

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.pool import StaticPool

from news_api.config import Settings
from news_api.db import Base
from news_api.main import create_app
from news_api.models import Analysis, Article


ARTICLE = {
    "title": "도서관 계획 100% 📰",
    "body": "A fictional council approved a library renovation plan. Construction is scheduled for next year.",
    "language": "en", "source_url": "https://example.com/news",
}


class FakeAI:
    def __init__(self):
        self.calls = []
        self.effect = None

    def analyze(self, title, body, mode, language):
        self.calls.append((title, body, mode, language))
        if self.effect:
            self.effect()
        return {"classification": {"status": "ok", "fake_score": 0.4},
                "summary": {"summary": "테스트용 모의 요약", "model": "test-double"}}


@pytest.fixture
def client_factory():
    with ExitStack() as stack:
        def make(enabled=False):
            # SQLite is only the isolated test fixture; the service uses MySQL.
            engine = create_engine("sqlite://", poolclass=StaticPool,
                                   connect_args={"check_same_thread": False})
            @event.listens_for(engine, "connect")
            def foreign_keys(connection, record):
                connection.execute("PRAGMA foreign_keys=ON")
            Base.metadata.create_all(engine)
            service = FakeAI()
            app = create_app(Settings(_env_file=None, ai_enabled=enabled), engine, service)
            client = stack.enter_context(TestClient(app))
            return client, app, service
        yield make


def create_article(client):
    response = client.post("/api/articles", json=ARTICLE)
    assert response.status_code == 201
    return response.json()


def test_crud_unicode_pagination_and_literal_search(client_factory):
    client, _, _ = client_factory()
    article = create_article(client)
    assert article["title"] == ARTICLE["title"]
    assert article["created_at"].endswith("+00:00")
    assert client.get(f'/api/articles/{article["id"]}').json()["body"] == ARTICLE["body"]
    client.post("/api/articles", json={**ARTICLE, "title": "Different article"})
    result = client.get("/api/articles", params={"q": "%", "page_size": 1}).json()
    assert result["total"] == 1
    assert result["items"][0]["id"] == article["id"]
    assert "body" not in result["items"][0]
    assert client.get("/api/articles", params={"page": 2, "page_size": 1}).json()["items"]
    assert client.get("/api/articles", params={"page": 0}).status_code == 422
    assert client.delete(f'/api/articles/{article["id"]}').status_code == 204
    assert client.get(f'/api/articles/{article["id"]}').status_code == 404


def test_stale_update_cannot_overwrite_newer_version(client_factory):
    client, _, _ = client_factory()
    article = create_article(client)
    url = f'/api/articles/{article["id"]}'
    payload = {**ARTICLE, "title": "Updated", "expected_version": 1}
    assert client.put(url, json=payload).json()["version"] == 2
    response = client.put(url, json={**payload, "title": "Stale"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "VERSION_CONFLICT"
    assert client.get(url).json()["title"] == "Updated"


@pytest.mark.parametrize("change", [{"body": "short"}, {"title": "   "},
                                   {"language": "invalid"}, {"source_url": "file:///secret"},
                                   {"body": "x" * 59001}, {"unexpected": "field"}])
def test_invalid_article_is_not_saved(client_factory, change):
    client, _, _ = client_factory()
    assert client.post("/api/articles", json={**ARTICLE, **change}).status_code == 422
    assert client.get("/api/stats").json()["articles"] == 0


def test_disabled_ai_creates_no_results(client_factory):
    client, _, service = client_factory()
    article = create_article(client)
    assert client.get("/api/health").json()["database"] == "connected"
    assert client.get("/api/capabilities").json()["ai_enabled"] is False
    response = client.post(f'/api/articles/{article["id"]}/analyses', json={"mode": "both"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "AI_NOT_ENABLED"
    assert client.get("/api/analyses").json()["total"] == 0
    assert service.calls == []


def test_job_uses_snapshot_and_cascades_on_delete(client_factory):
    client, app, service = client_factory(True)
    article = create_article(client)
    def edit_during_analysis():
        with app.state.sessions() as session:
            current = session.get(Article, article["id"])
            current.title = "Changed during inference"
            current.body = "Completely changed article content stored while inference is in progress."
            current.version += 1
            session.commit()
    service.effect = edit_during_analysis
    response = client.post(f'/api/articles/{article["id"]}/analyses', json={"mode": "both", "language": "ko"})
    assert response.status_code == 202
    queued = response.json()
    assert queued["status"] == "queued"
    job = client.get(f'/api/analyses/{queued["id"]}').json()
    assert job["status"] == "completed"
    assert job["article_version"] == 1 and job["article_title"] == ARTICLE["title"]
    assert service.calls == [(ARTICLE["title"], ARTICLE["body"], "both", "ko")]
    assert job["summary"]["summary"] == "테스트용 모의 요약"
    assert "article_body" not in job
    assert client.get("/api/stats").json()["completed"] == 1
    assert client.delete(f'/api/articles/{article["id"]}').status_code == 204
    assert client.get(f'/api/analyses/{queued["id"]}').status_code == 404


def test_failure_and_restart_are_saved(client_factory):
    client, app, service = client_factory(True)
    article = create_article(client)
    def fail():
        raise RuntimeError("Ollama unavailable")
    service.effect = fail
    job = client.post(f'/api/articles/{article["id"]}/analyses', json={"mode": "summarize"}).json()
    failed = client.get(f'/api/analyses/{job["id"]}').json()
    assert failed["status"] == "failed" and failed["error"] == "Ollama unavailable"
    with app.state.sessions() as session:
        interrupted = Analysis(article_id=article["id"], article_version=1,
                               article_title=ARTICLE["title"], article_body=ARTICLE["body"],
                               mode="both", language="ko", status="running")
        session.add(interrupted)
        session.commit()
        interrupted_id = interrupted.id
    assert client.delete(f'/api/articles/{article["id"]}').status_code == 409
    assert client.post(f'/api/articles/{article["id"]}/analyses', json={}).status_code == 409
    app.state.runner.recover_interrupted()
    recovered = client.get(f'/api/analyses/{interrupted_id}').json()
    assert recovered["status"] == "failed" and recovered["completed_at"]
    with app.state.sessions() as session:
        assert session.scalar(select(Analysis).where(Analysis.id == interrupted_id)).error
