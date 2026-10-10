from contextlib import ExitStack

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.pool import StaticPool

from news_api.config import Settings
from news_api.db import Base
from news_api.main import create_app
from news_api.models import Analysis, Article, GPTSummary


ARTICLE = {
    "title": "도서관 계획 100% 📰",
    "body": "A fictional council approved a library renovation plan. Construction is scheduled for next year.",
    "language": "en", "source_url": "https://example.com/news",
}


class FakeAI:
    def __init__(self):
        self.calls = []
        self.effect = None
        self.gpt_calls = []
        self.gpt_effect = None

    def analyze(self, title, body, mode, language):
        self.calls.append((title, body, mode, language))
        if self.effect:
            self.effect()
        return {"classification": {"status": "ok", "fake_score": 0.4},
                "summary": {"summary": "테스트용 모의 요약", "model": "test-double"}}

    def summarize_gpt(self, title, body, language):
        self.gpt_calls.append((title, body, language))
        if self.gpt_effect:
            self.gpt_effect()
        return {"summary": "GPT 테스트용 모의 요약", "model": "gpt-4o-mini", "chunks": 1,
                "usage": {"input_tokens": 3000, "output_tokens": 500, "total_tokens": 3500},
                "estimated_cost_usd": .00075}


@pytest.fixture
def client_factory():
    with ExitStack() as stack:
        def make(enabled=False, **settings):
            # SQLite is only the isolated test fixture; the service uses MySQL.
            engine = create_engine("sqlite://", poolclass=StaticPool,
                                   connect_args={"check_same_thread": False})
            @event.listens_for(engine, "connect")
            def foreign_keys(connection, record):
                connection.execute("PRAGMA foreign_keys=ON")
            Base.metadata.create_all(engine)
            service = FakeAI()
            app = create_app(Settings(_env_file=None, ai_enabled=enabled, **settings), engine, service)
            client = stack.enter_context(TestClient(app))
            return client, app, service
        yield make


def create_article(client):
    response = client.post("/api/articles", json=ARTICLE)
    assert response.status_code == 201
    return response.json()


def test_capabilities_returns_aggregate_comparison_without_loading_encoder(client_factory, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr("news_api.main.ROOT", tmp_path)
    folder = tmp_path / "artifacts"
    folder.mkdir()
    report = {"identical_partitions": True, "recommended_model": "baseline",
              "baseline": {"test": {"accuracy": .96, "rows": 6154}},
              "transformer": {"test": {"accuracy": .95, "rows": 6154}}}
    (folder / "model_comparison.json").write_text(json.dumps(report))
    client, _, _ = client_factory()
    response = client.get("/api/capabilities")
    assert response.status_code == 200
    assert response.json()["model_comparison"] == report
    assert "classifier_path" not in response.json()


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


def completed_summary(client):
    article = create_article(client)
    job = client.post(f'/api/articles/{article["id"]}/analyses', json={"mode": "summarize"}).json()
    return article, client.get(f'/api/analyses/{job["id"]}').json()


def test_ollama_remains_default_and_key_is_not_exposed(client_factory):
    client, _, service = client_factory(True, openai_api_key="test-secret-never-transmit")
    _, job = completed_summary(client)
    assert job["gpt_summary"] is None and service.gpt_calls == []
    caps = client.get('/api/capabilities')
    assert caps.json()["default_summary_provider"] == "ollama"
    assert caps.json()["gpt_summary_available"] is True
    assert "test-secret-never-transmit" not in caps.text


def test_gpt_uses_original_snapshot_and_returns_saved_result_without_duplicate_charge(client_factory):
    client, _, service = client_factory(True, openai_api_key="test-secret")
    article, job = completed_summary(client)
    client.put(f'/api/articles/{article["id"]}', json={**ARTICLE, "body": "Changed article body after Ollama summary was finished.", "expected_version": 1})
    url = f'/api/analyses/{job["id"]}/gpt-summary'
    queued = client.post(url)
    assert queued.status_code == 202 and queued.json()["gpt_summary"]["status"] == "queued"
    saved = client.get(f'/api/analyses/{job["id"]}').json()
    assert saved["summary"] == job["summary"]
    assert saved["gpt_summary"]["status"] == "completed"
    assert saved["gpt_summary"]["summary"]["usage"]["total_tokens"] == 3500
    assert service.gpt_calls == [(ARTICLE["title"], ARTICLE["body"], "ko")]
    assert client.post(url).status_code == 200
    assert len(service.gpt_calls) == 1
    assert client.get('/api/analyses').json()["items"][0]["gpt_summary"] == saved["gpt_summary"]
    assert client.delete(f'/api/articles/{article["id"]}').status_code == 204


def test_gpt_requires_configuration_and_completed_ollama_summary(client_factory):
    client, _, service = client_factory(True)
    _, job = completed_summary(client)
    assert client.post(f'/api/analyses/{job["id"]}/gpt-summary').json()["detail"]["code"] == "GPT_NOT_CONFIGURED"
    assert service.gpt_calls == []
    client, _, _ = client_factory(True, openai_api_key="test-secret")
    article = create_article(client)
    job = client.post(f'/api/articles/{article["id"]}/analyses', json={"mode": "classify"}).json()
    assert client.post(f'/api/analyses/{job["id"]}/gpt-summary').json()["detail"]["code"] == "OLLAMA_SUMMARY_REQUIRED"


def test_gpt_failure_does_not_erase_ollama_and_requires_explicit_retry(client_factory):
    client, app, service = client_factory(True, openai_api_key="test-secret")
    article, job = completed_summary(client)
    def fail():
        assert client.delete(f'/api/articles/{article["id"]}').status_code == 409
        raise RuntimeError("mock API failure")
    service.gpt_effect = fail
    url = f'/api/analyses/{job["id"]}/gpt-summary'
    client.post(url)
    saved = client.get(f'/api/analyses/{job["id"]}').json()
    assert saved["status"] == "completed" and saved["summary"] == job["summary"]
    assert saved["gpt_summary"]["status"] == "failed"
    assert client.post(url).json()["detail"]["code"] == "GPT_RETRY_REQUIRED"
    assert len(service.gpt_calls) == 1
    service.gpt_effect = None
    assert client.post(url + '?retry=true').status_code == 202
    assert len(service.gpt_calls) == 2
    with app.state.sessions() as session:
        extra = session.get(GPTSummary, job["id"])
        extra.status = "running"
        session.commit()
    app.state.runner.recover_interrupted()
    assert client.get(f'/api/analyses/{job["id"]}').json()["gpt_summary"]["status"] == "failed"


def test_gpt_pending_returns_existing_job_and_queue_limits_include_gpt(client_factory):
    client, app, service = client_factory(True, openai_api_key="test-secret")
    _, job = completed_summary(client)
    with app.state.sessions() as session:
        session.add(GPTSummary(analysis_id=job["id"], status="running"))
        session.commit()
    assert client.post(f'/api/analyses/{job["id"]}/gpt-summary').status_code == 200
    assert service.gpt_calls == []
    with app.state.sessions() as session:
        for _ in range(9):
            parent = Analysis(article_id=job["article_id"], article_version=1, article_title="test",
                              article_body=ARTICLE["body"], mode="summarize", language="ko", status="completed")
            session.add(parent)
            session.flush()
            session.add(GPTSummary(analysis_id=parent.id, status="queued"))
        session.commit()
    assert client.post(f'/api/articles/{job["article_id"]}/analyses', json={}).status_code == 429


AUTOMATION_KEY = "test-only-automation-key-at-least-32-characters"
AUTOMATION_HEADERS = {"Authorization": "Bearer " + AUTOMATION_KEY}


def test_automation_is_disabled_by_default_and_requires_authentication(client_factory):
    payload = {**ARTICLE, "request_id": "sheet:row:1"}
    client, _, service = client_factory(True, automation_api_key=AUTOMATION_KEY)
    assert client.post('/api/automation/news', json=payload, headers=AUTOMATION_HEADERS).status_code == 503
    assert service.calls == [] and client.get('/api/stats').json()["articles"] == 0
    client, _, _ = client_factory(True, automation_enabled=True, automation_api_key=AUTOMATION_KEY)
    assert client.post('/api/automation/news', json=payload).status_code == 401
    assert client.get('/api/automation/jobs/no-such-id').status_code == 401
    assert client.post('/api/automation/news', json={**payload, "body": " "}, headers=AUTOMATION_HEADERS).status_code == 422


def test_automation_roundtrip_is_idempotent_and_conflicting_payload_is_rejected(client_factory):
    client, _, service = client_factory(True, automation_enabled=True, automation_api_key=AUTOMATION_KEY, openai_api_key="test-secret")
    payload = {**ARTICLE, "request_id": "sheet:row:1"}
    first = client.post('/api/automation/news', json=payload, headers=AUTOMATION_HEADERS)
    assert first.status_code == 202
    job_id = first.json()["id"]
    second = client.post('/api/automation/news', json=payload, headers=AUTOMATION_HEADERS)
    assert second.status_code == 200 and second.json()["id"] == job_id
    assert len(service.calls) == 1 and client.get('/api/stats').json()["articles"] == 1
    conflict = client.post('/api/automation/news', json={**payload, "body": "Different source body for the exact same request id."}, headers=AUTOMATION_HEADERS)
    assert conflict.status_code == 409
    saved = client.get(f'/api/automation/jobs/{job_id}', headers=AUTOMATION_HEADERS).json()
    assert saved["status"] == "completed" and saved["summary"]["summary"]
    assert client.post(f'/api/automation/jobs/{job_id}/gpt-summary').status_code == 401
    client.post(f'/api/automation/jobs/{job_id}/gpt-summary', headers=AUTOMATION_HEADERS)
    assert len(service.gpt_calls) == 1
