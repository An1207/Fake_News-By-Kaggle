"""Real MySQL + HTTP smoke test. Removes only the fixture article it creates."""
import argparse
import json
from pathlib import Path
from uuid import uuid4

import httpx
from sqlalchemy import inspect, select, text

from news_api.config import Settings
from news_api.db import make_engine, make_sessions
from news_api.models import Analysis, Article


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    arguments = parser.parse_args()
    settings = Settings()
    engine = make_engine(settings)
    sessions = make_sessions(engine)
    article_id = None
    report = {"url": arguments.url, "checks": []}
    def passed(name):
        report["checks"].append(name)
    try:
        with engine.connect() as connection:
            report["database_version"] = connection.scalar(text("SELECT VERSION()"))
            assert {"articles", "analyses", "alembic_version"} <= set(inspect(connection).get_table_names())
            passed("MySQL migration tables")
        with httpx.Client(base_url=arguments.url, timeout=10, trust_env=False) as client:
            assert client.get("/api/health").json()["database"] == "connected"
            ai_enabled = client.get("/api/capabilities").json()["ai_enabled"]
            passed("React proxy -> FastAPI -> MySQL")
            token = str(uuid4())
            # More than 64 KiB in UTF-8 exercises LONGTEXT rather than MySQL TEXT.
            payload = {"title": f"통합 검증 100% 📰 {token}", "body": ("한글과 이모지 저장 검증입니다. 📰 " * 2000).strip(),
                       "source_url": "https://example.com/test", "language": "ko"}
            created = client.post("/api/articles", json=payload)
            created.raise_for_status()
            article = created.json()
            article_id = article["id"]
            assert client.get(f"/api/articles/{article_id}").json()["body"] == payload["body"]
            passed("create/read Unicode and >64 KiB UTF-8 body")
            with sessions() as session:
                assert session.get(Article, article_id).body == payload["body"]
                fixture = Analysis(article_id=article_id, article_version=1,
                                   article_title=payload["title"], article_body=payload["body"],
                                   mode="summarize", language="ko", status="completed",
                                   summary={"summary": "DB JSON 저장 검증용 모의 결과 📰", "model": "test-double"})
                session.add(fixture)
                session.commit()
                analysis_id = fixture.id
            passed("independent DB session persistence and JSON history")
            records = client.get("/api/analyses", params={"article_id": article_id}).json()
            assert records["items"][0]["summary"]["model"] == "test-double"
            assert "article_body" not in records["items"][0]
            result = client.get("/api/articles", params={"q": token}).json()
            assert result["total"] == 1 and result["items"][0]["id"] == article_id
            assert "body" not in result["items"][0]
            passed("search and lean paginated responses")
            updated = client.put(f"/api/articles/{article_id}", json={**payload, "title": f"수정 {token}", "expected_version": 1})
            assert updated.status_code == 200 and updated.json()["version"] == 2
            stale = client.put(f"/api/articles/{article_id}", json={**payload, "expected_version": 1})
            assert stale.status_code == 409
            history = client.get(f"/api/analyses/{analysis_id}").json()
            assert history["article_version"] == 1 and history["article_title"] == payload["title"]
            passed("optimistic update conflict and immutable analysis version")
            if not ai_enabled:
                disabled = client.post(f"/api/articles/{article_id}/analyses", json={"mode": "both"})
                assert disabled.status_code == 409 and disabled.json()["detail"]["code"] == "AI_NOT_ENABLED"
                passed("disabled inference creates no job")
            assert client.delete(f"/api/articles/{article_id}").status_code == 204
            assert client.get(f"/api/articles/{article_id}").status_code == 404
            assert client.get(f"/api/analyses/{analysis_id}").status_code == 404
            passed("article deletion cascades history")
            article_id = None
        report["status"] = "passed"
        destination = Path(__file__).resolve().parent.parent / "artifacts" / "web-smoke.json"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if article_id:
            with sessions() as session:
                fixture = session.get(Article, article_id)
                if fixture:
                    session.delete(fixture)
                    session.commit()
        engine.dispose()


if __name__ == "__main__":
    main()
