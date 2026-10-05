"""Run the real classifier and Ollama through the HTTP service on a fictional fixture."""
import json
from pathlib import Path
import time

import httpx
from sqlalchemy import select

from news_api.config import Settings
from news_api.db import make_engine, make_sessions
from news_api.models import Article


def main():
    engine = make_engine(Settings())
    try:
        with make_sessions(engine)() as session:
            article = session.scalar(select(Article).where(
                Article.title.startswith("연결 테스트용 예시: 도서관 보수 계획"),
            ).order_by(Article.created_at.desc()))
            if article is None:
                raise RuntimeError("First save the fictional sample article from the React screen.")
            article_id = article.id
    finally:
        engine.dispose()
    started = time.monotonic()
    with httpx.Client(base_url="http://127.0.0.1:5173", timeout=15, trust_env=False) as client:
        assert client.get("/api/capabilities").json()["ai_enabled"] is True
        response = client.post(f"/api/articles/{article_id}/analyses", json={"mode": "both", "language": "ko"})
        response.raise_for_status()
        queued = response.json()
        assert queued["status"] == "queued"
        print("Real classifier + Ollama job accepted; waiting for saved results.", flush=True)
        deadline = started + 360
        while time.monotonic() < deadline:
            job = client.get(f'/api/analyses/{queued["id"]}').json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(3)
        if job["status"] != "completed":
            raise RuntimeError(f'AI job {job["status"]}: {job.get("error")}')
        assert job["classification"]["status"] == "ok"
        assert job["summary"]["summary"].strip()
        assert job["summary"]["model"] == "exaone3.5:2.4b"
        # A second request verifies persisted results rather than the initial 202 response.
        records = client.get("/api/analyses", params={"article_id": article_id}).json()
        assert any(item["id"] == job["id"] and item["status"] == "completed" for item in records["items"])
        report = {"status": "passed", "fixture": "fictional library article",
                  "seconds": round(time.monotonic() - started, 2), "analysis": job}
        destination = Path(__file__).resolve().parent.parent / "artifacts" / "web-smoke-ai.json"
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
