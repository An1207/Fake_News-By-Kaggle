"""Real MySQL + in-process FastAPI, mocked LLM transport; deletes only its fixture."""

from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select, text

from news_api.config import Settings
from news_api.db import make_engine, make_sessions
from news_api.main import create_app
from news_api.models import AutomationRequest, GPTSummary


def main():
    # Overrides belong only to this isolated app object, never .env/live service.
    settings = Settings().model_copy(update={"ai_enabled": True, "openai_api_key": SecretStr("mock-not-a-real-api-key"),
                                            "automation_enabled": True,
                                            "automation_api_key": SecretStr("mock-automation-key-only-for-this-test")})
    engine = make_engine(settings)
    sessions = make_sessions(engine)
    fixture_id = None
    client = TestClient(create_app(settings, engine))  # No lifespan/recovery of user jobs.
    payload = {"request_id": "gpt-smoke:" + str(uuid4()), "title": "GPT MySQL verification: fictional test",
               "body": "This fictional library renovation has not started. Construction is planned next September.",
               "language": "en", "summary_language": "ko", "mode": "summarize"}
    headers = {"Authorization": "Bearer " + settings.automation_api_key.get_secret_value()}
    gpt_response = {"status": "completed", "model": "gpt-4o-mini",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "- 모의 GPT 요약: 가상 도서관 공사는 아직 시작되지 않았습니다."}]}],
                    "usage": {"input_tokens": 3000, "output_tokens": 500, "total_tokens": 3500,
                              "input_tokens_details": {"cached_tokens": 0}}}
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == "0002"
        with patch('news_ai.summarizer.OllamaSummarizer._request', return_value={"message": {"content": "- 모의 Ollama 요약"}, "done": True}), \
             patch('news_ai.openai_summary.OpenAISummarizer._request', return_value=gpt_response) as gpt_call:
            created = client.post('/api/automation/news', json=payload, headers=headers)
            created.raise_for_status()
            job = created.json()
            fixture_id, job_id = job['article_id'], job['id']
            replay = client.post('/api/automation/news', json=payload, headers=headers)
            assert replay.status_code == 200 and replay.json()['id'] == job_id
            original = client.get(f'/api/analyses/{job_id}').json()['summary']
            assert client.post(f'/api/analyses/{job_id}/gpt-summary').status_code == 202
            assert client.post(f'/api/analyses/{job_id}/gpt-summary').status_code == 200
            saved = client.get(f'/api/automation/jobs/{job_id}', headers=headers).json()
            assert saved['summary'] == original
            assert saved['gpt_summary']['summary']['estimated_cost_usd'] == .00075
            assert saved['gpt_summary']['summary']['usage']['total_tokens'] == 3500
            assert gpt_call.call_count == 1
            with sessions() as session:
                assert session.get(GPTSummary, job_id).summary['usage']['input_tokens'] == 3000
                assert session.scalar(select(AutomationRequest).where(AutomationRequest.analysis_id == job_id))
            deleted = client.delete(f'/api/articles/{fixture_id}')
            deleted.raise_for_status()
            fixture_id = None
            with sessions() as session:
                assert session.get(GPTSummary, job_id) is None
                assert session.scalar(select(AutomationRequest).where(AutomationRequest.analysis_id == job_id)) is None
        print('PASS: MySQL 0002, automation idempotency, GPT snapshot/result/token persistence, one mocked call, fixture cascade cleanup. No paid API/Make call.')
    finally:
        if fixture_id:
            client.delete(f'/api/articles/{fixture_id}')
        client.close()
        engine.dispose()


if __name__ == '__main__':
    main()
