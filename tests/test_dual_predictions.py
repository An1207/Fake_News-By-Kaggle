import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from news_ai.model import predict
from news_api.ai import LocalAIService
from news_api.config import Settings
from news_api.db import Base
from news_api.main import create_app

BODY = 'official government budget report public library school research ' * 4


def bundle():
    pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                         ('classifier', LogisticRegression(C=2, class_weight='balanced'))])
    pipeline.fit(['official government budget report', 'public library school research',
                  'fabricated rumor conspiracy hoax', 'imaginary rumor fake conspiracy'], [0, 0, 1, 1])
    return {'pipeline': pipeline, 'metadata': {'model': 'tiny test classifier'}}


@pytest.fixture
def service(tmp_path):
    old, new = tmp_path / 'old.joblib', tmp_path / 'new.joblib'
    joblib.dump(bundle(), old)
    joblib.dump(bundle(), new)
    settings = Settings(_env_file=None, ai_enabled=True, classifier_path=new,
                        baseline_classifier_path=old, transformer_classifier_path=new)
    return LocalAIService(settings)


def test_linear_explanation_reconstructs_probability_and_directions():
    result = predict(bundle(), '', BODY)
    detail = result['explanation']
    assert detail['encoder_contribution'] == 0
    assert detail['logit'] == pytest.approx(detail['intercept'] + detail['tfidf_contribution'])
    assert 1 / (1 + np.exp(-detail['logit'])) == pytest.approx(result['fake_score'])
    assert all(value['contribution'] > 0 for value in detail['toward_fake'])
    assert all(value['contribution'] < 0 for value in detail['toward_real'])
    assert detail['input']['matched_tfidf_features'] > 0
    assert detail['parameters']['C'] == 2
    assert len(detail['toward_real']) <= 5


def test_adapter_returns_both_models_and_reuses_loaded_artifacts(service, monkeypatch):
    first = service.analyze('', BODY, 'classify', 'ko')['classification']
    assert set(first['models']) == {'baseline', 'transformer'}
    assert first['agreement'] is True and first['selected_model'] == 'transformer'
    assert first['fake_score'] == first['models']['transformer']['fake_score']
    monkeypatch.setattr('news_ai.model.load_classifier', lambda path: pytest.fail('model loaded twice'))
    assert service.analyze('', BODY, 'classify', 'ko')['classification'] == first


def test_adapter_preserves_baseline_when_new_artifact_missing_and_abstains(service):
    service.settings.transformer_classifier_path.unlink()
    service.settings.classifier_path = service.settings.baseline_classifier_path
    result = service.analyze('', BODY, 'classify', 'ko')['classification']
    assert result['status'] == 'ok' and result['models']['transformer']['status'] == 'model_unavailable'
    assert result['agreement'] is None
    korean = service.analyze('', '오늘 국회에서 논의했습니다. ' * 30, 'classify', 'ko')['classification']
    assert korean['models']['baseline']['status'] == 'unsupported_language'
    assert 'explanation' not in korean


def test_dual_results_and_explanation_survive_api_database_roundtrip(service):
    engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    with TestClient(create_app(service.settings, engine, service)) as client:
        article = client.post('/api/articles', json={'title': 'Fictional library report', 'body': BODY, 'language': 'en'}).json()
        queued = client.post(f"/api/articles/{article['id']}/analyses", json={'mode': 'classify'}).json()
        job = client.get(f"/api/analyses/{queued['id']}").json()
        assert job['status'] == 'completed' and job['summary'] is None
        models = job['classification']['models']
        for result in models.values():
            assert result['explanation']['parameters']['decision_threshold'] == .5
            assert result['explanation']['toward_real']
        saved = client.get('/api/analyses', params={'article_id': article['id']}).json()
        assert saved['items'][0]['classification'] == job['classification']
