"""Exercise real dual-model inference and JSON persistence on a fictional demo."""
import argparse
import json
import math
from pathlib import Path
import time

import httpx

TITLE = "두 모델 비교 테스트: 가상 도서관 보수 계획"
BODY = ("This is a fictional article for testing the news workspace. Cedar City Council approved a library "
        "renovation plan with a budget of two million dollars. Construction is scheduled to begin next September. "
        "The library will stay open during most of the work. This example is not a real news report.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:5173')
    args = parser.parse_args()
    started = time.monotonic()
    with httpx.Client(base_url=args.base_url, timeout=20, trust_env=False) as client:
        client.get('/api/health').raise_for_status()
        caps = client.get('/api/capabilities').json()
        assert caps['ai_enabled'] and all(caps['classification_models_present'].values())
        candidates = client.get('/api/articles', params={'q': TITLE}).json()['items']
        article = next((value for value in candidates if value['title'] == TITLE), None)
        if article is None:
            response = client.post('/api/articles', json={'title': TITLE, 'body': BODY, 'language': 'en'})
            response.raise_for_status()
            article = response.json()
        response = client.post(f"/api/articles/{article['id']}/analyses", json={'mode': 'classify'})
        response.raise_for_status()
        queued = response.json()
        print('Both real models queued; waiting for saved predictions.', flush=True)
        while time.monotonic() - started < 240:
            response = client.get(f"/api/analyses/{queued['id']}")
            response.raise_for_status()
            job = response.json()
            if job['status'] in {'completed', 'failed'}:
                break
            time.sleep(2)
        assert job['status'] == 'completed', job.get('error') or job['status']
        for result in job['classification']['models'].values():
            assert result['status'] == 'ok'
            detail = result['explanation']
            assert math.isclose(detail['intercept'] + detail['tfidf_contribution'] + detail['encoder_contribution'], detail['logit'], abs_tol=1e-6)
            assert math.isclose(1 / (1 + math.exp(-detail['logit'])), result['fake_score'], abs_tol=1e-6)
        records = client.get('/api/analyses', params={'article_id': article['id']}).json()['items']
        persisted = next(value for value in records if value['id'] == job['id'])
        assert persisted['classification'] == job['classification']
        assert job['summary'] is None
        report = {'status': 'passed', 'seconds': round(time.monotonic() - started, 2),
                  'fixture': 'fictional library article', 'analysis': persisted}
        destination = Path(__file__).resolve().parents[1] / 'artifacts' / 'web-smoke-dual.json'
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'status': 'passed', 'seconds': report['seconds'],
                          'models': {key: {'fake_score': value['fake_score'], 'logit': value['explanation']['logit']}
                                     for key, value in job['classification']['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
