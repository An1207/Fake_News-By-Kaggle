import json

import pytest

from news_ai.cli import main
from news_ai.comparison import comparison_text


@pytest.fixture
def reports(tmp_path):
    old_scores = {"rows": 100, "accuracy": .9, "macro_f1": .88, "fake_f1": .87, "roc_auc": .97}
    new_scores = {"rows": 100, "accuracy": .95, "macro_f1": .93, "fake_f1": .92, "roc_auc": .99}
    report = {"identical_partitions": True, "transformer_artifact": "eurobert/classifier.joblib",
              "baseline": {"test": old_scores, "validation": {"macro_f1": .87}},
              "transformer": {"test": new_scores, "validation": {"macro_f1": .92}},
              "recommended_model": "transformer",
              "paired_test": {"both_correct": 88, "both_wrong": 3,
                              "baseline_only_correct": 2, "transformer_only_correct": 7}}
    old = {"test": old_scores, "actual_features": 60000, "selected_C": 2, "decision_threshold": .5}
    new = {"test": new_scores, "actual_features": 60768, "selected_C": 2, "decision_threshold": .5,
           "embedding_weight": .5, "encoder_release_date": "2025-03-10", "encoder_revision_date": "2025-04-17",
           "encoder": {"max_length": 128, "revision": "pinned"}}
    for relative, content in [("model_comparison.json", report), ("baseline/metrics.json", old), ("eurobert/metrics.json", new)]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding='utf-8')
    return tmp_path


def test_compare_cli_prints_calculated_differences_without_model_binaries(reports, capsys):
    assert main(['compare', '--artifact-dir', str(reports)]) == 0
    output = capsys.readouterr().out
    assert '90.00%' in output and '95.00%' in output and '+5.00%p' in output
    assert '오답: 10 → 5건 / 오답 감소 +5건 (+50.00%)' in output
    assert '앞 128토큰' in output and '60,768' in output
    assert '전체 미세조정은 하지 않았습니다' in output
    assert '외부 사실 검증 결과가 아닙니다' in output


def test_comparison_refuses_mixed_or_inconsistent_evaluation(reports):
    path = reports / 'eurobert/metrics.json'
    metadata = json.loads(path.read_text())
    metadata['test']['accuracy'] = .99
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='metrics differ'):
        comparison_text(reports)
