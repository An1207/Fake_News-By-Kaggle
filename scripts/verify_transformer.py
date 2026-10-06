"""Verify real saved artifacts and inference without downloading the news CSVs."""
import json
from pathlib import Path
import time

import numpy as np

from news_ai.model import load_classifier, predict
from news_ai.selection import recommended_classifier
from news_ai.text import model_text
from news_ai.transformer import encoder_for, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "artifacts"
    comparison = json.loads((folder / "model_comparison.json").read_text(encoding="utf-8"))
    baseline_path = folder / "classifier.joblib"
    copied_path = folder / "baseline" / "classifier.joblib"
    derived_path = folder / comparison["transformer_artifact"]
    baseline, derived = load_classifier(baseline_path), load_classifier(derived_path)
    assert sha256(baseline_path) == sha256(copied_path) == comparison["lineage"]["baseline_sha256"]
    old_vectorizer = baseline["pipeline"].named_steps["tfidf"]
    new_vectorizer = dict(derived["pipeline"].named_steps["features"].transformer_list)["tfidf"]
    assert old_vectorizer.vocabulary_ == new_vectorizer.vocabulary_
    assert np.array_equal(old_vectorizer.idf_, new_vectorizer.idf_)
    assert derived["metadata"]["encoder_trainable"] is False
    assert derived["metadata"]["test"] == comparison["transformer"]["test"]
    assert comparison["baseline"]["test"]["rows"] == comparison["transformer"]["test"]["rows"] == 6154
    assert derived["metadata"]["leakage_check"]["exact_baseline_manifest"]
    body = ("This is a fictional article for testing. The city council approved a library renovation budget. "
            "Officials discussed the plan in a public meeting. Construction is scheduled to begin next year. "
            "The library will stay open during most of the work.")
    started = time.monotonic()
    result = predict(derived, "Fictional library renovation plan", body)
    elapsed = time.monotonic() - started
    assert result["status"] == "ok"
    content = model_text("Fictional library renovation plan", body)
    scores = derived["pipeline"].predict_proba([content])[0, 1]
    assert np.isclose(result["fake_score"], scores, atol=1e-6)
    expected_path = derived_path if comparison["recommended_model"] == "transformer" else baseline_path
    assert recommended_classifier(folder).resolve() == expected_path.resolve()
    output = {"status": "passed", "baseline_preserved": True, "copied_tfidf_identical": True,
              "saved_pipeline_matches_predict": True, "test_rows": 6154,
              "recommended_model": comparison["recommended_model"], "inference_seconds_cold": round(elapsed, 3)}
    reference_path = derived_path.parent / "validation_reference.npz"
    if reference_path.is_file():
        reference = np.load(reference_path, allow_pickle=False)
        encoder = encoder_for(json.dumps(derived["encoder"], sort_keys=True))
        actual = encoder.encode(reference["texts"].tolist()).astype(np.float64)
        expected = reference["expected"].astype(np.float64)
        cosine = (actual * expected).sum(1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
        assert cosine.min() > .999
        output["execution_min_cosine_against_original_fp32"] = float(cosine.min())
    (derived_path.parent / "inference_verification.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
