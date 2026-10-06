import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from news_ai.model import predict
from news_ai.selection import recommended_classifier
from news_ai.transformer import (
    FrozenEuroBertFeatures, aligned_partitions, augment, backup_baseline,
    cached_embeddings, sha256,
)


class StubEncoder:
    dimension = 2

    def __init__(self):
        self.rows = 0

    def encode(self, texts):
        self.rows += len(texts)
        return np.array([[1., 0.] if "official" in t else [0., 1.] for t in texts], dtype=np.float32)


def test_default_requires_completed_recommended_artifact(tmp_path):
    baseline = tmp_path / "classifier.joblib"
    derived = tmp_path / "eurobert" / "classifier.joblib"
    comparison = tmp_path / "model_comparison.json"
    comparison.write_text(json.dumps({"recommended_model": "transformer"}))
    assert recommended_classifier(tmp_path) == baseline
    derived.parent.mkdir()
    derived.touch()
    assert recommended_classifier(tmp_path) == derived
    comparison.write_text(json.dumps({"recommended_model": "baseline"}))
    assert recommended_classifier(tmp_path) == baseline
    custom = tmp_path / "custom" / "classifier.joblib"
    custom.parent.mkdir()
    custom.touch()
    comparison.write_text(json.dumps({"recommended_model": "transformer", "transformer_artifact": "custom/classifier.joblib"}))
    assert recommended_classifier(tmp_path) == custom


def test_backup_preserves_original_and_rejects_existing_mismatch(tmp_path):
    source = tmp_path / "classifier.joblib"
    for name in [source.name, "metrics.json", "data_audit.json", "split_manifest.csv.gz"]:
        (tmp_path / name).write_bytes(name.encode())
    copied = backup_baseline(source)
    assert sha256(copied) == sha256(source)
    assert backup_baseline(source) == copied
    copied.write_bytes(b"unexpected model")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        backup_baseline(source)
    assert source.read_bytes() == b"classifier.joblib"


def test_manifest_uses_original_order_and_rejects_content_change(tmp_path):
    frame = pd.DataFrame({"row_id": ["b", "a", "c"], "body_hash": ["b", "a", "c"],
                          "group_id": ["b", "a", "c"], "label": [0, 1, 0], "sources": ["isot"] * 3})
    assignments = frame.iloc[[1, 0, 2]].assign(split=["train", "validation", "test"])
    path = tmp_path / "manifest.csv.gz"
    assignments.to_csv(path, index=False)
    parts = aligned_partitions(frame, path)
    assert parts["train"]["row_id"].tolist() == ["a"]
    frame.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="mismatch"):
        aligned_partitions(frame, path)


def test_feature_cache_resumes_and_invalidates_changed_inputs(tmp_path, monkeypatch):
    stub = StubEncoder()
    monkeypatch.setattr("news_ai.transformer.encoder_for", lambda spec: stub)
    spec = {"revision": "pinned", "max_length": 128}
    texts = ["official report", "fabricated rumor", "official budget"]
    first = np.array(cached_embeddings(texts, spec, tmp_path, 2))
    assert stub.rows == 3
    assert np.array_equal(first, cached_embeddings(texts, spec, tmp_path, 1))
    assert stub.rows == 3
    # Simulate an interrupted final batch; only the missing row is recomputed.
    state_path = next(tmp_path.glob("*.json"))
    state = json.loads(state_path.read_text())
    state["completed"] = 2
    state_path.write_text(json.dumps(state))
    assert np.array_equal(first, cached_embeddings(texts, spec, tmp_path, 2))
    assert stub.rows == 4
    cached_embeddings(texts, {**spec, "max_length": 64}, tmp_path)
    cached_embeddings([*texts[:2], "changed content"], spec, tmp_path)
    assert stub.rows == 10


def test_derived_artifact_roundtrip_uses_encoder_in_pipeline_and_predict(tmp_path, monkeypatch):
    stub = StubEncoder()
    monkeypatch.setattr("news_ai.transformer.encoder_for", lambda spec: stub)
    texts = ["official government report", "official budget research",
             "fabricated rumor hoax", "fabricated imaginary conspiracy"]
    vectorizer = TfidfVectorizer().fit(texts)
    dense = stub.encode(texts)
    fitted = LogisticRegression(C=10).fit(augment(vectorizer.transform(texts), dense, 2.), [0, 0, 1, 1])
    spec = {"revision": "pinned"}
    pipeline = Pipeline([
        ("features", FeatureUnion([("tfidf", vectorizer), ("eurobert", FrozenEuroBertFeatures(spec).fit([]))],
                                  transformer_weights={"tfidf": 1., "eurobert": 2.})),
        ("classifier", fitted),
    ])
    bundle = {"pipeline": pipeline, "encoder": spec, "embedding_weight": 2., "metadata": {"model": "derived"}}
    path = tmp_path / "derived.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    article = "fabricated rumor hoax imaginary conspiracy " * 5
    result = predict(restored, "", article)
    assert result["label"] == 1 and result["model"] == "derived"
    assert result["fake_score"] == pytest.approx(restored["pipeline"].predict_proba([article])[0, 1])
    explanation = result["explanation"]
    assert explanation["encoder_contribution"] != 0
    assert explanation["parameters"]["encoder_dimensions"] == 2
    assert 1 / (1 + np.exp(-explanation["logit"])) == pytest.approx(result["fake_score"])
    count = stub.rows
    assert predict(restored, "", "오늘 국회에서 논의했습니다. " * 30)["status"] == "unsupported_language"
    assert stub.rows == count


def test_onnx_uses_tokenizer_padding_id_instead_of_model_padding_id(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    tokenizers = pytest.importorskip("tokenizers")
    from news_ai.onnx_encoder import OnnxEuroBertEncoder
    monkeypatch.setattr("news_ai.onnx_encoder.ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=lambda *args, **kwargs: object(),
    ))
    snapshot = tmp_path / ".hf-cache" / "models--EuroBERT--EuroBERT-210m" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"hello": 0, "<|end_of_text|>": 1, "<|pad|>": 2, "<unk>": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(snapshot / "tokenizer.json"))
    (snapshot / "config.json").write_text(json.dumps({"pad_token_id": 1, "pad_token": "<|end_of_text|>", "hidden_size": 2}))
    (snapshot / "tokenizer_config.json").write_text(json.dumps({"pad_token": "<|pad|>"}))
    model_path = tmp_path / "encoder.onnx"
    model_path.write_bytes(b"test graph placeholder")
    runtime = OnnxEuroBertEncoder({"onnx_path": "encoder.onnx", "onnx_sha256": sha256(model_path),
                                  "revision": "revision", "max_length": 128, "threads": 2})
    result = runtime.tokenize(["hello", "hello hello"])
    assert result["input_ids"].tolist() == [[0, 2], [0, 0]]
    assert result["attention_mask"].tolist() == [[1, 0], [1, 1]]
