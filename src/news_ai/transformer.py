"""Frozen EuroBERT features + the preserved TF-IDF vocabulary + LogisticRegression.

The original artifact is never overwritten. All hyperparameters are selected on
validation; the original split manifest is the authority for this experiment.
"""

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from .data import prepare_data
from .model import _texts, evaluate, load_classifier, write_json

MODEL_ID = "EuroBERT/EuroBERT-210m"
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_baseline(source: Path) -> Path:
    destination = source.parent / "baseline"
    destination.mkdir(parents=True, exist_ok=True)
    for name in [source.name, "metrics.json", "data_audit.json", "split_manifest.csv.gz"]:
        original, copied = source.parent / name, destination / name
        if not original.is_file():
            raise FileNotFoundError(original)
        if copied.exists() and sha256(original) != sha256(copied):
            raise ValueError(f"Existing baseline backup differs: {name}; refusing to overwrite.")
        if not copied.exists():
            shutil.copy2(original, copied)
        if sha256(original) != sha256(copied):
            raise RuntimeError(f"Baseline copy verification failed: {name}")
    return destination / source.name


class EuroBertEncoder:
    def __init__(self, spec: dict):
        try:
            import torch
            from transformers import AutoTokenizer, EuroBertConfig, EuroBertModel
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("Install the encoder dependencies: uv sync --extra transformer --extra web --extra dev") from error

        self.torch = torch
        self.spec = spec
        torch.set_num_threads(spec.get("threads", 4))
        local = snapshot_download(
            MODEL_ID, revision=spec["revision"], cache_dir=ROOT / ".hf-cache",
            allow_patterns=["*.json", "*.safetensors", "*.model"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=False)
        # Original checkpoints predate native Transformers support. Load their
        # config and weights using the official native class, never remote code.
        config = EuroBertConfig.from_pretrained(local)
        self.model, loading = EuroBertModel.from_pretrained(
            local, config=config, attn_implementation="sdpa", output_loading_info=True,
        )
        if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError(f"Encoder checkpoint did not load completely: {loading}")
        self.model.eval().requires_grad_(False)
        if spec.get("quantization") == "dynamic-int8":
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
            )
        self.dimension = config.hidden_size

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.spec["max_length"],
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
        output = vectors.float().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("Non-finite encoder output.")
        return output


@lru_cache(maxsize=1)
def encoder_for(serialized_spec: str) -> EuroBertEncoder:
    spec = json.loads(serialized_spec)
    if spec.get("backend") in {"onnx", "openvino"}:
        from .onnx_encoder import OnnxEuroBertEncoder
        return OnnxEuroBertEncoder(spec)
    return EuroBertEncoder(spec)


def augment(features, embeddings: np.ndarray, weight: float):
    return hstack([features, csr_matrix(embeddings * weight)], format="csr")


class FrozenEuroBertFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, spec: dict, batch_size: int = 16):
        self.spec = spec
        self.batch_size = batch_size

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        texts = list(X)
        encoder = encoder_for(json.dumps(self.spec, sort_keys=True))
        if not texts:
            return np.empty((0, encoder.dimension), dtype=np.float32)
        return np.concatenate([encoder.encode(texts[i:i + self.batch_size])
                               for i in range(0, len(texts), self.batch_size)])


def aligned_partitions(frame: pd.DataFrame, manifest: Path) -> dict[str, pd.DataFrame]:
    assignment = pd.read_csv(manifest, dtype={"row_id": str})
    if frame["row_id"].duplicated().any() or assignment["row_id"].duplicated().any():
        raise ValueError("Duplicate row identity.")
    actual = frame.set_index("row_id")
    if set(actual.index) != set(assignment["row_id"]):
        raise ValueError("Dataset differs from preserved baseline manifest.")
    ordered = actual.loc[assignment["row_id"]].reset_index()
    for field in ["body_hash", "group_id", "label", "sources"]:
        if not np.array_equal(ordered[field].astype(str), assignment[field].astype(str)):
            raise ValueError(f"Baseline manifest mismatch: {field}")
    ordered["split"] = assignment["split"].to_numpy()
    if set(ordered["split"]) != {"train", "validation", "test"}:
        raise ValueError("Unexpected partitions.")
    parts = {name: ordered[ordered["split"] == name].copy() for name in ["train", "validation", "test"]}
    for i, left in enumerate(parts.values()):
        for right in list(parts.values())[i + 1:]:
            for field in ["body_hash", "group_id"]:
                if set(left[field]) & set(right[field]):
                    raise ValueError(f"Partition leakage: {field}")
    return parts


def cached_embeddings(texts: list[str], spec: dict, directory: Path, batch_size: int = 16):
    """Flush embeddings before committing progress; corpus/spec keyed for safe resume."""
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for content in texts:
        raw = content.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    identity = digest.hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / f"{identity}.json"
    array_path = directory / f"{identity}.npy"
    encoder = encoder_for(json.dumps(spec, sort_keys=True))
    shape = (len(texts), encoder.dimension)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": 0}
    if state_path.exists() and not array_path.exists():
        raise ValueError("Embedding cache state exists without data.")
    vectors = np.lib.format.open_memmap(array_path, mode="r+" if array_path.exists() else "w+", dtype="float32", shape=shape)
    if vectors.shape != shape or vectors.dtype != np.float32:
        raise ValueError("Embedding cache shape mismatch.")
    start = int(state["completed"])
    if not 0 <= start <= len(texts):
        raise ValueError("Invalid embedding progress.")
    begun = time.monotonic()
    next_log = begun
    for offset in range(start, len(texts), batch_size):
        stop = min(offset + batch_size, len(texts))
        vectors[offset:stop] = encoder.encode(texts[offset:stop])
        vectors.flush()
        temporary = state_path.with_suffix(".tmp")
        write_json(temporary, {"completed": stop, "rows": len(texts), "spec": spec, "corpus_sha256": identity})
        temporary.replace(state_path)
        now = time.monotonic()
        if now >= next_log or stop == len(texts):
            rate = (stop - start) / max(now - begun, .001)
            remaining = (len(texts) - stop) / max(rate, .001)
            print(f"Embeddings {stop:,}/{len(texts):,}; {rate:.1f} articles/s; ETA {remaining / 60:.1f} min", flush=True)
            next_log = now + 30
    return vectors


def train_transformer(data_dir: Path, baseline_path: Path, output_dir: Path,
                      revision: str, max_length: int = 128, batch_size: int = 8,
                      threads: int = 2, quantization: str = "none", backend: str = "openvino", device: str = "AUTO") -> dict:
    if not revision or revision == "main":
        raise ValueError("Use a pinned pre-May-2025 model revision.")
    from huggingface_hub import HfApi
    commit = HfApi().list_repo_commits(MODEL_ID, revision=revision)[0]
    if commit.commit_id != revision or commit.created_at >= datetime(2025, 5, 1, tzinfo=timezone.utc):
        raise ValueError("Encoder revision must be a full commit hash published before May 2025.")
    if not 16 <= max_length <= 8192 or batch_size < 1 or threads < 1:
        raise ValueError("Invalid encoder options.")
    if baseline_path.resolve().parent == output_dir.resolve():
        raise ValueError("Derived model output must differ from baseline directory.")
    preserved = backup_baseline(baseline_path)
    baseline_hash = sha256(preserved)
    baseline = load_classifier(preserved)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {"model_id": MODEL_ID, "revision": revision, "max_length": max_length,
            "pooling": "masked-mean-l2", "quantization": quantization, "threads": threads}
    if backend in {"onnx", "openvino"}:
        from .onnx_encoder import export_encoder
        export_path = output_dir / "encoder_export.json"
        if not export_path.is_file():
            exported = export_encoder(revision, output_dir)
        else:
            exported = json.loads(export_path.read_text(encoding="utf-8"))
        if exported["encoder"]["revision"] != revision or not exported["tokenizer_parity"] or exported["validation_min_cosine"] < .999:
            raise ValueError("ONNX export does not match this revision or has not passed parity validation.")
        spec.update({key: exported["encoder"][key] for key in ["backend", "quantization", "onnx_path", "onnx_sha256"]})
        spec["backend"] = backend
        if backend == "openvino":
            if device == "AUTO":
                import openvino as ov
                device = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
            spec.update(device=device, inference_precision="f16" if device == "GPU" else "f32")
    elif backend != "torch":
        raise ValueError("Unsupported encoder backend.")
    write_json(output_dir / "training_state.json", {"status": "preparing", "encoder": spec})
    frame, audit = prepare_data(data_dir, "auto")
    parts = aligned_partitions(frame, preserved.parent / "split_manifest.csv.gz")
    del frame
    print(f"Verified baseline partitions: { {name: len(part) for name, part in parts.items()} }", flush=True)
    texts = {name: _texts(part) for name, part in parts.items()}
    vectorizer = baseline["pipeline"].named_steps["tfidf"]
    sparse = {name: vectorizer.transform(content) for name, content in texts.items()}
    old_scores = {name: baseline["pipeline"].named_steps["classifier"].predict_proba(features)[:, 1]
                  for name, features in sparse.items() if name != "train"}
    old_test = evaluate(parts["test"]["label"], old_scores["test"])
    if abs(old_test["accuracy"] - baseline["metadata"]["test"]["accuracy"]) > 1e-12:
        raise ValueError("Preserved baseline accuracy does not reproduce.")
    write_json(output_dir / "training_state.json", {"status": "encoding", "encoder": spec})
    vectors = {name: cached_embeddings(content, spec, output_dir / "embedding_cache", batch_size)
               for name, content in texts.items()}
    # Releasing the encoder's native memory makes room for sparse LR training.
    encoder_for.cache_clear()
    del texts
    for part in parts.values():
        part.drop(columns=["title", "text"], inplace=True)
    candidates, winner, best_f1, selected_weight = [], None, -1., None
    for weight in [0.5, 1., 2.]:
        x_train = augment(sparse["train"], vectors["train"], weight)
        x_validation = augment(sparse["validation"], vectors["validation"], weight)
        for c in [0.5, 1., 2.]:
            classifier = LogisticRegression(C=c, solver="liblinear", max_iter=1000,
                                            class_weight="balanced", random_state=42)
            print(f"Training derived LR: embedding weight={weight}, C={c}", flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(x_train, parts["train"]["label"])
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError("Derived classifier did not converge.")
            result = evaluate(parts["validation"]["label"], classifier.predict_proba(x_validation)[:, 1])
            candidates.append({"C": c, "embedding_weight": weight, "validation": result})
            print(f"Validation accuracy={result['accuracy']:.6f} macro F1={result['macro_f1']:.6f}", flush=True)
            if result["macro_f1"] > best_f1:
                winner, selected_weight, best_f1 = classifier, weight, result["macro_f1"]
        del x_train, x_validation
    baseline_validation = evaluate(parts["validation"]["label"], old_scores["validation"])
    recommended = "transformer" if best_f1 > baseline_validation["macro_f1"] else "baseline"
    # Selection is finalized before the derived classifier sees held-out scores.
    scores = winner.predict_proba(augment(sparse["test"], vectors["test"], selected_weight))[:, 1]
    test_metrics = evaluate(parts["test"]["label"], scores)
    labels = parts["test"]["label"].to_numpy()
    old_correct, new_correct = (old_scores["test"] >= .5) == labels, (scores >= .5) == labels
    import sklearn, platform
    from importlib.metadata import version
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "TF-IDF + frozen EuroBERT-210M + balanced LogisticRegression",
        "encoder": spec, "encoder_release_date": "2025-03-10", "encoder_paper_date": "2025-03-07",
        "encoder_revision_date": commit.created_at.isoformat(),
        "encoder_trainable": False, "seed": 42, "decision_threshold": .5,
        "selected_C": float(winner.C), "embedding_weight": selected_weight,
        "actual_features": int(winner.n_features_in_),
        "lineage": {"baseline_sha256": baseline_hash, "copied_vocabulary": True,
                    "split_manifest_sha256": sha256(preserved.parent / "split_manifest.csv.gz")},
        "versions": {"torch": version("torch"), "transformers": version("transformers"),
                     "onnxruntime": version("onnxruntime") if backend in {"onnx", "openvino"} else None,
                     "openvino": version("openvino") if backend == "openvino" else None,
                     "python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "data_audit": audit, "splits": baseline["metadata"]["splits"],
        "leakage_check": {"body_overlap": 0, "connected_group_overlap": 0, "exact_baseline_manifest": True},
        "validation_candidates": candidates, "test": test_metrics,
        "limitations": baseline["metadata"]["limitations"] + [
            f"Encoder reads the first {max_length} tokens; TF-IDF reads the complete normalized article.",
            "EuroBERT weights are frozen; this is feature transfer, not full Transformer fine-tuning.",
            "Encoder pretraining corpus overlap with these public benchmarks is not independently audited.",
            "CPU Transformer inference is slower than the original sparse-only LogisticRegression.",
            "Intel GPU inference uses mixed float16 precision; small device-dependent numerical differences may occur.",
        ],
    }
    feature_union = FeatureUnion(
        [("tfidf", vectorizer), ("eurobert", FrozenEuroBertFeatures(spec, batch_size).fit([]))],
        transformer_weights={"tfidf": 1., "eurobert": selected_weight},
    )
    bundle = {"pipeline": Pipeline([("features", feature_union), ("classifier", winner)]),
              "encoder": spec, "embedding_weight": selected_weight, "metadata": report}
    joblib.dump(bundle, output_dir / "classifier.joblib", compress=3)
    write_json(output_dir / "metrics.json", report)
    comparison = {
        "created_at_utc": report["created_at_utc"], "identical_partitions": True,
        "selection_metric": "validation_macro_f1", "recommended_model": recommended,
        "transformer_artifact": Path(os.path.relpath(output_dir / "classifier.joblib", baseline_path.parent)).as_posix(),
        "baseline": {"model": baseline["metadata"]["model"], "validation": baseline_validation, "test": old_test},
        "transformer": {"model": report["model"], "validation": max(candidates, key=lambda c: c["validation"]["macro_f1"])["validation"], "test": test_metrics},
        "test_accuracy_difference_percentage_points": 100 * (test_metrics["accuracy"] - old_test["accuracy"]),
        "test_macro_f1_difference": test_metrics["macro_f1"] - old_test["macro_f1"],
        "paired_test": {"both_correct": int((old_correct & new_correct).sum()),
                        "both_wrong": int((~old_correct & ~new_correct).sum()),
                        "baseline_only_correct": int((old_correct & ~new_correct).sum()),
                        "transformer_only_correct": int((~old_correct & new_correct).sum())},
        "lineage": report["lineage"],
    }
    temporary_comparison = baseline_path.parent / "model_comparison.tmp"
    write_json(temporary_comparison, comparison)
    temporary_comparison.replace(baseline_path.parent / "model_comparison.json")
    write_json(output_dir / "training_state.json", {"status": "complete", "encoder": spec})
    if sha256(baseline_path) != baseline_hash:
        raise RuntimeError("Baseline artifact changed during experiment.")
    print(f"FINAL baseline accuracy={old_test['accuracy']:.6f}; transformer accuracy={test_metrics['accuracy']:.6f}; recommended={recommended}", flush=True)
    return comparison
