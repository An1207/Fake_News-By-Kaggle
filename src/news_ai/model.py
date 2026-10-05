"""Train-only TF-IDF fitting, group splits, held-out evaluation, and inference."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, confusion_matrix,
    f1_score, log_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .data import LABELS, prepare_data
from .text import model_text, word_count


def split_data(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Approximately 80/10/10; proportions vary with connected group sizes."""
    if frame.groupby("label")["group_id"].nunique().min() < 10:
        raise ValueError("Need at least 10 distinct groups in each class for an 80/10/10 split.")
    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    rest_indices, test_indices = next(outer.split(frame, frame["label"], frame["group_id"]))
    rest = frame.iloc[rest_indices]
    inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=seed + 1)
    train_indices, validation_indices = next(inner.split(rest, rest["label"], rest["group_id"]))
    partitions = (
        rest.iloc[train_indices].copy(), rest.iloc[validation_indices].copy(),
        frame.iloc[test_indices].copy(),
    )
    for i, left in enumerate(partitions):
        if set(left["label"]) != {0, 1}:
            raise ValueError("A split lacks one class. Change seed or inspect oversized groups.")
        for right in partitions[i + 1:]:
            for field in ["group_id", "body_hash"]:
                if set(left[field]) & set(right[field]):
                    raise ValueError(f"Data leakage detected in {field}.")
    return partitions


def evaluate(labels, scores: np.ndarray) -> dict:
    predictions = (scores >= 0.5).astype(int)
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "fake_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "brier_score": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, np.column_stack([1 - scores, scores]), labels=[0, 1])),
        "confusion_matrix_labels": ["real", "fake"],
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=["real", "fake"],
            output_dict=True, zero_division=0,
        ),
    }


def _texts(frame: pd.DataFrame) -> list[str]:
    return [model_text(row.title, row.text) for row in frame.itertuples()]


def train(data_dir: Path, output_dir: Path, seed: int = 42,
          max_features: int = 60000, fake_label: str = "auto") -> dict:
    frame, audit = prepare_data(data_dir, fake_label)
    train_frame, validation, test = split_data(frame, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.concat([
        part.assign(split=name)[["row_id", "body_hash", "group_id", "label", "sources", "split"]]
        for name, part in zip(["train", "validation", "test"], [train_frame, validation, test])
    ])
    assignments.to_csv(output_dir / "split_manifest.csv.gz", index=False, compression="gzip")
    split_details = {
        name: {"rows": len(part), "groups": part["group_id"].nunique(),
               "label_counts": {LABELS[int(k)]: int(v) for k, v in part["label"].value_counts().items()}}
        for name, part in zip(["train", "validation", "test"], [train_frame, validation, test])
    }
    print(f"Split: { {name: value['rows'] for name, value in split_details.items()} }", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.98,
        max_features=max_features, sublinear_tf=True, dtype=np.float64,
    )
    print("Fitting TF-IDF on training articles only...", flush=True)
    x_train = vectorizer.fit_transform(_texts(train_frame))
    x_validation = vectorizer.transform(_texts(validation))
    candidates = []
    winner = None
    winning_f1 = -1.0
    for regularization in [0.5, 1.0, 2.0]:
        classifier = LogisticRegression(
            C=regularization, solver="liblinear", max_iter=1000,
            class_weight="balanced", random_state=seed,
        )
        print(f"Training LogisticRegression C={regularization}...", flush=True)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x_train, train_frame["label"])
        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise RuntimeError("Classifier did not converge; increase max_iter before trusting results.")
        validation_metrics = evaluate(validation["label"], classifier.predict_proba(x_validation)[:, 1])
        candidates.append({"C": regularization, "validation": validation_metrics})
        if validation_metrics["macro_f1"] > winning_f1:
            winner = classifier
            winning_f1 = validation_metrics["macro_f1"]
    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", winner)])
    print("Evaluating selected model on held-out test articles...", flush=True)
    test_scores = pipeline.predict_proba(_texts(test))[:, 1]
    metrics = evaluate(test["label"], test_scores)
    per_source = {}
    for source in ["isot", "welfake"]:
        mask = test["sources"].str.split("|").map(lambda sources: source in sources).to_numpy()
        per_source[source] = evaluate(test.loc[mask, "label"], test_scores[mask])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "TF-IDF (1,2)-grams + balanced LogisticRegression",
        "seed": seed,
        "decision_threshold": 0.5,
        "selected_C": float(winner.C),
        "max_features": max_features,
        "actual_features": len(vectorizer.vocabulary_),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "data_audit": audit,
        "splits": split_details,
        "leakage_check": {"body_overlap": 0, "connected_group_overlap": 0},
        "validation_candidates": candidates,
        "test": metrics,
        "test_by_dataset_membership": per_source,
        "majority_baseline_test_accuracy": float(test["label"].value_counts(normalize=True).max()),
        "limitations": [
            "English historical benchmark; not verified on Korean or current news.",
            "Random article-group split does not test publisher or temporal generalization.",
            "Exact bodies, substantial titles and opening 80 words grouped; other paraphrases may remain.",
            "Dataset-membership test sets overlap; they are not independent cross-dataset experiments.",
            "Logistic scores are uncalibrated pattern similarity, not factual truth probabilities.",
            "Source cues remain despite Reuters dateline/token removal; no external fact checking.",
        ],
    }
    joblib.dump({"pipeline": pipeline, "metadata": report}, output_dir / "classifier.joblib", compress=3)
    write_json(output_dir / "metrics.json", report)
    write_json(output_dir / "data_audit.json", audit)
    errors = test.assign(fake_score=test_scores, prediction=(test_scores >= 0.5).astype(int))
    errors = errors[errors["prediction"] != errors["label"]]
    errors[["row_id", "title", "label", "prediction", "fake_score", "sources"]].to_csv(
        output_dir / "test_errors.csv", index=False, encoding="utf-8-sig",
    )
    print(f"Test accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}", flush=True)
    return report


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_classifier(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Run news-ai train first.")
    # joblib is pickle-based: only load artifacts generated by this project/trusted users.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "metadata" not in bundle:
        raise ValueError("Unexpected classifier artifact format.")
    return bundle


def predict(bundle: dict, title: str, text: str) -> dict:
    content = model_text(title, text)
    letters = [character for character in content if character.isalpha()]
    latin_ratio = sum(character.isascii() for character in letters) / max(len(letters), 1)
    if latin_ratio < 0.8:
        return {"status": "unsupported_language", "message": "분류 모델은 영어 데이터로 학습했습니다. 영어 기사를 입력해 주세요."}
    if word_count(content) < 20:
        return {"status": "insufficient_text", "message": "분류에는 제목과 본문을 합쳐 영어 단어 20개 이상이 필요합니다."}
    vectorizer = bundle["pipeline"].named_steps["tfidf"]
    features = vectorizer.transform([content])
    if features.nnz == 0:
        return {"status": "outside_vocabulary", "message": "학습한 어휘와 겹치는 표현이 없어 분류하지 못했습니다."}
    classifier = bundle["pipeline"].named_steps["classifier"]
    fake_index = list(classifier.classes_).index(1)
    fake_score = float(classifier.predict_proba(features)[0, fake_index])
    label = int(fake_score >= 0.5)
    return {
        "status": "ok", "label": label, "label_name": LABELS[label],
        "display_label": "가짜 뉴스 패턴에 가까움" if label else "진짜 뉴스 패턴에 가까움",
        "fake_score": fake_score, "real_score": 1 - fake_score,
        "uncertain": max(fake_score, 1 - fake_score) < 0.65,
        "note": "학습 데이터의 문체·어휘에 따른 모델 점수이며 사실 검증 결과가 아닙니다.",
    }
