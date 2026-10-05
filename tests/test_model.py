import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from news_ai.model import evaluate, predict
from news_ai.text import model_text


def bundle():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(C=10)),
    ])
    pipeline.fit([
        "official government report confirmed research scientific published",
        "official report scientists confirm research government published",
        "fabricated conspiracy rumor shocking hoax imaginary",
        "shocking fabricated conspiracy imaginary rumor hoax",
    ], [0, 0, 1, 1])
    return {"pipeline": pipeline, "metadata": {}}


def test_canonical_prediction_scores_and_labels():
    result = predict(bundle(), "", "fabricated conspiracy rumor shocking hoax imaginary " * 4)
    assert result["status"] == "ok"
    assert result["label"] == 1
    assert result["label_name"] == "fake"
    assert 0 <= result["fake_score"] <= 1
    assert np.isclose(result["fake_score"] + result["real_score"], 1)


def test_korean_short_and_unseen_inputs_abstain():
    model = bundle()
    assert predict(model, "", "오늘 국회에서 예산안을 논의했습니다. " * 10)["status"] == "unsupported_language"
    assert predict(model, "", "official report")["status"] == "insufficient_text"
    assert predict(model, "", "xyzabc defxyz " * 20)["status"] == "outside_vocabulary"


def test_source_shortcut_cleaning_preserves_article_content():
    cleaned = model_text("Budget news", "WASHINGTON (Reuters) - Parliament approved the budget. https://example.org")
    assert "Parliament approved the budget." in cleaned
    assert "Reuters" not in cleaned
    assert "WASHINGTON" not in cleaned
    assert "https" not in cleaned


def test_evaluation_uses_fake_as_positive_label():
    metrics = evaluate([0, 0, 1, 1], np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["fake_f1"] == metrics["accuracy"] == 1
    assert metrics["confusion_matrix_rows_actual_columns_predicted"] == [[2, 0], [0, 2]]
