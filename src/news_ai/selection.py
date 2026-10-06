"""Default selection uses the completed validation-based comparison report."""
import json
from pathlib import Path


def recommended_classifier(artifact_dir: Path = Path("artifacts")) -> Path:
    baseline = artifact_dir / "classifier.joblib"
    derived = artifact_dir / "eurobert" / "classifier.joblib"
    comparison = artifact_dir / "model_comparison.json"
    if comparison.is_file():
        report = json.loads(comparison.read_text(encoding="utf-8"))
        derived = artifact_dir / report.get("transformer_artifact", "eurobert/classifier.joblib")
        if report.get("recommended_model") == "transformer" and derived.is_file():
            return derived
    return baseline
