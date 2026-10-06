import argparse
import json
from pathlib import Path
import sys

from .summarizer import OllamaError, OllamaSummarizer
from .selection import recommended_classifier


def _article(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", default="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="News article body")
    group.add_argument("--file", type=Path, help="UTF-8 plain text article file")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="SAI fake news classifier and Ollama summarizer")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download both Kaggle datasets; preserve existing files")
    download.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    prepare = commands.add_parser("prepare", help="Audit labels, duplicates and groups")
    fit = commands.add_parser("train", help="Train and evaluate the baseline")
    for subparser in [prepare, fit]:
        subparser.add_argument("--data-dir", type=Path, default=Path("."))
        subparser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
        subparser.add_argument("--welfake-fake-label", choices=["auto", "0", "1"], default="auto")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--max-features", type=int, default=60000)
    derived = commands.add_parser("train-transformer", help="Preserve baseline and train EuroBERT + TF-IDF + LR")
    derived.add_argument("--data-dir", type=Path, default=Path("."))
    derived.add_argument("--baseline", type=Path, default=Path("artifacts/classifier.joblib"))
    derived.add_argument("--output-dir", type=Path, default=Path("artifacts/eurobert"))
    derived.add_argument("--revision", required=True, help="Pinned pre-May-2025 Hugging Face commit")
    derived.add_argument("--max-length", type=int, default=128)
    derived.add_argument("--batch-size", type=int, default=8)
    derived.add_argument("--threads", type=int, default=2)
    derived.add_argument("--backend", choices=["openvino", "onnx", "torch"], default="openvino")
    derived.add_argument("--device", choices=["AUTO", "CPU", "GPU"], default="AUTO", help="OpenVINO device")
    derived.add_argument("--quantization", choices=["dynamic-int8", "none"], default="none", help="Torch backend only")
    export = commands.add_parser("export-encoder", help="Export and validate the frozen CPU ONNX encoder")
    export.add_argument("--revision", required=True)
    export.add_argument("--output-dir", type=Path, default=Path("artifacts/eurobert"))
    inference = commands.add_parser("predict", help="Classify an English article")
    inference.add_argument("--model", type=Path, default=recommended_classifier())
    _article(inference)
    summary = commands.add_parser("summarize", help="Summarize with a local Ollama model")
    _article(summary)
    summary.add_argument("--llm-model", default=None)
    summary.add_argument("--language", choices=["ko", "en"], default="ko")
    status = commands.add_parser("ollama-status", help="List local Ollama models")
    status.add_argument("--llm-model", default=None)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "download":
            from .data import download_datasets

            download_datasets(arguments.data_dir)
        elif arguments.command == "prepare":
            from .data import prepare_data
            from .model import write_json

            _, audit = prepare_data(arguments.data_dir, arguments.welfake_fake_label)
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
            write_json(arguments.output_dir / "data_audit.json", audit)
            print(json.dumps(audit, ensure_ascii=False, indent=2))
        elif arguments.command == "train":
            from .model import train

            if arguments.max_features < 100:
                raise ValueError("--max-features must be at least 100.")
            train(arguments.data_dir, arguments.output_dir, arguments.seed,
                  arguments.max_features, arguments.welfake_fake_label)
        elif arguments.command == "export-encoder":
            from .onnx_encoder import export_encoder
            export_encoder(arguments.revision, arguments.output_dir)
        elif arguments.command == "train-transformer":
            from .transformer import train_transformer

            train_transformer(arguments.data_dir, arguments.baseline, arguments.output_dir,
                              arguments.revision, arguments.max_length, arguments.batch_size,
                              arguments.threads, arguments.quantization, arguments.backend, arguments.device)
        elif arguments.command in {"predict", "summarize"}:
            text = arguments.file.read_text(encoding="utf-8-sig") if arguments.file else arguments.text
            if arguments.command == "predict":
                from .model import load_classifier, predict

                result = predict(load_classifier(arguments.model), arguments.title, text)
            else:
                result = OllamaSummarizer(model=arguments.llm_model).summarize(
                    arguments.title, text, arguments.language,
                ).to_dict()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            models = OllamaSummarizer(model=arguments.llm_model).list_models()
            print(json.dumps({"models": models}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, OllamaError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
