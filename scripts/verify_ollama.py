r"""Run real local LLM inference and save examples for human review.

Usage: .venv\Scripts\python.exe -X utf8 scripts/verify_ollama.py
This smoke check verifies execution, not general summary accuracy.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from news_ai.summarizer import OllamaSummarizer

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    # Force the smaller budget to exercise the multi-chunk path as well.
    llm = OllamaSummarizer(chunk_chars=3000)
    print("Available models:", llm.list_models(), flush=True)
    short_article = (
        "This is a fictional news article for a software test. "
        "Cedar City Council approved a $2 million library renovation on March 12, 2026. "
        "The vote was seven to two. Construction is scheduled to begin on September 1, 2026 "
        "and finish in June 2027. Finance officer Mina Park said the money would come from "
        "existing capital reserves. The council did not approve a tax increase. "
        "The library will remain open during the work, except for a two-week closure in December. "
        "The renovation has not started yet."
    )
    with (ROOT / "True.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        long_index, long_article = next(
            (index, row) for index, row in enumerate(reader)
            if 3500 <= len(row["text"]) <= 5000
        )
    cases = [
        {"name": "short_synthetic", "source": "fictional test fixture",
         "title": "Council approves library renovation", "text": short_article,
         "review_points": ["$2 million", "March 12, 2026", "no tax increase", "renovation not yet started"]},
        {"name": "long_dataset_article", "source": f"True.csv:{long_index}",
         "title": long_article["title"], "text": long_article["text"],
         "review_points": ["all input chunks processed", "preserve article attribution, negation and numbers"]},
    ]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": llm.model,
        "base_url": llm.base_url,
        "scope": "Real inference smoke check; small samples do not establish summary quality.",
        "cases": [],
    }
    destination = ROOT / "artifacts" / "ollama_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    for case in cases:
        print(f"Summarizing {case['name']}...", flush=True)
        started = time.perf_counter()
        record = dict(case)
        try:
            result = llm.summarize(case["title"], case["text"], language="ko")
            if case["name"] == "long_dataset_article" and result.chunks < 2:
                raise RuntimeError("Long-article smoke check did not exercise multiple chunks.")
            record.update(status="ok", result=result.to_dict())
            print(result.summary, flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            record.update(status="error", error=str(error))
            exit_code = 1
            print(f"Failed: {error}", flush=True)
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        report["cases"].append(record)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Elapsed: {record['elapsed_seconds']} seconds", flush=True)
        if exit_code:
            break
    print(f"Saved review examples: {destination}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
