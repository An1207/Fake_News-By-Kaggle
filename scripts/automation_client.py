"""Run the Make-compatible HTTP contract locally; preview only unless --execute."""

import argparse
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener

from news_ai.openai_summary import NoRedirect
from news_api.schemas import AutomationInput

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "integrations/make/request.example.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--execute", action="store_true", help="Explicitly submit to an enabled backend; default is offline preview")
    parser.add_argument("--gpt", action="store_true", help="Request paid GPT supplement after Ollama completes")
    parser.add_argument("--wait-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    payload = AutomationInput.model_validate_json(args.input.read_text(encoding="utf-8")).model_dump(mode="json")
    contract = json.loads((ROOT / "integrations/make/pipeline.json").read_text(encoding="utf-8"))
    base = args.base_url.rstrip("/")
    parsed = urlparse(base)
    if (parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})) or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        parser.error("Use an HTTPS base origin or local HTTP origin without credentials/path.")
    if not 1 <= args.wait_seconds <= 1800:
        parser.error("--wait-seconds must be between 1 and 1800")
    if not args.execute:
        print(json.dumps({"mode": "offline_preview", "pipeline_enabled": contract["enabled"],
                          "submit": base + contract["submit"]["path"], "request_id": payload["request_id"],
                          "body_characters": len(payload["body"]), "gpt_requested": args.gpt}, ensure_ascii=False))
        return 0
    key = os.getenv("AUTOMATION_API_KEY", "")
    if len(key) < 32:
        parser.error("Set AUTOMATION_API_KEY in process environment; do not pass it as a command-line argument")
    opener = build_opener(NoRedirect())

    def call(path, body=None):
        request = Request(base + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
                          headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=15) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}; check backend activation, authentication and request state.") from None
        except (URLError, TimeoutError):
            raise RuntimeError("No response. Check the saved request/job before resubmitting; API usage may already have occurred.") from None

    def wait(job, gpt=False):
        deadline = time.monotonic() + args.wait_seconds
        while True:
            result = job.get("gpt_summary") if gpt else job
            if result and result["status"] in contract["poll"]["terminal_statuses"]:
                return job
            if time.monotonic() >= deadline:
                raise RuntimeError("Polling timed out. Processing may continue; poll the saved analysis_id instead of submitting a new request.")
            time.sleep(3)
            job = call(contract["poll"]["path"].format(analysis_id=job["id"]))

    job = call(contract["submit"]["path"], payload)
    print("analysis_id=" + job["id"], flush=True)
    job = wait(job)
    if args.gpt and job["status"] == "completed":
        job = wait(call(contract["optional_gpt"]["path"].format(analysis_id=job["id"]), {}), gpt=True)
    print(json.dumps(job, ensure_ascii=False, indent=2))
    return 0 if job["status"] == "completed" and (not args.gpt or job["gpt_summary"]["status"] == "completed") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(str(error))
        raise SystemExit(1)
