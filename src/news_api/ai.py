"""Adapter to existing news_ai modules; activated only with AI_ENABLED=true."""

from threading import Lock

from .config import Settings


class LocalAIService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._classifier = None
        self._load_lock = Lock()

    def analyze(self, title: str, body: str, mode: str, language: str) -> dict:
        result = {"classification": None, "summary": None}
        if mode in {"classify", "both"}:
            from news_ai.model import load_classifier, predict

            with self._load_lock:
                if self._classifier is None:
                    self._classifier = load_classifier(self.settings.classifier_path)
            result["classification"] = predict(self._classifier, title, body)
        if mode in {"summarize", "both"}:
            from news_ai.summarizer import OllamaSummarizer

            result["summary"] = OllamaSummarizer(
                model=self.settings.ollama_model, base_url=self.settings.ollama_base_url,
                timeout=self.settings.ollama_timeout,
            ).summarize(title, body, language).to_dict()
        return result
