"""Adapter to existing news_ai modules; activated only with AI_ENABLED=true."""

from threading import Lock

from .config import Settings


class LocalAIService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._classifier = None
        self._classifiers = {}
        self._load_lock = Lock()

    def analyze(self, title: str, body: str, mode: str, language: str) -> dict:
        result = {"classification": None, "summary": None}
        if mode in {"classify", "both"}:
            from news_ai.model import load_classifier, predict

            with self._load_lock:
                for key, path in [("baseline", self.settings.baseline_classifier_path),
                                  ("transformer", self.settings.transformer_classifier_path)]:
                    if key not in self._classifiers and path.is_file():
                        self._classifiers[key] = load_classifier(path)
                selected = next((key for key, path in [
                    ("baseline", self.settings.baseline_classifier_path),
                    ("transformer", self.settings.transformer_classifier_path)]
                    if path.resolve() == self.settings.classifier_path.resolve()), None)
                if selected is None and self._classifier is None:
                    self._classifier = load_classifier(self.settings.classifier_path)
            models = {key: predict(bundle, title, body) for key, bundle in self._classifiers.items()}
            for key in ["baseline", "transformer"]:
                models.setdefault(key, {"status": "model_unavailable", "message": "모델 학습 파일이 아직 없습니다."})
            primary = models[selected] if selected else predict(self._classifier, title, body)
            comparable = all(value["status"] == "ok" for value in models.values())
            result["classification"] = {**primary, "models": models, "selected_model": selected,
                "agreement": models["baseline"]["label"] == models["transformer"]["label"] if comparable else None}
        if mode in {"summarize", "both"}:
            from news_ai.summarizer import OllamaSummarizer

            result["summary"] = OllamaSummarizer(
                model=self.settings.ollama_model, base_url=self.settings.ollama_base_url,
                timeout=self.settings.ollama_timeout,
            ).summarize(title, body, language).to_dict()
        return result

    def summarize_gpt(self, title: str, body: str, language: str) -> dict:
        from news_ai.openai_summary import OpenAISummarizer

        return OpenAISummarizer(self.settings.openai_api_key.get_secret_value(),
                                timeout=self.settings.openai_timeout,
                                max_output_tokens=self.settings.openai_max_output_tokens).summarize(title, body, language)
