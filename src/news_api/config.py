from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL
from news_ai.selection import recommended_classifier

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    mysql_host: str = "127.0.0.1"
    mysql_port: int = Field(default=13306, ge=1, le=65535)
    mysql_database: str = "sai_news"
    mysql_user: str = "sai_app"
    mysql_password: SecretStr = SecretStr("")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    ai_enabled: bool = False
    classifier_path: Path = Field(default_factory=lambda: recommended_classifier(ROOT / "artifacts"))
    baseline_classifier_path: Path = ROOT / "artifacts" / "classifier.joblib"
    transformer_classifier_path: Path = ROOT / "artifacts" / "eurobert" / "classifier.joblib"
    ollama_model: str = "exaone3.5:2.4b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = Field(default=180, gt=0)

    @property
    def database_url(self) -> URL:
        return URL.create(
            "mysql+pymysql", username=self.mysql_user,
            password=self.mysql_password.get_secret_value(), host=self.mysql_host,
            port=self.mysql_port, database=self.mysql_database, query={"charset": "utf8mb4"},
        )
