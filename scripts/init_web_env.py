"""Generate local development DB passwords without printing them."""

from pathlib import Path
import secrets

root = Path(__file__).resolve().parent.parent
destination = root / ".env"
if destination.exists():
    print("Existing .env preserved.")
else:
    values = [
        "MYSQL_HOST=127.0.0.1", "MYSQL_PORT=13306", "MYSQL_DATABASE=sai_news", "MYSQL_USER=sai_app",
        "MYSQL_PASSWORD=" + secrets.token_urlsafe(24),
        "MYSQL_ROOT_PASSWORD=" + secrets.token_urlsafe(32),
        "AI_ENABLED=false", "API_PORT=8011", "OLLAMA_MODEL=exaone3.5:2.4b", "OLLAMA_BASE_URL=http://localhost:11434",
        "OLLAMA_TIMEOUT=180", 'CORS_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]',
        "OPENAI_API_KEY=", "OPENAI_TIMEOUT=90", "OPENAI_MAX_OUTPUT_TOKENS=1000",
        "AUTOMATION_ENABLED=false", "AUTOMATION_API_KEY=",
    ]
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(values) + "\n")
    print("Created .env with project-specific passwords. Passwords are not displayed.")
