"""Shared text handling used identically during training and prediction."""

import hashlib
import html
import re
import unicodedata


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(str(value)))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def fingerprint(value: str) -> str:
    # Punctuation/case/whitespace differences do not create new articles.
    normalized = re.sub(r"[^\w]+", " ", normalize_text(value).casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def model_text(title: str, body: str) -> str:
    body = normalize_text(body)
    # Remove the most obvious source shortcut without changing the summary input.
    body = re.sub(r"^.{0,120}?\(Reuters\)\s*[-–—]\s*", "", body, flags=re.I)
    combined = normalize_text(f"{title}\n{body}")
    combined = re.sub(r"\bReuters\b", " ", combined, flags=re.I)
    combined = re.sub(r"https?://\S+|www\.\S+", " ", combined)
    return normalize_text(combined)


def word_count(value: str) -> int:
    return len(re.findall(r"\b[A-Za-z]+\b", value))
