from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_serializer, field_validator


class ArticleInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=20, max_length=59000)
    source_url: HttpUrl | None = None
    language: Literal["en", "ko", "unknown"] = "en"

    @field_validator("source_url")
    @classmethod
    def url_length(cls, value):
        if value is not None and len(str(value)) > 2048:
            raise ValueError("Source URL must not exceed 2048 characters.")
        return value


class ArticleUpdate(ArticleInput):
    expected_version: int = Field(ge=1)


class TimestampModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    @field_serializer("created_at", "updated_at", "started_at", "completed_at", check_fields=False)
    def serialize_time(self, value: datetime | None):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc).isoformat() if value.tzinfo is None else value.isoformat()


class ArticleOut(TimestampModel):
    id: str
    title: str
    body: str
    source_url: str | None
    language: str
    version: int
    created_at: datetime
    updated_at: datetime


class ArticleListItem(TimestampModel):
    id: str
    title: str
    preview: str
    source_url: str | None
    language: str
    version: int
    created_at: datetime
    updated_at: datetime


class ArticlePage(BaseModel):
    items: list[ArticleListItem]
    total: int
    page: int
    page_size: int


class AnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["classify", "summarize", "both"] = "both"
    language: Literal["ko", "en"] = "ko"


class AnalysisOut(TimestampModel):
    id: str
    article_id: str
    article_version: int
    article_title: str
    mode: str
    language: str
    status: str
    classification: dict | None
    summary: dict | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class AnalysisPage(BaseModel):
    items: list[AnalysisOut]
    total: int
    page: int
    page_size: int
