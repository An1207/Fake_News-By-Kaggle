from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utc_now() -> datetime:
    # MySQL DATETIME is stored in UTC; attach UTC when serializing responses.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (Index("ix_articles_created_id", "created_at", "id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT(), "mysql"))
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    language: Mapped[str] = mapped_column(String(8), default="en")
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now)
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="article", passive_deletes=True)


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_article_created", "article_id", "created_at"),
        Index("ix_analyses_status", "status"),
        CheckConstraint("status IN ('queued', 'running', 'completed', 'failed')", name="ck_analysis_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    article_id: Mapped[str] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    article_version: Mapped[int] = mapped_column(Integer)
    article_title: Mapped[str] = mapped_column(String(300))
    article_body: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT(), "mysql"))
    mode: Mapped[str] = mapped_column(String(16))
    language: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    classification: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    article: Mapped[Article] = relationship(back_populates="analyses")
