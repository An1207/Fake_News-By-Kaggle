"""Initial articles and immutable analysis snapshots."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import LONGTEXT

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    longtext = sa.Text().with_variant(LONGTEXT(), "mysql")
    op.create_table("articles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("body", longtext, nullable=False),
        sa.Column("source_url", sa.String(2048), nullable=True),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_articles_created_id", "articles", ["created_at", "id"])
    op.create_table("analyses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("article_id", sa.String(36), sa.ForeignKey("articles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("article_version", sa.Integer(), nullable=False),
        sa.Column("article_title", sa.String(300), nullable=False),
        sa.Column("article_body", longtext, nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("classification", sa.JSON(), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'failed')", name="ck_analysis_status"),
    )
    op.create_index("ix_analyses_article_created", "analyses", ["article_id", "created_at"])
    op.create_index("ix_analyses_status", "analyses", ["status"])


def downgrade():
    op.drop_table("analyses")
    op.drop_table("articles")
