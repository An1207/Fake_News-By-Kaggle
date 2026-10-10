"""GPT supplements and idempotent automation intake; existing summaries preserved."""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("gpt_summaries",
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'failed')", name="ck_gpt_status"),
    )
    op.create_index("ix_gpt_status", "gpt_summaries", ["status"])
    op.create_table("automation_requests",
        sa.Column("request_hash", sa.String(64), primary_key=True),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade():
    op.drop_table("automation_requests")
    op.drop_table("gpt_summaries")
