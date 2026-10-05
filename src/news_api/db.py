from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import Settings


class Base(DeclarativeBase):
    pass


def make_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url, pool_pre_ping=True, pool_size=5, max_overflow=5,
        hide_parameters=True, connect_args={"connect_timeout": 5},
    )


def make_sessions(engine: Engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
