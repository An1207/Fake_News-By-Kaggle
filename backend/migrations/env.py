from alembic import context

from news_api.config import Settings
from news_api.db import Base, make_engine
from news_api import models  # Register metadata for future autogeneration.

if context.is_offline_mode():
    context.configure(url=Settings().database_url, target_metadata=Base.metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = make_engine(Settings())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
