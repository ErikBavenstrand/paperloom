from collections.abc import Generator

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from paperloom.infrastructure.persistence.orm import Base


@pytest.fixture(scope="module")
def in_memory_db() -> Engine:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="module")
def in_memory_sqlite_session(in_memory_db: Engine) -> Generator[Session, None, None]:
    session_factory = sessionmaker(bind=in_memory_db)
    session = session_factory()

    try:
        yield session
    finally:
        session.close()
        in_memory_db.dispose()


@pytest.fixture(scope="module")
def in_memory_sqlite_session_factory(
    in_memory_db: Engine,
) -> Generator[sessionmaker[Session], None, None]:
    try:
        yield sessionmaker(bind=in_memory_db)
    finally:
        in_memory_db.dispose()
