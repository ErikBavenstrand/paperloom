from types import TracebackType

from sqlalchemy.orm import Session, sessionmaker

from paperloom.application.ports.persistence.unit_of_work import AbstractUnitOfWork
from paperloom.infrastructure.persistence.repository import SqlAlchemyPaperRepository


class SqlAlchemyUnitOfWork(AbstractUnitOfWork):
    """A `SQLAlchemy` Unit of Work for managing transactions."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Initializes the `SqlAlchemyUnitOfWork`.

        Args:
            session_factory: The `SQLAlchemy` session factory.
        """
        self.session_factory = session_factory

    def __enter__(self) -> "SqlAlchemyUnitOfWork":
        """Enter the Unit of Work context.

        Returns:
            The Unit of Work.
        """
        self.session: Session = self.session_factory()
        self.papers = SqlAlchemyPaperRepository(self.session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the Unit of Work context.

        Args:
            exc_type: The exception type.
            exc_value: The exception instance.
            traceback: The traceback object.
        """
        super().__exit__(exc_type, exc_value, traceback)
        self.session.close()

    def commit(self) -> None:
        """Commit the transaction."""
        self.session.commit()

    def rollback(self) -> None:
        """Rollback the transaction."""
        self.session.rollback()
