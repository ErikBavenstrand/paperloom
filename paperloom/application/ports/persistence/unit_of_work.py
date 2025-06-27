from abc import ABC, abstractmethod
from types import TracebackType

from paperloom.application.ports.persistence.repository import AbstractPaperRepository


class AbstractUnitOfWork(ABC):
    """Abstract Unit of Work for managing transactions."""

    papers: AbstractPaperRepository
    """A `Paper` domain object repository."""

    def __enter__(self) -> "AbstractUnitOfWork":
        """Enter the Unit of Work context.

        Returns:
            The Unit of Work.
        """
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
        self.rollback()

    @abstractmethod
    def commit(self) -> None:
        """Commit the transaction."""
        raise NotImplementedError

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the transaction."""
        raise NotImplementedError
