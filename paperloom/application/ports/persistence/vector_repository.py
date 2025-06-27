import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass

from paperloom.domain import model


class VectoryRepositoryInsertionError(Exception):
    """Exception raised when there is an error inserting embeddings into the vector repository."""


class VectoryRepositoryDeletionError(Exception):
    """Exception raised when there is an error deleting embeddings from the vector repository."""


class VectoryRepositoryQueryError(Exception):
    """Exception raised when there is an error querying the vector repository."""


@dataclass
class VectorSearchFilter:
    """Technology-agnostic filter for vector search."""

    category_identifiers: list[model.CategoryIdentifier] | None = None
    """List of category identifiers to filter the search results by."""

    published_after: datetime.date | None = None
    """Date to filter the search results by, only papers published on or after this date will be included."""

    published_before: datetime.date | None = None
    """Date to filter the search results by, only papers published on or before this date will be included."""


class AbstractPaperVectorRepository(ABC):
    """Abstract base class for a `Paper` vector repository."""

    @abstractmethod
    def insert_embeddings(self, embeddings: list[list[float]], papers: list[model.Paper]) -> None:
        """Insert embeddings and metadata into the vector repository.

        Args:
            embeddings: List of embeddings to insert.
            papers: List of `Paper` domain objects corresponding to the embeddings.

        Raises:
            VectoryRepositoryInsertionError: If there is an error inserting the embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_embeddings(self, arxiv_ids: list[str]) -> None:
        """Delete embeddings from the vector repository.

        Args:
            arxiv_ids: List of IDs of the embeddings to delete.

        Raises:
            VectoryRepositoryDeletionError: If there is an error deleting the embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def query_embedding(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        filters: VectorSearchFilter | None,
    ) -> list[str]:
        """Query the vector repository for similar embeddings.

        Args:
            query_embedding: The embedding to query against.
            top_k: The number of similar embeddings to return.
            filters: Optional filters to apply to the query.

        Raises:
            VectoryRepositoryQueryError: If there is an error querying the vector repository.

        Returns:
            List of metadata for the top_k similar embeddings.
        """
        raise NotImplementedError
