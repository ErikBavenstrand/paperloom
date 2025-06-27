import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from paperloom.domain import model


class CategoryFetchError(Exception):
    """Raised when fetching the categories fails."""


class CategoryParseError(Exception):
    """Raised when parsing the categories fails."""


class PaperMissingFieldError(Exception):
    """Raised when a required field is missing in the paper."""

    def __init__(self, entry: dict[str, Any]) -> None:
        """Initializes the error with the entry that is missing a field.

        Args:
            entry: The entry that is missing a required field.
        """
        super().__init__(f"Missing required field in entry: {entry.keys()}")
        self.entry = entry


@dataclass(frozen=True)
class PaperDTO:
    """Data Transfer Object for an ArXiv paper."""

    arxiv_id: str
    """The ArXiv ID of the paper."""

    title: str
    """The title of the paper."""

    abstract: str
    """The abstract of the paper."""

    published_at: datetime.date
    """The date the paper was published."""

    categories: list[str]
    """The categories the paper belongs to."""

    def __eq__(self, other: object) -> bool:
        """Check if two `PaperDTO` objects are equal.

        Args:
            other: The other object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, PaperDTO):
            return NotImplemented
        return self.arxiv_id == other.arxiv_id

    def __hash__(self) -> int:
        """Return the hash value of the `PaperDTO` object.

        Returns:
            The hash value of the `PaperDTO` object.
        """
        return hash(self.arxiv_id)


@dataclass(frozen=True)
class CategoryDTO:
    """Data Transfer Object for an ArXiv category."""

    archive: str
    """The archive to which the category belongs."""

    subcategory: str | None
    """The subcategory of the category."""

    archive_name: str | None
    """The name of the category archive."""

    category_name: str | None
    """The name of the category."""

    description: str | None
    """The description of the category."""


class AbstractPaperExtractor(ABC):
    """Abstract paper extractor for fetching papers."""

    @abstractmethod
    def fetch_latest(self, categories: list[model.Category]) -> list[PaperDTO]:
        """Fetches the latest papers for the given categories.

        Args:
            categories: The `Category` domain objects to filter the papers by.

        Raises:
            PaperMissingFieldError: If a required field is missing in the paper.

        Returns:
            A list of `PaperDTO` objects representing the papers.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_historical(
        self,
        categories: list[model.Category],
        from_date: datetime.date | None,
        to_date: datetime.date | None,
    ) -> list[PaperDTO]:
        """Fetches historical papers for the given categories and time range.

        Args:
            categories: The `Cateogry` domain objjects to filter the papers by.
            from_date: Optional from date filter (inclusive).
            to_date: Optional to date filter (inclusive).

        Raises:
            PaperMissingFieldError: If a required field is missing in the paper.

        Returns:
            A list of `PaperDTO` objects representing the papers.
        """
        raise NotImplementedError


class AbstractCategoryExtractor(ABC):
    """Abstract category extractor for fetching categories."""

    @abstractmethod
    def fetch_categories(self) -> list[CategoryDTO]:
        """Fetches all categories.

        Raises:
            CategoryFetchError: If fetching categories fails.
            CategoryParseError: If parsing categories fails.

        Returns:
            A list of `CategoryDTO` objects representing the categories.
        """
        raise NotImplementedError
