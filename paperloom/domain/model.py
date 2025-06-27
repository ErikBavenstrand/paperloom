import datetime
from dataclasses import dataclass, field


class InvalidCategoryError(Exception):
    """Custom exception for invalid category strings."""

    def __init__(self, category_string: str) -> None:
        """Initialize the exception with a category string.

        Args:
            category_string: The invalid category string that caused the error.
        """
        super().__init__(f"Invalid category string: {category_string}")
        self.category_string = category_string


@dataclass(frozen=True)
class Paper:
    """Domain object for an ArXiv paper."""

    arxiv_id: str
    """The ArXiv ID of the paper."""

    title: str
    """The title of the paper."""

    abstract: str
    """The abstract of the paper."""

    published_at: datetime.date
    """The date the paper was published."""

    categories: list["Category"] = field(default_factory=list)
    """The categories the paper belongs to."""

    @property
    def published_at_int(self) -> int:
        """Return the published date as an integer.

        Returns:
            The published date as an integer in YYYYMMDD format.
        """
        return int(self.published_at.strftime("%Y%m%d"))

    @property
    def summary_url(self) -> str:
        """Return the URL to the summary of the paper.

        Returns:
            The URL to the summary of the paper.
        """
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def pdf_url(self) -> str:
        """Return the URL to the PDF of the paper.

        Returns:
            The URL to the PDF of the paper.
        """
        return f"https://arxiv.org/pdf/{self.arxiv_id}"

    @property
    def html_url(self) -> str:
        """Return the URL to the HTML version of the paper.

        Returns:
            The URL to the HTML version of the paper.
        """
        return f"https://arxiv.org/html/{self.arxiv_id}"

    def __repr__(self) -> str:
        """Return the string representation of the `Paper` domain object.

        Returns:
            The string representation of the `Paper` domain object.
        """
        return (
            f"Paper(arxiv_id={self.arxiv_id!r}, title={self.title!r}, "
            f"published_at={self.published_at!r}, categories={self.categories!r}, ...)"
        )

    def __eq__(self, other: object) -> bool:
        """Check if two `Paper` objects are equal.

        Args:
            other: The other object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, Paper):
            return NotImplemented
        return self.arxiv_id == other.arxiv_id

    def __hash__(self) -> int:
        """Return the hash value of the `Paper` object.

        Returns:
            The hash value of the `Paper` object.
        """
        return hash(self.arxiv_id)


@dataclass(frozen=True)
class CategoryIdentifier:
    """Domain object for an ArXiv category identifier."""

    archive: str
    """The archive to which the category belongs (e.g., "astro-ph")."""

    subcategory: str | None = None
    """The subcategory of the category (e.g., "SR" for "astro-ph.SR")."""

    @staticmethod
    def from_string(category_string: str) -> "CategoryIdentifier":
        """Create a `CategoryIdentifier` domain object from a string.

        Args:
            category_string: The category string in the format "archive.subcategory".

        Raises:
            InvalidCategoryError: If the category string is invalid.

        Returns:
            The `CategoryIdentifier` domain object.
        """
        parts = category_string.split(".")
        if len(parts) < 1 or len(parts) > 2:
            raise InvalidCategoryError(category_string)
        return CategoryIdentifier(*parts)

    def __str__(self) -> str:
        """Return the string representation of the `CategoryIdentifier` domain object.

        Returns:
            The string representation of the `CategoryIdentifier` domain object.
        """
        return f"{self.archive}.{self.subcategory}" if self.subcategory else self.archive

    def __repr__(self) -> str:
        """Return the string representation of the `CategoryIdentifier` domain object.

        Returns:
            The string representation of the `CategoryIdentifier` domain object.
        """
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """Check if two `CategoryIdentifier` objects are equal.

        Args:
            other: The other object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, CategoryIdentifier):
            return NotImplemented
        return self.archive == other.archive and self.subcategory == other.subcategory

    def __hash__(self) -> int:
        """Return the hash value of the `CategoryIdentifier` object.

        Returns:
            The hash value of the `CategoryIdentifier` object.
        """
        return hash((self.archive, self.subcategory))


@dataclass(frozen=True)
class Category:
    """Domain object for an ArXiv category."""

    identifier: CategoryIdentifier
    """The archive to which the category belongs (e.g., "astro-ph")."""

    archive_name: str | None = None
    """The name of the category archive (e.g., "Astrophysics")."""

    category_name: str | None = None
    """The name of the category (e.g., "Solar and Stellar Astrophysics")."""

    description: str | None = None
    """The description of the category."""

    subcategories: list["Category"] = field(default_factory=list)
    """List of subcategories."""

    def __repr__(self) -> str:
        """Return the string representation of the `Category` domain object.

        Returns:
            The string representation of the `Category` domain object.
        """
        return (
            f"Category(identifier={self.identifier!r}, archive_name={self.archive_name!r}, "
            f"category_name={self.category_name!r}, description={self.description!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Check if two `Category` objects are equal.

        Args:
            other: The other object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, Category):
            return NotImplemented
        return self.identifier == other.identifier

    def __hash__(self) -> int:
        """Return the hash value of the `Category` object.

        Returns:
            The hash value of the `Category` object.
        """
        return hash(self.identifier)
