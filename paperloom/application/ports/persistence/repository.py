from abc import ABC, abstractmethod

from paperloom.domain import model


class CategoriesNotFoundError(Exception):
    """Exception raised when categories are not found in the repository."""

    def __init__(self, category_identifiers: list[model.CategoryIdentifier]) -> None:
        """Initializes the `CategoriesNotFoundError` exception.

        Args:
            category_identifiers: A list of `CategoryIdentifier` domain objects representing the categories
                that were not found.
        """
        super().__init__(f"Categories {category_identifiers!r} not found in the repository.")


class PapersNotFoundError(Exception):
    """Exception raised when papers and not found in the repository."""

    def __init__(self, arxiv_ids: list[str]) -> None:
        """Initializes the `PapersNotFoundError` exception.

        Args:
            arxiv_ids: The ArXiv IDs of the papers that were not found.
        """
        super().__init__(f"Papers with ArXiv IDs {arxiv_ids!r} not found in the repository.")


class AbstractPaperRepository(ABC):
    """Abstract `Paper` domain object repository."""

    @abstractmethod
    def upsert_categories(self, categories: list[model.Category]) -> None:
        """Upserts a list of `Category` domain objects into the database.

        Args:
            categories: A list of `Category` domain objects to upsert.
        """
        raise NotImplementedError

    @abstractmethod
    def get_category(self, category_identifier: model.CategoryIdentifier) -> model.Category | None:
        """Fetches a `Category` domain object from the database.

        Args:
            category_identifier: The `CategoryIdentifier` domain object.

        Returns:
            The `Category` domain object if found, otherwise `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_categories(self, category_identifiers: list[model.CategoryIdentifier]) -> None:
        """Deletes the specified `Category` domain objects from the database.

        Args:
            category_identifiers: A list of `CategoryIdentifier` domain objects representing the categories to delete.

        Raises:
            CategoryNotFoundError: If any of the categories are not found in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def list_categories(self) -> list[model.Category]:
        """Lists all `Category` domain objects in the database.

        Returns:
            A list of `Category` domain objects.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_papers(self, papers: list[model.Paper]) -> None:
        """Upserts a list of `Paper` domain objects into the database.

        Args:
            papers: A list of `Paper` domain objects to upsert.

        Raises:
            CategoriesNotFoundError: If any of the categories are not found in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def get_paper(self, arxiv_id: str) -> model.Paper | None:
        """Fetches a `Paper` domain object from the database.

        Args:
            arxiv_id: The ArXiv ID of the paper.

        Returns:
            The `Paper` domain object if found, otherwise `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_papers(self, arxiv_ids: list[str]) -> None:
        """Deletes the specified `Paper` domain objects from the database.

        Args:
            arxiv_ids: A list of ArXiv IDs representing the papers to delete.

        Raises:
            PaperNotFoundError: If any of the papers are not found in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def list_papers(self, *, limit: int | None) -> list[model.Paper]:
        """Lists all `Paper` domain objects in the database.

        Args:
            limit: The maximum number of papers to return.

        Returns:
            A list of `Paper` domain objects.
        """
        raise NotImplementedError
