import datetime

from paperloom.application.ports.paper_extractor import AbstractCategoryExtractor, AbstractPaperExtractor
from paperloom.application.ports.persistence.repository import CategoriesNotFoundError
from paperloom.application.ports.persistence.unit_of_work import AbstractUnitOfWork
from paperloom.domain import model


class NoCategoriesError(Exception):
    """Exception raised when no categories are found in the repository."""

    def __init__(self) -> None:
        """Initializes the `NoCategoriesError` exception."""
        super().__init__("No categories found in the repository.")


LEGACY_TO_CANONICAL_CATEGORIES = {
    "chem-ph": "physics.chem-ph",
    "alg-geom": "math.AG",
    "cmp-lg": "cs.CL",
    "acc-phys": "physics.acc-ph",
    "adap-org": "nlin.AO",
    "chao-dyn": "nlin.CD",
    "ao-sci": "physics.ao-ph",
    "plasm-ph": "physics.plasm-ph",
    "supr-con": "cond-mat.supr-con",
    "funct-an": "math.FA",
    "dg-ga": "math.DG",
    "patt-sol": "nlin.PS",
    "q-alg": "math.QA",
    "bayes-an": "physics.data-an",
    "mtrl-th": "cond-mat.mtrl-sci",
    "comp-gas": "nlin.CG",
    "solv-int": "nlin.SI",
    "atom-ph": "physics.atom-ph",
}

CANONICAL_TO_LEGACY_CATEGORIES = {v: k for k, v in LEGACY_TO_CANONICAL_CATEGORIES.items()}


def _get_legacy_categories(categories: set[model.Category]) -> set[model.Category]:
    """Get legacy categories from the given set of categories.

    Some categories in the repository may have legacy identifiers that need to be converted
    to their canonical form. This function filters the categories and converts them to their
    legacy identifiers if they exist in the `CANONICAL_TO_LEGACY_CATEGORIES` mapping.

    Args:
        categories: A set of `Category` domain objects to filter and convert.

    Returns:
        A set of `Category` domain objects with legacy identifiers.
    """
    return {
        model.Category(
            identifier=model.CategoryIdentifier.from_string(
                CANONICAL_TO_LEGACY_CATEGORIES[str(category.identifier)],
            ),
            archive_name=category.archive_name,
            category_name=category.category_name,
            description=category.description,
        )
        for category in categories
        if str(category.identifier) in CANONICAL_TO_LEGACY_CATEGORIES
    }


def fetch_and_store_categories(
    uow: AbstractUnitOfWork,
    category_extractor: AbstractCategoryExtractor,
) -> set[model.Category]:
    """Fetches categories and stores them in the repository.

    Args:
        uow: The unit of work to manage repository transactions.
        category_extractor: The category extractor to fetch categories from.

    Raises:
        CategoryFetchError: If fetching categories fails.
        CategoryParseError: If parsing categories fails.

    Returns:
        A set of all `Category` domain objects fetched from the extractor.
    """
    category_dtos = category_extractor.fetch_categories()
    categories = {
        model.Category(
            identifier=model.CategoryIdentifier(archive=category.archive, subcategory=category.subcategory),
            archive_name=category.archive_name,
            category_name=category.category_name,
            description=category.description,
        )
        for category in category_dtos
    }

    with uow:
        uow.papers.upsert_categories(categories)
        uow.commit()

    return categories


def _resolve_categories(uow: AbstractUnitOfWork, category_strings: set[str] | None) -> set[model.Category]:
    """Resolves the categories to be used for fetching papers.

    If `categories` is None, it fetches all archives (top-level categories) from the repository.

    Args:
        uow: The unit of work to manage repository transactions.
        category_strings: A set of category strings to filter the papers by, (e.g., ["cs.AI", "math.ST"]).

    Raises:
        CategoriesNotFoundError: If any of the `category_strings` categories does not exist in the repository.
        NoCategoriesError: If `categories` is None and no categories are found in the repository.

    Returns:
        A set of `Category` domain objects representing the categories to be used for fetching papers.
    """
    if category_strings is not None:
        categories = set()
        for category_string in category_strings:
            category_identifier = model.CategoryIdentifier.from_string(category_string)
            category = uow.papers.get_category(category_identifier)
            if category is None:
                raise CategoriesNotFoundError({category_identifier})
            categories.add(category)

        return categories

    categories = uow.papers.list_categories()
    if not categories:
        raise NoCategoriesError

    return set(categories)


def fetch_and_store_latest_papers(
    uow: AbstractUnitOfWork,
    paper_extractor: AbstractPaperExtractor,
    *,
    category_strings: set[str] | None,
) -> set[model.Paper]:
    """Fetches the latest papers and stores them in the repository.

    It also enriches the categories of the papers with the ones stored in the repository, if they exist.
    Otherwise, it creates new categories and stores them in the repository.

    Args:
        uow: The unit of work to manage repository transactions.
        paper_extractor: The paper extractor to fetch papers with.
        category_strings: A set of category strings to filter the papers by, (e.g., {"cs.AI", "math.ST"}).
            If None, fetches papers from all categories.

    Raises:
        PaperMissingFieldError: If a required field is missing in the paper.
        NoCategoriesError: If `categories` is None and no categories are found in the repository.

    Returns:
        A set of `Paper` domain objects representing the papers fetched from the extractor.
    """
    with uow:
        categories = _resolve_categories(uow, category_strings)
        categories |= _get_legacy_categories(categories)

        paper_dtos = paper_extractor.fetch_latest(categories)

        enriched_category_mapping = {str(category.identifier): category for category in uow.papers.list_categories()}
        papers = {
            model.Paper(
                arxiv_id=paper_dto.arxiv_id,
                title=paper_dto.title,
                abstract=paper_dto.abstract,
                published_at=paper_dto.published_at,
                categories={
                    enriched_category_mapping[LEGACY_TO_CANONICAL_CATEGORIES.get(category_str, category_str)]
                    for category_str in paper_dto.categories
                },
            )
            for paper_dto in paper_dtos
        }
        uow.papers.upsert_papers(papers)
        uow.commit()

    return papers


def fetch_and_store_historical_papers(
    uow: AbstractUnitOfWork,
    paper_extractor: AbstractPaperExtractor,
    *,
    category_strings: set[str] | None,
    from_date: datetime.date | None = None,
    to_date: datetime.date | None = None,
) -> set[model.Paper]:
    """Fetches historical papers and stores them in the repository.

    It also enriches the categories of the papers with the ones stored in the repository, if they exist.
    Otherwise, it creates new categories and stores them in the repository.

    Args:
        uow: The unit of work to manage repository transactions.
        paper_extractor: The paper extractor to fetch papers with.
        category_strings: A set of category strings to filter the papers by, (e.g., {"cs.AI", "math.ST"}).
            If None, fetches papers from all categories.
        from_date: Optional from date filter (inclusive).
        to_date: Optional to date filter (inclusive).

    Raises:
        PaperMissingFieldError: If a required field is missing in the paper.
        NoCategoriesError: If `categories` is None and no categories are found in the repository.

    Returns:
        A set of `Paper` domain objects representing the historical papers fetched from the extractor.
    """
    with uow:
        categories = _resolve_categories(uow, category_strings)
        categories |= _get_legacy_categories(categories)

        paper_dtos = paper_extractor.fetch_historical(categories, from_date, to_date)

        enriched_category_mapping = {str(category.identifier): category for category in uow.papers.list_categories()}
        papers = {
            model.Paper(
                arxiv_id=paper_dto.arxiv_id,
                title=paper_dto.title,
                abstract=paper_dto.abstract,
                published_at=paper_dto.published_at,
                categories={
                    enriched_category_mapping[LEGACY_TO_CANONICAL_CATEGORIES.get(category_str, category_str)]
                    for category_str in paper_dto.categories
                },
            )
            for paper_dto in paper_dtos
        }
        uow.papers.upsert_papers(papers)
        uow.commit()

    return papers
