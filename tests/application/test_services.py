import datetime

import pytest

from cli import CategoryParseError
from paperloom.application.ports.paper_extractor import (
    AbstractCategoryExtractor,
    AbstractPaperExtractor,
    CategoryDTO,
    CategoryFetchError,
    PaperDTO,
)
from paperloom.application.ports.persistence.repository import CategoriesNotFoundError, PapersNotFoundError
from paperloom.application.ports.persistence.unit_of_work import AbstractPaperRepository, AbstractUnitOfWork
from paperloom.application.services import (
    NoCategoriesError,
    fetch_and_store_categories,
    fetch_and_store_historical_papers,
    fetch_and_store_latest_papers,
)
from paperloom.domain import model


# ---- Fake Implementations ----
class FakePapersRepository(AbstractPaperRepository):
    def __init__(self) -> None:
        self.categories: set[model.Category] = set()
        self.papers: set[model.Paper] = set()

    def upsert_categories(self, categories: set[model.Category]) -> None:
        self.categories ^= categories

    def get_category(self, category_identifier: model.CategoryIdentifier) -> model.Category | None:
        return next((c for c in self.categories if c.identifier == category_identifier), None)

    def delete_categories(self, category_identifiers: set[model.CategoryIdentifier]) -> None:
        missing = category_identifiers - {c.identifier for c in self.categories}
        if missing:
            raise CategoriesNotFoundError(missing)
        self.categories = {c for c in self.categories if c.identifier not in category_identifiers}

    def list_categories(self) -> list[model.Category]:
        return sorted(self.categories, key=lambda c: str(c.identifier))

    def upsert_papers(self, papers: set[model.Paper]) -> None:
        missing = {cat.identifier for p in papers for cat in p.categories} - {c.identifier for c in self.categories}
        if missing:
            raise CategoriesNotFoundError(missing)
        self.papers ^= papers

    def get_paper(self, arxiv_id: str) -> model.Paper | None:
        return next((p for p in self.papers if p.arxiv_id == arxiv_id), None)

    def delete_papers(self, arxiv_ids: set[str]) -> None:
        missing = arxiv_ids - {p.arxiv_id for p in self.papers}
        if missing:
            raise PapersNotFoundError(missing)
        self.papers = {p for p in self.papers if p.arxiv_id not in arxiv_ids}

    def list_papers(self, *, limit: int | None = None) -> list[model.Paper]:
        papers = sorted(self.papers, key=lambda p: p.arxiv_id)
        return papers if limit is None else papers[:limit]


class FakeUnitOfWork(AbstractUnitOfWork):
    def __init__(self) -> None:
        self.papers = FakePapersRepository()

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class FakeArXivCategoryExtractor(AbstractCategoryExtractor):
    def __init__(
        self,
        categories: set[CategoryDTO],
        *,
        fetch_error: bool = False,
        parse_error: bool = False,
    ) -> None:
        self.categories = categories
        self.fetch_error = fetch_error
        self.parse_error = parse_error

    def fetch_categories(self) -> set[CategoryDTO]:
        if self.fetch_error:
            raise CategoryFetchError
        if self.parse_error:
            raise CategoryParseError
        return self.categories


class FakePaperExtractor(AbstractPaperExtractor):
    def __init__(self, papers: list[PaperDTO]) -> None:
        self.papers = papers

    def _flatten_ids(self, categories: set[model.Category]) -> set[str]:
        ids = {str(c.identifier) for c in categories}
        for c in categories:
            ids.update(str(sub.identifier) for sub in c.subcategories)
        return ids

    def fetch_latest(self, categories: set[model.Category]) -> set[PaperDTO]:
        category_identifiers = self._flatten_ids(categories)
        return {
            paper
            for paper in self.papers
            if any(category_identifier in paper.categories for category_identifier in category_identifiers)
        }

    def fetch_historical(
        self, categories: set[model.Category], from_date: datetime.date | None, to_date: datetime.date | None
    ) -> set[PaperDTO]:
        category_identifiers = self._flatten_ids(categories)
        return {
            paper
            for paper in self.papers
            if any(category_identifier in paper.categories for category_identifier in category_identifiers)
            and (from_date is None or paper.published_at >= from_date)
            and (to_date is None or paper.published_at <= to_date)
        }


# ---- Fixtures ----
@pytest.fixture
def uow() -> FakeUnitOfWork:
    return FakeUnitOfWork()


@pytest.fixture
def category_dtos() -> set[CategoryDTO]:
    return {
        CategoryDTO("cs", "AI", "Computer Science", "Artificial Intelligence", "AI papers"),
        CategoryDTO("math", None, "Mathematics", None, None),
    }


@pytest.fixture
def successful_category_extractor(category_dtos: set[CategoryDTO]) -> FakeArXivCategoryExtractor:
    return FakeArXivCategoryExtractor(category_dtos)


@pytest.fixture
def sample_papers() -> list[PaperDTO]:
    base = datetime.date(2023, 1, 1)
    category_list = ["cs.AI", "cs.AI", "cs.ML", "cs.LG", "cs.CV", "cs.CV"]
    return [
        PaperDTO(
            arxiv_id=f"1234.5678v{i}",
            title=f"Paper {i}",
            abstract="",
            published_at=base + datetime.timedelta(days=i),
            categories={category_list[i]},
        )
        for i in range(len(category_list))
    ]


@pytest.fixture
def paper_extractor(sample_papers: list[PaperDTO]) -> FakePaperExtractor:
    return FakePaperExtractor(sample_papers)


@pytest.fixture
def populated_uow(uow: FakeUnitOfWork) -> FakeUnitOfWork:
    subcategories = {
        model.Category(model.CategoryIdentifier.from_string(sub)) for sub in ["cs.AI", "cs.ML", "cs.LG", "cs.CV"]
    }
    categories = {model.Category(model.CategoryIdentifier.from_string("cs"), subcategories=subcategories)}
    uow.papers.upsert_categories(categories | subcategories)
    return uow


# ---- Tests ----
class TestFetchAndStoreCategories:
    def test_success(self, uow: FakeUnitOfWork, successful_category_extractor: FakeArXivCategoryExtractor) -> None:
        fetch_and_store_categories(uow, successful_category_extractor)
        stored = uow.papers.list_categories()
        expected = sorted(
            [
                model.Category(
                    model.CategoryIdentifier("cs", "AI"), "Computer Science", "Artificial Intelligence", "AI papers"
                ),
                model.Category(model.CategoryIdentifier("math", None), "Mathematics", None, None),
            ],
            key=lambda c: str(c.identifier),
        )
        assert stored == expected

    @pytest.mark.parametrize(
        ("fetch_err", "parse_err", "exc"),
        [
            (True, False, CategoryFetchError),
            (False, True, CategoryParseError),
        ],
        ids=["fetch_error", "parse_error"],
    )
    def test_errors(
        self,
        uow: FakeUnitOfWork,
        fetch_err: bool,
        parse_err: bool,
        exc: type[Exception],
    ) -> None:
        extractor = FakeArXivCategoryExtractor(set(), fetch_error=fetch_err, parse_error=parse_err)
        with pytest.raises(exc):
            fetch_and_store_categories(uow, extractor)


class TestFetchAndStoreLatestPapers:
    @pytest.mark.parametrize(
        ("category_strings", "expected_ids"),
        [
            (None, {f"1234.5678v{i}" for i in range(6)}),
            ({"cs.AI"}, {"1234.5678v0", "1234.5678v1"}),
        ],
        ids=["all", "cs.AI"],
    )
    def test_success(
        self,
        populated_uow: FakeUnitOfWork,
        paper_extractor: FakePaperExtractor,
        category_strings: set[str] | None,
        expected_ids: set[str],
    ) -> None:
        fetch_and_store_latest_papers(populated_uow, paper_extractor, category_strings=category_strings)
        stored = populated_uow.papers.list_papers(limit=None)
        assert {p.arxiv_id for p in stored} == expected_ids

    def test_missing_category_raises(self, uow: FakeUnitOfWork, paper_extractor: FakePaperExtractor) -> None:
        with pytest.raises(CategoriesNotFoundError):
            fetch_and_store_latest_papers(uow, paper_extractor, category_strings={"cs"})

    def test_no_categories_raises(self, uow: FakeUnitOfWork, paper_extractor: FakePaperExtractor) -> None:
        with pytest.raises(NoCategoriesError):
            fetch_and_store_latest_papers(uow, paper_extractor, category_strings=None)


class TestFetchHistorical:
    @pytest.mark.parametrize(
        ("from_date", "to_date", "expected_ids"),
        [
            (None, None, {f"1234.5678v{i}" for i in range(6)}),
            (datetime.date(2023, 1, 2), datetime.date(2023, 1, 4), {"1234.5678v1", "1234.5678v2", "1234.5678v3"}),
            (None, datetime.date(2023, 1, 3), {"1234.5678v0", "1234.5678v1", "1234.5678v2"}),
            (datetime.date(2023, 1, 5), None, {"1234.5678v4", "1234.5678v5"}),
        ],
        ids=["all", "mid_range", "up_to", "from_only"],
    )
    def test_historical_date_filter(
        self,
        populated_uow: FakeUnitOfWork,
        paper_extractor: FakePaperExtractor,
        from_date: datetime.date | None,
        to_date: datetime.date | None,
        expected_ids: set[str],
    ) -> None:
        fetch_and_store_historical_papers(
            populated_uow, paper_extractor, category_strings={"cs"}, from_date=from_date, to_date=to_date
        )
        stored = populated_uow.papers.list_papers(limit=None)
        assert {p.arxiv_id for p in stored} == expected_ids

    def test_historical_missing_category_raises(
        self,
        uow: FakeUnitOfWork,
        paper_extractor: FakePaperExtractor,
    ) -> None:
        with pytest.raises(CategoriesNotFoundError):
            fetch_and_store_historical_papers(uow, paper_extractor, category_strings={"cs"})
