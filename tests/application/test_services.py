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
from paperloom.application.services import fetch_and_store_categories, fetch_and_store_latest_papers
from paperloom.domain import model


class FakePapersRepository(AbstractPaperRepository):
    def __init__(self) -> None:
        self.categories: set[model.Category] = set()
        self.papers: set[model.Paper] = set()

    def upsert_categories(self, categories: list[model.Category]) -> None:
        for category in categories:
            if category not in self.categories:
                self.categories.add(category)
            else:
                self.categories.remove(category)
                self.categories.add(category)

    def get_category(self, category_identifier: model.CategoryIdentifier) -> model.Category | None:
        for category in self.categories:
            if category.identifier == category_identifier:
                return category
        return None

    def get_subcategories(self, archive: str) -> list[model.Category]:
        return [
            category
            for category in self.categories
            if category.identifier.archive == archive and category.identifier.subcategory is not None
        ]

    def delete_categories(self, category_identifiers: list[model.CategoryIdentifier]) -> None:
        matching_categories = [category for category in self.categories if category.identifier in category_identifiers]
        missing_categories = [
            category_identifier
            for category_identifier in category_identifiers
            if category_identifier not in [category.identifier for category in matching_categories]
        ]

        if missing_categories:
            raise CategoriesNotFoundError(missing_categories)

        for category in matching_categories:
            self.categories.remove(category)

    def list_categories(self) -> list[model.Category]:
        return sorted(self.categories, key=lambda x: str(x.identifier))

    def upsert_papers(self, papers: list[model.Paper]) -> None:
        missing_categories = [
            category.identifier for paper in papers for category in paper.categories if category not in self.categories
        ]
        if missing_categories:
            raise CategoriesNotFoundError(missing_categories)

        for paper in papers:
            if paper not in self.papers:
                self.papers.add(paper)
            else:
                self.papers.remove(paper)
                self.papers.add(paper)

    def get_paper(self, arxiv_id: str) -> model.Paper | None:
        for paper in self.papers:
            if paper.arxiv_id == arxiv_id:
                return paper
        return None

    def delete_papers(self, arxiv_ids: list[str]) -> None:
        matching_papers = [paper for paper in self.papers if paper.arxiv_id in arxiv_ids]
        missing_papers = [
            arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in [paper.arxiv_id for paper in matching_papers]
        ]
        if missing_papers:
            raise PapersNotFoundError(missing_papers)

        for paper in matching_papers:
            self.papers.remove(paper)

    def list_papers(self, *, limit: int | None) -> list[model.Paper]:
        sorted_papers = sorted(self.papers, key=lambda x: x.arxiv_id)
        if limit is None:
            return sorted_papers
        return sorted_papers[:limit]


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
        categories: list[CategoryDTO],
        *,
        fetch_error: bool | None = None,
        parse_error: bool | None = None,
    ) -> None:
        self.categories = categories
        self.fetch_error = CategoryFetchError if fetch_error else None
        self.parse_error = CategoryParseError if parse_error else None

    def fetch_categories(self) -> list[CategoryDTO]:
        if self.fetch_error:
            raise self.fetch_error

        if self.parse_error:
            raise self.parse_error

        return self.categories


class FakePaperExtractor(AbstractPaperExtractor):
    def __init__(self, papers: list[PaperDTO]) -> None:
        self.papers = papers

    def fetch_latest(self, categories: list[model.Category]) -> list[PaperDTO]:
        category_identifiers = [category.identifier for category in categories]
        return [
            paper
            for paper in self.papers
            if any(str(category_identifier) in paper.categories for category_identifier in category_identifiers)
            or any(
                category_identifier.subcategory is None
                and category_identifier.archive in {categories.split(".")[0] for categories in paper.categories}
                for category_identifier in category_identifiers
            )
        ]

    def fetch_historical(
        self,
        categories: list[model.Category],
        from_date: datetime.date | None,
        to_date: datetime.date | None,
    ) -> list[PaperDTO]:
        raise NotImplementedError


class TestFetchAndStoreCategories:
    def test_fetch_categories_success(self) -> None:
        fake_categories = [
            CategoryDTO(
                archive="cs",
                subcategory="AI",
                archive_name="Computer Science",
                category_name="Artificial Intelligence",
                description="AI papers",
            ),
            CategoryDTO(
                archive="math",
                subcategory=None,
                archive_name="Mathematics",
                category_name=None,
                description=None,
            ),
        ]
        fake_extractor = FakeArXivCategoryExtractor(fake_categories)
        fake_uow = FakeUnitOfWork()

        fetch_and_store_categories(fake_uow, fake_extractor)

        expected_categories = [
            model.Category(
                identifier=model.CategoryIdentifier(archive="cs", subcategory="AI"),
                archive_name="Computer Science",
                category_name="Artificial Intelligence",
                description="AI papers",
            ),
            model.Category(
                identifier=model.CategoryIdentifier(archive="math", subcategory=None),
                archive_name="Mathematics",
                category_name=None,
                description=None,
            ),
        ]

        stored_categories = fake_uow.papers.list_categories()
        assert len(stored_categories) == 2
        for category, expected_category in zip(stored_categories, expected_categories, strict=True):
            assert category.identifier == expected_category.identifier
            assert category.archive_name == expected_category.archive_name
            assert category.category_name == expected_category.category_name
            assert category.description == expected_category.description

    @pytest.mark.parametrize(
        ("fetch_error", "parse_error"),
        [
            (True, False),
            (False, True),
        ],
    )
    def test_fetch_categories_errors(self, fetch_error: bool, parse_error: bool) -> None:
        fake_extractor = FakeArXivCategoryExtractor([], fetch_error=fetch_error, parse_error=parse_error)
        fake_uow = FakeUnitOfWork()

        if fetch_error:
            with pytest.raises(CategoryFetchError):
                fetch_and_store_categories(fake_uow, fake_extractor)

        if parse_error:
            with pytest.raises(CategoryParseError):
                fetch_and_store_categories(fake_uow, fake_extractor)


class TestFetchAndStoreLatestPapers:
    def test_fetch_latest_papers_success(self) -> None:
        fake_papers = [
            PaperDTO(
                arxiv_id=f"1234.5678v{i}",
                title=f"Test Paper {i}",
                abstract="",
                published_at=datetime.date(2023, 10, 1),
                categories=categories,
            )
            for i, (categories) in enumerate([
                ["cs.AI"],
                ["cs.AI"],
                ["cs.ML"],
                ["cs.LG"],
                ["cs.CV"],
                ["cs.CV"],
            ])
        ]
        fake_extractor = FakePaperExtractor(fake_papers)
        fake_uow = FakeUnitOfWork()

        categories = [
            model.Category(model.CategoryIdentifier.from_string("cs")),
            model.Category(model.CategoryIdentifier.from_string("cs.AI")),
            model.Category(model.CategoryIdentifier.from_string("cs.ML")),
            model.Category(model.CategoryIdentifier.from_string("cs.LG")),
            model.Category(model.CategoryIdentifier.from_string("cs.CV")),
        ]
        fake_uow.papers.upsert_categories(categories)

        fetch_and_store_latest_papers(
            category_strings=["cs"],
            paper_extractor=fake_extractor,
            uow=fake_uow,
        )

        stored_papers = fake_uow.papers.list_papers(limit=None)
        assert len(stored_papers) == 6
