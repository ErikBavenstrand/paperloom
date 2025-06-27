import datetime

import pytest
from sqlalchemy.orm import Session

from paperloom.application.ports.persistence.repository import CategoriesNotFoundError, PapersNotFoundError
from paperloom.domain import model
from paperloom.infrastructure.persistence.repository import SqlAlchemyPaperRepository


@pytest.fixture
def sample_paper() -> model.Paper:
    return model.Paper(
        arxiv_id="2025.12345",
        title="Sample Paper",
        abstract="This is a sample abstract.",
        published_at=datetime.date(2025, 1, 1),
        categories=[
            model.Category(model.CategoryIdentifier("cs", "CV")),
            model.Category(model.CategoryIdentifier("cs", "CL")),
        ],
    )


class TestSqlAlchemyPaperRepository:
    def test_upsert_and_get_paper(self, in_memory_sqlite_session: Session, sample_paper: model.Paper) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        repo.upsert_categories(sample_paper.categories)
        repo.upsert_papers([sample_paper])
        retrieved_paper = repo.get_paper(sample_paper.arxiv_id)

        assert retrieved_paper is not None
        assert retrieved_paper.arxiv_id == sample_paper.arxiv_id
        assert retrieved_paper.title == sample_paper.title
        assert retrieved_paper.abstract == sample_paper.abstract
        assert retrieved_paper.published_at == sample_paper.published_at
        assert set(retrieved_paper.categories) == set(sample_paper.categories)

        repo.delete_papers([sample_paper.arxiv_id])
        all_papers = repo.list_papers()
        assert len(all_papers) == 0

    def test_prevent_duplicate_upsert_paper(self, in_memory_sqlite_session: Session, sample_paper: model.Paper) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        repo.upsert_categories(sample_paper.categories)
        repo.upsert_papers([sample_paper, sample_paper])

        all_papers = repo.list_papers()
        assert len(all_papers) == 1
        assert all_papers[0].arxiv_id == sample_paper.arxiv_id

        repo.delete_papers([sample_paper.arxiv_id])
        papers = repo.list_papers()
        assert len(papers) == 0

    def test_delete_paper(self, in_memory_sqlite_session: Session, sample_paper: model.Paper) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        repo.upsert_categories(sample_paper.categories)
        repo.upsert_papers([sample_paper])
        repo.delete_papers([sample_paper.arxiv_id])

        all_papers = repo.list_papers()
        assert len(all_papers) == 0

    def test_list_papers(self, in_memory_sqlite_session: Session) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        sample_paper_1 = model.Paper(
            arxiv_id="2025.67890",
            title="Another Sample Paper",
            abstract="This is another sample abstract.",
            published_at=datetime.date(2025, 1, 2),
            categories=[model.Category(model.CategoryIdentifier("cs", "NLP"))],
        )
        sample_paper_2 = model.Paper(
            arxiv_id="2025.54321",
            title="Yet Another Sample Paper",
            abstract="This is yet another sample abstract.",
            published_at=datetime.date(2025, 1, 1),
            categories=[model.Category(model.CategoryIdentifier("cs", "CV"))],
        )

        repo.upsert_categories(sample_paper_1.categories + sample_paper_2.categories)
        repo.upsert_papers([sample_paper_1, sample_paper_2])

        papers = repo.list_papers()

        assert len(papers) == 2
        assert sample_paper_1.arxiv_id in [paper.arxiv_id for paper in papers]
        assert sample_paper_2.arxiv_id in [paper.arxiv_id for paper in papers]

        repo.delete_papers([sample_paper_1.arxiv_id, sample_paper_2.arxiv_id])

        all_papers = repo.list_papers()
        assert len(all_papers) == 0

    def test_upsert_update_paper(self, in_memory_sqlite_session: Session, sample_paper: model.Paper) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        repo.upsert_categories(sample_paper.categories)
        repo.upsert_papers([sample_paper])
        updated_paper = model.Paper(
            arxiv_id=sample_paper.arxiv_id,
            title="Updated Title",
            abstract=sample_paper.abstract,
            published_at=sample_paper.published_at,
            categories=sample_paper.categories,
        )
        repo.upsert_papers([updated_paper])

        retrieved_paper = repo.get_paper(sample_paper.arxiv_id)
        assert retrieved_paper is not None
        assert retrieved_paper.title == "Updated Title"

        repo.delete_papers([sample_paper.arxiv_id])
        all_papers = repo.list_papers()
        assert len(all_papers) == 0

    def test_delete_paper_not_found(self, in_memory_sqlite_session: Session) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        with pytest.raises(PapersNotFoundError):
            repo.delete_papers(["nonexistent_id"])

    def test_upsert_category(self, in_memory_sqlite_session: Session) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        categories = [
            model.Category(model.CategoryIdentifier("cs", "AI")),
            model.Category(model.CategoryIdentifier("cs", "ML")),
        ]
        for category in categories:
            repo.upsert_categories([category])

        retrieved_categories = [repo.get_category(category.identifier) for category in categories]
        assert len(retrieved_categories) == len(categories)
        for retrieved_category, category in zip(retrieved_categories, categories, strict=True):
            assert isinstance(retrieved_category, model.Category)
            assert retrieved_category == category

        repo.delete_categories([category.identifier for category in categories])

    def test_upsert_update_category(self, in_memory_sqlite_session: Session) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        category = model.Category(model.CategoryIdentifier("cs", "AI"))
        repo.upsert_categories([category])

        updated_category = model.Category(model.CategoryIdentifier("cs", "AI"), description="Updated description")
        repo.upsert_categories([updated_category])

        retrieved_category = repo.get_category(category.identifier)
        assert retrieved_category is not None
        assert retrieved_category.description == "Updated description"

        repo.delete_categories([category.identifier])

    def test_delete_category(self, in_memory_sqlite_session: Session) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        category = model.Category(model.CategoryIdentifier("cs", "AI"))
        repo.upsert_categories([category])
        repo.delete_categories([category.identifier])

        retrieved_category = repo.get_category(category.identifier)
        assert retrieved_category is None

    def test_delete_category_not_found(self, in_memory_sqlite_session: Session) -> None:
        repo = SqlAlchemyPaperRepository(in_memory_sqlite_session)

        with pytest.raises(CategoriesNotFoundError):
            repo.delete_categories([model.CategoryIdentifier("Non", "existent")])
