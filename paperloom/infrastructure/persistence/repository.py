from collections.abc import Iterator
from itertools import islice

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session
from tqdm.auto import tqdm

from paperloom.application.ports.persistence.repository import (
    AbstractPaperRepository,
    CategoriesNotFoundError,
    PapersNotFoundError,
)
from paperloom.config import DATABASE_CHUNK_SIZE
from paperloom.domain import model
from paperloom.infrastructure.persistence.orm import CategoryORM, PaperORM, paper_category


class SqlAlchemyPaperRepository(AbstractPaperRepository):
    """A `Paper` domain object repository using `SQLAlchemy`."""

    def __init__(self, session: Session) -> None:
        """Initializes the `SqlAlchemyPaperRepository`.

        Args:
            session: The `SQLAlchemy` session.
        """
        self.session = session

    def upsert_categories(self, categories: set[model.Category]) -> None:
        """Upserts a set of `Category` domain objects into the database.

        Args:
            categories: A set of `Category` domain objects to upsert.
        """
        for chunk in tqdm(
            self._chunk_set(categories, DATABASE_CHUNK_SIZE),
            desc="Upserting categories into the database",
            unit="chunk",
            total=len(categories) // DATABASE_CHUNK_SIZE + 1,
        ):
            # Prepare chunk data
            values = [
                {
                    CategoryORM.archive.key: category.identifier.archive,
                    CategoryORM.subcategory.key: category.identifier.subcategory,
                    CategoryORM.archive_name.key: category.archive_name,
                    CategoryORM.category_name.key: category.category_name,
                    CategoryORM.description.key: category.description,
                }
                for category in chunk
            ]

            # Execute upsert
            stmt = insert(CategoryORM).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[CategoryORM.archive, CategoryORM.subcategory],
                set_={
                    CategoryORM.archive_name.key: stmt.excluded.archive_name,
                    CategoryORM.category_name.key: stmt.excluded.category_name,
                    CategoryORM.description.key: stmt.excluded.description,
                },
            )
            self.session.execute(stmt)
            self.session.flush()

    def get_category(self, category_identifier: model.CategoryIdentifier) -> model.Category | None:
        """Fetches a `Category` domain object from the database.

        Args:
            category_identifier: The `CategoryIdentifier` domain object.

        Returns:
            The `Category` domain object if found, otherwise `None`.
        """
        category_orm = self.session.query(CategoryORM).filter_by(identifier=str(category_identifier)).first()

        return self._to_category(category_orm) if category_orm else None

    def delete_categories(self, category_identifiers: set[model.CategoryIdentifier]) -> None:
        """Deletes the specified `Category` domain objects from the database.

        Args:
            category_identifiers: A set of `CategoryIdentifier` domain objects representing the categories to delete.

        Raises:
            CategoriesNotFoundError: If any of the categories are not found in the database.
        """
        for chunk in self._chunk_set(category_identifiers, DATABASE_CHUNK_SIZE):
            self._delete_categories_chunk(chunk)
            self.session.flush()

    def _delete_categories_chunk(self, category_identifiers: set[model.CategoryIdentifier]) -> None:
        """Deletes a chunk of `Category` domain objects from the database.

        Args:
            category_identifiers: A set of `CategoryIdentifier` domain objects representing the categories to delete.

        Raises:
            CategoriesNotFoundError: If any of the categories are not found in the database.
        """
        existing_category_identifiers = (
            self.session.execute(
                select(CategoryORM.identifier).where(
                    CategoryORM.identifier.in_({str(category) for category in category_identifiers}),
                ),
            )
            .scalars()
            .all()
        )

        missing_categories = category_identifiers - {
            model.CategoryIdentifier.from_string(existing_category_identifier)
            for existing_category_identifier in existing_category_identifiers
        }
        if missing_categories:
            raise CategoriesNotFoundError(missing_categories)

        self.session.execute(
            delete(CategoryORM)
            .where(CategoryORM.identifier.in_({str(category) for category in category_identifiers}))
            .execution_options(synchronize_session=False),
        )

    def list_categories(self) -> list[model.Category]:
        """Lists all `Category` domain objects in the database.

        Returns:
            A list of `Category` domain objects.
        """
        categories_orm = self.session.query(CategoryORM).order_by(CategoryORM.id).all()
        return [self._to_category(category_orm) for category_orm in categories_orm]

    def upsert_papers(self, papers: set[model.Paper]) -> None:
        """Upserts a set of `Paper` domain objects into the database.

        Args:
            papers: A set of `Paper` domain objects to upsert.

        Raises:
            CategoriesNotFoundError: If any of the categories are not found in the database.
        """
        for chunk in tqdm(
            self._chunk_set(papers, DATABASE_CHUNK_SIZE),
            desc="Upserting papers into the database",
            unit="chunk",
            total=len(papers) // DATABASE_CHUNK_SIZE + 1,
        ):
            # Prepare chunk data
            values = [
                {
                    PaperORM.arxiv_id.key: paper.arxiv_id,
                    PaperORM.title.key: paper.title,
                    PaperORM.abstract.key: paper.abstract,
                    PaperORM.published_at.key: paper.published_at,
                }
                for paper in chunk
            ]

            # Execute upsert of papers
            stmt = insert(PaperORM).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[PaperORM.arxiv_id],
                set_={
                    PaperORM.title.key: stmt.excluded.title,
                    PaperORM.abstract.key: stmt.excluded.abstract,
                    PaperORM.published_at.key: stmt.excluded.published_at,
                },
            )
            self.session.execute(stmt)
            self.session.flush()

            # Get `PaperORM` IDs for the upserted papers from the session
            arxiv_ids = {paper.arxiv_id for paper in chunk}
            paper_id_map: dict[str, int] = dict(
                self.session.execute(  # type: ignore[arg-type]
                    select(PaperORM.arxiv_id, PaperORM.id).where(PaperORM.arxiv_id.in_(arxiv_ids)),
                ).all()
            )

            # Get `CategoryORM` objects for the categories of the upserted papers
            paper_category_identifiers = {str(category.identifier) for paper in chunk for category in paper.categories}
            category_id_map: dict[str, int] = dict(
                self.session.execute(  # type: ignore[arg-type]
                    select(CategoryORM.identifier, CategoryORM.id).where(
                        CategoryORM.identifier.in_(paper_category_identifiers)
                    )
                ).all()
            )

            # Handle categories for the upserted papers
            category_refs = []
            for paper in chunk:
                paper_id = paper_id_map[paper.arxiv_id]
                for category in paper.categories:
                    category_id = category_id_map[str(category.identifier)]
                    category_refs.append({
                        "paper_id": paper_id,
                        "category_id": category_id,
                    })

            if category_refs:
                # Delete existing paper-category associations for the upserted papers
                self.session.execute(delete(paper_category).where(paper_category.c.paper_id.in_(paper_id_map.values())))

                # Insert new paper-category associations
                self.session.execute(insert(paper_category).values(category_refs))

            self.session.flush()

    def get_paper(self, arxiv_id: str) -> model.Paper | None:
        """Fetches a `Paper` domain object from the database.

        Args:
            arxiv_id: The ArXiv ID of the paper.

        Returns:
            The `Paper` domain object if found, otherwise `None`.
        """
        paper_orm = self.session.query(PaperORM).filter_by(arxiv_id=arxiv_id).first()
        return self._to_paper(paper_orm) if paper_orm else None

    def delete_papers(self, arxiv_ids: set[str]) -> None:
        """Deletes the specified `Paper` domain objects from the database.

        Args:
            arxiv_ids: A set of ArXiv IDs representing the papers to delete.

        Raises:
            PapersNotFoundError: If any of the papers are not found in the database.
        """
        paper_orms = self.session.query(PaperORM).filter(PaperORM.arxiv_id.in_(arxiv_ids)).all()

        missing_papers = arxiv_ids - {paper_orm.arxiv_id for paper_orm in paper_orms}
        if missing_papers:
            raise PapersNotFoundError(missing_papers)

        for paper_orm in paper_orms:
            self.session.delete(paper_orm)

        self.session.flush()

    def list_papers(self, *, limit: int | None = 50) -> list[model.Paper]:
        """Lists all `Paper` domain objects in the database.

        Args:
            limit: The maximum number of papers to return.

        Returns:
            A list of `Paper` domain objects.
        """
        papers_orm = self.session.query(PaperORM).order_by(PaperORM.id).limit(limit).all()
        return [self._to_paper(paper_orm) for paper_orm in papers_orm]

    @staticmethod
    def _chunk_set[T](items: set[T], chunk_size: int) -> Iterator[set[T]]:
        """Splits a set into chunks of a specified size.

        Args:
            items: The set to split.
            chunk_size: The size of each chunk.

        Yields:
            A set of sets, where each inner set is a chunk of the original set.
        """
        items_iter = iter(items)
        while True:
            chunk = set(islice(items_iter, chunk_size))
            if not chunk:
                break
            yield chunk

    @staticmethod
    def _to_category(category_orm: CategoryORM) -> model.Category:
        """Converts a `CategoryORM` ORM object to a `Category` domain object.

        Args:
            category_orm: The `CategoryORM` ORM object to convert.

        Returns:
            The converted `Category` domain object.
        """
        return model.Category(
            identifier=model.CategoryIdentifier(
                archive=category_orm.archive,
                subcategory=category_orm.subcategory,
            ),
            archive_name=category_orm.archive_name,
            category_name=category_orm.category_name,
            description=category_orm.description,
            subcategories={SqlAlchemyPaperRepository._to_category(sub_orm) for sub_orm in category_orm.subcategories},
        )

    @staticmethod
    def _to_paper(paper_orm: PaperORM) -> model.Paper:
        """Converts a `PaperORM` ORM object to a `Paper` domain object.

        Args:
            paper_orm: The `PaperORM` ORM object to convert.

        Returns:
            The converted `Paper` domain object.
        """
        return model.Paper(
            arxiv_id=paper_orm.arxiv_id,
            title=paper_orm.title,
            abstract=paper_orm.abstract,
            published_at=paper_orm.published_at,
            categories={SqlAlchemyPaperRepository._to_category(category_orm) for category_orm in paper_orm.categories},
        )
