from sqlalchemy.orm import Session

from paperloom.application.ports.persistence.repository import (
    AbstractPaperRepository,
    CategoriesNotFoundError,
    PapersNotFoundError,
)
from paperloom.domain import model
from paperloom.infrastructure.persistence.orm import CategoryORM, PaperORM


class SqlAlchemyPaperRepository(AbstractPaperRepository):
    """A `Paper` domain object repository using `SQLAlchemy`."""

    def __init__(self, session: Session) -> None:
        """Initializes the `SqlAlchemyPaperRepository`.

        Args:
            session: The `SQLAlchemy` session.
        """
        self.session = session

    def upsert_categories(self, categories: list[model.Category]) -> None:
        """Upserts a list of `Category` domain objects into the database.

        Args:
            categories: A list of `Category` domain objects to upsert.
        """
        existing_category_orms = (
            self.session.query(CategoryORM)
            .filter(CategoryORM.identifier.in_({str(category.identifier) for category in categories}))
            .all()
        )

        existing_category_map = {category_orm.identifier: category_orm for category_orm in existing_category_orms}

        for category in categories:
            category_orm = existing_category_map.get(str(category.identifier))
            if category_orm:
                category_orm.archive_name = category.archive_name
                category_orm.category_name = category.category_name
                category_orm.description = category.description
            else:
                self.session.add(self._to_category_orm(category))

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

    def delete_categories(self, category_identifiers: list[model.CategoryIdentifier]) -> None:
        """Deletes the specified `Category` domain objects from the database.

        Args:
            category_identifiers: A list of `CategoryIdentifier` domain objects representing the categories to delete.

        Raises:
            CategoriesNotFoundError: If any of the categories are not found in the database.
        """
        category_orms = (
            self.session.query(CategoryORM)
            .filter(
                CategoryORM.identifier.in_({str(category_identifier) for category_identifier in category_identifiers}),
            )
            .all()
        )

        missing_categories = list(
            set(category_identifiers)
            - {model.CategoryIdentifier.from_string(category_orm.identifier) for category_orm in category_orms},
        )
        if missing_categories:
            raise CategoriesNotFoundError(missing_categories)

        for category_orm in category_orms:
            self.session.delete(category_orm)

        self.session.flush()

    def list_categories(self) -> list[model.Category]:
        """Lists all `Category` domain objects in the database.

        Returns:
            A list of `Category` domain objects.
        """
        categories_orm = self.session.query(CategoryORM).order_by(CategoryORM.id).all()
        return [self._to_category(category_orm) for category_orm in categories_orm]

    def upsert_papers(self, papers: list[model.Paper]) -> None:
        """Upserts a list of `Paper` domain objects into the database.

        Args:
            papers: A list of `Paper` domain objects to upsert.

        Raises:
            CategoriesNotFoundError: If any of the categories are not found in the database.
        """
        categories = {category for paper in papers for category in paper.categories}
        category_orms = (
            self.session.query(CategoryORM)
            .filter(CategoryORM.identifier.in_({str(category.identifier) for category in categories}))
            .all()
        )
        missing_categories = list(
            {category.identifier for category in categories}
            - {model.CategoryIdentifier.from_string(category_orm.identifier) for category_orm in category_orms},
        )
        if missing_categories:
            raise CategoriesNotFoundError(missing_categories)

        category_orm_map = {category_orm.identifier: category_orm for category_orm in category_orms}

        existing_paper_orms = (
            self.session.query(PaperORM).filter(PaperORM.arxiv_id.in_({paper.arxiv_id for paper in papers})).all()
        )
        existing_paper_map = {paper_orm.arxiv_id: paper_orm for paper_orm in existing_paper_orms}

        for paper in set(papers):
            paper_orm = existing_paper_map.get(paper.arxiv_id)
            category_orms = [category_orm_map[str(category.identifier)] for category in paper.categories]

            if paper_orm:
                paper_orm.title = paper.title
                paper_orm.abstract = paper.abstract
                paper_orm.published_at = paper.published_at
                paper_orm.categories = category_orms
            else:
                paper_orm = self._to_paper_orm(paper, category_orms)
                self.session.add(paper_orm)

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

    def delete_papers(self, arxiv_ids: list[str]) -> None:
        """Deletes the specified `Paper` domain objects from the database.

        Args:
            arxiv_ids: A list of ArXiv IDs representing the papers to delete.

        Raises:
            PapersNotFoundError: If any of the papers are not found in the database.
        """
        paper_orms = self.session.query(PaperORM).filter(PaperORM.arxiv_id.in_(set(arxiv_ids))).all()

        missing_papers = list(set(arxiv_ids) - {paper_orm.arxiv_id for paper_orm in paper_orms})
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
    def _to_category_orm(category: model.Category) -> CategoryORM:
        """Converts a `Category` domain object to a `CategoryORM` ORM object.

        Args:
            category: The `Category` domain object to convert.

        Returns:
            The converted `CategoryORM` ORM.
        """
        return CategoryORM(
            archive=category.identifier.archive,
            subcategory=category.identifier.subcategory,
            archive_name=category.archive_name,
            category_name=category.category_name,
            description=category.description,
        )

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
            subcategories=[SqlAlchemyPaperRepository._to_category(sub_orm) for sub_orm in category_orm.subcategories],
        )

    @staticmethod
    def _to_paper_orm(paper: model.Paper, category_orms: list[CategoryORM]) -> PaperORM:
        """Converts a `Paper` domain object to a `PaperORM` ORM object.

        Args:
            paper: The `Paper` domain object to convert.
            category_orms: The list of `CategoryORM` ORM objects associated with the paper.

        Returns:
            The converted `PaperORM` ORM.
        """
        return PaperORM(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
            published_at=paper.published_at,
            categories=category_orms,
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
            categories=[SqlAlchemyPaperRepository._to_category(category_orm) for category_orm in paper_orm.categories],
        )
