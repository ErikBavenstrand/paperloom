from datetime import date

from paperloom.domain import model


class TestPaper:
    def test_init(self) -> None:
        arxiv_id = "2101.00001v1"
        title = "Sample Paper"
        abstract = "This is a sample abstract."
        published_at = date(2025, 1, 1)
        categories = [model.Category(model.CategoryIdentifier("cs", "CV"))]
        paper = model.Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            published_at=published_at,
            categories=categories,
        )
        assert paper.arxiv_id == arxiv_id
        assert paper.title == title
        assert paper.abstract == abstract
        assert paper.published_at == published_at
        assert paper.categories == categories

    def test_multi_category(self) -> None:
        arxiv_id = "2101.00001v1"
        title = "Sample Paper"
        abstract = "This is a sample abstract."
        published_at = date(2025, 1, 1)
        categories = [
            model.Category(model.CategoryIdentifier("cs", "CV")),
            model.Category(model.CategoryIdentifier("cs", "CL")),
        ]
        paper = model.Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            published_at=published_at,
            categories=categories,
        )
        assert paper.arxiv_id == arxiv_id
        assert paper.title == title
        assert paper.abstract == abstract
        assert paper.published_at == published_at
        assert paper.categories == categories

    def test_no_categories(self) -> None:
        arxiv_id = "2101.00001v1"
        title = "Sample Paper"
        abstract = "This is a sample abstract."
        published_at = date(2025, 1, 1)
        paper = model.Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            published_at=published_at,
        )
        assert paper.arxiv_id == arxiv_id
        assert paper.title == title
        assert paper.abstract == abstract
        assert paper.published_at == published_at
        assert paper.categories == []

    def test_url_properties(self) -> None:
        arxiv_id = "2101.00001v1"
        paper = model.Paper(
            arxiv_id=arxiv_id,
            title="Sample Paper",
            abstract="This is a sample abstract.",
            published_at=date(2025, 1, 1),
        )

        assert paper.summary_url == f"https://arxiv.org/abs/{arxiv_id}"
        assert paper.pdf_url == f"https://arxiv.org/pdf/{arxiv_id}"
        assert paper.html_url == f"https://arxiv.org/html/{arxiv_id}"

    def test_published_at_int(self) -> None:
        arxiv_id = "2101.00001v1"
        published_at = date(2025, 1, 1)
        paper = model.Paper(
            arxiv_id=arxiv_id,
            title="Sample Paper",
            abstract="This is a sample abstract.",
            published_at=published_at,
        )
        assert paper.published_at_int == 20250101


class TestCategory:
    def test_init(self) -> None:
        category = model.Category(model.CategoryIdentifier("cs", "CV"))
        assert category.identifier.archive == "cs"
        assert category.identifier.subcategory == "CV"

    def test_init_no_subcategory(self) -> None:
        category = model.Category(model.CategoryIdentifier("cs"))
        assert category.identifier.archive == "cs"
        assert category.identifier.subcategory is None

    def test_str_init(self) -> None:
        category_identifier = model.CategoryIdentifier.from_string("cs.CV")
        assert category_identifier.archive == "cs"
        assert category_identifier.subcategory == "CV"

    def test_str_init_no_subcategory(self) -> None:
        category_identifier = model.CategoryIdentifier.from_string("cs")
        assert category_identifier.archive == "cs"
        assert category_identifier.subcategory is None

    def test_category_equality(self) -> None:
        category1 = model.Category(model.CategoryIdentifier("cs", "CV"))
        category2 = model.Category(model.CategoryIdentifier("cs", "CV"))
        assert category1 == category2

    def test_category_hash_equality(self) -> None:
        category1 = model.Category(model.CategoryIdentifier("cs", "CV"))
        category2 = model.Category(model.CategoryIdentifier("cs", "CV"))
        assert hash(category1) == hash(category2)

    def test_category_identifier(self) -> None:
        category = model.Category(model.CategoryIdentifier("cs", "CV"))
        assert str(category.identifier) == "cs.CV"

    def test_category_identifier_no_subcategory(self) -> None:
        category = model.Category(model.CategoryIdentifier("cs"))
        assert str(category.identifier) == "cs"
