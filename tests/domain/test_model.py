import copy
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

from paperloom.domain.model import (
    Category,
    CategoryIdentifier,
    InvalidCategoryError,
    Paper,
)


# ---- Fixtures ----
@pytest.fixture
def base_paper_args() -> dict[str, Any]:
    return {
        "arxiv_id": "2101.00001v1",
        "title": "Sample Paper",
        "abstract": "This is a sample abstract.",
        "published_at": date(2025, 1, 1),
        "categories": None,
    }


@pytest.fixture
def paper_factory(base_paper_args: dict[str, Any]) -> Callable[..., Paper]:
    def _create(**overrides: Any) -> Paper:
        args = copy.deepcopy(base_paper_args)
        args.update(overrides)

        if args.get("categories") is None:
            args.pop("categories")
        return Paper(**args)

    return _create


# ---- Tests ----
class TestPaper:
    @pytest.mark.parametrize(
        ("input_categories", "expected_categories"),
        [
            (None, set()),
            ({Category(CategoryIdentifier("cs", "CV"))}, {Category(CategoryIdentifier("cs", "CV"))}),
            (
                {Category(CategoryIdentifier("cs", "CV")), Category(CategoryIdentifier("cs", "CL"))},
                {Category(CategoryIdentifier("cs", "CV")), Category(CategoryIdentifier("cs", "CL"))},
            ),
        ],
        ids=["no_categories", "single_category", "multi_category"],
    )
    def test_categories(
        self,
        paper_factory: Callable[..., Paper],
        input_categories: set[Category] | None,
        expected_categories: set[Category],
    ) -> None:
        paper = paper_factory(categories=input_categories)
        assert paper.categories == expected_categories

    @pytest.mark.parametrize(
        ("property_name", "expected_value"),
        [
            ("summary_url", "https://arxiv.org/abs/2101.00001v1"),
            ("pdf_url", "https://arxiv.org/pdf/2101.00001v1"),
            ("html_url", "https://arxiv.org/html/2101.00001v1"),
            ("published_at_int", 20250101),
        ],
        ids=["summary_url", "pdf_url", "html_url", "published_at_int"],
    )
    def test_url_and_date_properties(
        self, paper_factory: Callable[..., Paper], property_name: str, expected_value: Any
    ) -> None:
        paper = paper_factory()
        assert getattr(paper, property_name) == expected_value

    def test_repr_contains_key_fields(self, paper_factory: Callable[..., Paper]) -> None:
        paper = paper_factory()
        repr_str = repr(paper)
        # Check that repr contains arxiv_id, title, published_at, and categories
        assert "arxiv_id='2101.00001v1'" in repr_str
        assert "title='Sample Paper'" in repr_str
        assert "published_at=datetime.date(2025, 1, 1)" in repr_str
        assert "categories=" in repr_str

    @pytest.mark.parametrize(
        ("args1", "args2", "should_equal"),
        [
            (("2101.00001v1", "Sample Paper"), ("2101.00001v1", "Sample Paper"), True),
            (("2101.00001v1", "Sample Paper"), ("2101.00002v1", "Sample Paper"), False),
            (("2101.00001v1", "Sample Paper"), ("2101.00001v2", "Sample Paper"), False),
            (("2101.00001v1", "Sample Paper"), ("2101.00001v1", "Another Title"), True),
        ],
        ids=["same", "diff_id", "diff_version", "diff_title"],
    )
    def test_equality_and_hash(self, args1: tuple[str, str], args2: tuple[str, str], should_equal: bool) -> None:
        p1 = Paper(arxiv_id=args1[0], title=args1[1], abstract="", published_at=date(2025, 1, 1))
        p2 = Paper(arxiv_id=args2[0], title=args2[1], abstract="", published_at=date(2025, 1, 1))

        assert (p1 == p2) is should_equal
        assert (hash(p1) == hash(p2)) is should_equal
        # Non-Paper comparisons
        assert (p1 == object()) is False
        assert (p1.arxiv_id == object()) is False


# CategoryIdentifier and Category tests
class TestCategory:
    @pytest.mark.parametrize(
        ("input_str", "exp_archive", "exp_subcategory", "exp_exception"),
        [
            ("cs.CV", "cs", "CV", None),
            ("cs", "cs", None, None),
            ("cs.AI.LLM", None, None, InvalidCategoryError),
        ],
        ids=["valid_sub", "valid_no_sub", "invalid_format"],
    )
    def test_from_string(
        self, input_str: str, exp_archive: str | None, exp_subcategory: str | None, exp_exception: type | None
    ) -> None:
        if exp_exception:
            with pytest.raises(exp_exception):
                CategoryIdentifier.from_string(input_str)
        else:
            ident = CategoryIdentifier.from_string(input_str)
            assert ident.archive == exp_archive
            assert ident.subcategory == exp_subcategory

    @pytest.mark.parametrize(
        ("a1", "s1", "a2", "s2", "equal"),
        [
            ("cs", "CV", "cs", "CV", True),
            ("cs", "CV", "cs", "CL", False),
            ("cs", None, "cs", None, True),
            ("cs", "CV", "math", "CV", False),
        ],
        ids=["eq_same", "neq_sub", "eq_none_sub", "neq_archive"],
    )
    def test_category_equality_and_hash(self, a1: str, s1: str | None, a2: str, s2: str | None, equal: bool) -> None:
        ci1 = CategoryIdentifier(a1) if s1 is None else CategoryIdentifier(a1, s1)
        ci2 = CategoryIdentifier(a2) if s2 is None else CategoryIdentifier(a2, s2)
        c1 = Category(ci1)
        c2 = Category(ci2)

        assert (ci1 == ci2) is equal
        assert (hash(ci1) == hash(ci2)) is equal
        assert (c1 == c2) is equal
        assert (hash(c1) == hash(c2)) is equal
        # Non-comparable
        assert (ci1 == object()) is False
        assert (c1 == 123) is False

    @pytest.mark.parametrize(
        ("category", "expected_repr"),
        [
            (
                Category(
                    CategoryIdentifier("cs", "CV"),
                    archive_name="Computer Science",
                    category_name="Computer Vision",
                    description="Computer Vision category",
                ),
                (
                    "Category(identifier='cs.CV', archive_name='Computer Science', "
                    "category_name='Computer Vision', description='Computer Vision category')"
                ),
            ),
            (
                Category(CategoryIdentifier("cs"), archive_name="Computer Science"),
                "Category(identifier='cs', archive_name='Computer Science', category_name=None, description=None)",
            ),
        ],
        ids=["with_subcategory", "without_subcategory"],
    )
    def test_category_repr(
        self,
        category: Category,
        expected_repr: str,
    ) -> None:
        assert repr(category) == expected_repr
