import datetime
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

from paperloom.application.ports.paper_extractor import (
    CategoryDTO,
    CategoryFetchError,
    CategoryParseError,
    PaperMissingFieldError,
)
from paperloom.domain import model
from paperloom.infrastructure.paper_extractor import (
    ArXivCategoryExtractor,
    JSONPaperExtractor,
    PaperDTO,
    RSSPaperExtractor,
)


# ---- Fixtures ----
@pytest.fixture
def categories_filter() -> set[model.Category]:
    archive = "cs"
    subcategories = {model.Category(identifier=model.CategoryIdentifier(archive, sub)) for sub in ["AI", "CV", "NLP"]}
    return {
        model.Category(
            identifier=model.CategoryIdentifier(archive, None),
            subcategories=subcategories,
        )
    }


@pytest.fixture
def json_lines_file(tmp_path: Path) -> Callable[[list[dict[str, Any]]], Path]:
    def _create(lines: list[dict[str, Any]]) -> Path:
        file_path = tmp_path / "papers.jsonl"
        with file_path.open("w", encoding="utf-8") as f:
            for entry in lines:
                f.write(json.dumps(entry) + "\n")
        return file_path

    return _create


# ---- Helper Functions ----
def make_rss_paper_entry(
    arxiv_id: str,
    title: str,
    summary: str,
    published: tuple[int, int, int],
    tags: list[str] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": arxiv_id,
        "title": title,
        "summary": summary,
        "published_parsed": time.struct_time((published[0], published[1], published[2], 0, 0, 0, 0, 0, 0)),
    }
    if tags is not None:
        entry["tags"] = [{"term": tag} for tag in tags]
    return entry


def generate_category_html(category_data: dict[str, list[dict[str, str]]]) -> str:
    html_content = ["<html>", '<div id="category_taxonomy_list">']
    for group_name, categories in category_data.items():
        html_content.append(f"<h2>{group_name}</h2>")
        for category in categories:
            archive = category["archive"]
            sub = category.get("subcategory")
            identifier = f"{archive}.{sub}" if sub else archive
            archive_name = category.get("archive_name")
            html_content.append("<div>")
            if archive_name:
                html_content.append(f"<h3>{archive_name} ({archive})</h3>")
            html_content.extend((
                f"<h4>{identifier} ({category['category_name']})</h4>",
                f"<p>{category.get('description', '')}</p>",
                "</div>",
            ))
    html_content.extend(("</div>", "</html>"))
    return "\n".join(html_content)


class TestRSSPaperExtractor:
    @pytest.mark.parametrize(
        ("entries", "expected_dtos", "limit"),
        [
            (
                [
                    make_rss_paper_entry("2025.12345", "Sample Paper", "Abstract 1", (2025, 1, 1), ["cs.CV", "cs.CL"]),
                    make_rss_paper_entry("2025.67890", "Another Paper", "Abstract 2", (2025, 1, 2), ["cs.NLP"]),
                ],
                [
                    PaperDTO(
                        arxiv_id="2025.12345",
                        title="Sample Paper",
                        abstract="Abstract 1",
                        published_at=datetime.date(2025, 1, 1),
                        categories={"cs.CV", "cs.CL"},
                    ),
                    PaperDTO(
                        arxiv_id="2025.67890",
                        title="Another Paper",
                        abstract="Abstract 2",
                        published_at=datetime.date(2025, 1, 2),
                        categories={"cs.NLP"},
                    ),
                ],
                2,
            ),
            (
                [],
                [],
                5,
            ),
        ],
        ids=["success_two_papers", "success_no_papers"],
    )
    def test_rss_fetch_latest(
        self,
        categories_filter: set[model.Category],
        entries: list[dict[str, Any]],
        expected_dtos: list[PaperDTO],
        limit: int,
    ) -> None:
        extractor = RSSPaperExtractor()
        extractor.RSS_FEED_LIMIT = limit  # type: ignore[assignment]
        mock_response = {"entries": entries}

        with patch("feedparser.parse", return_value=mock_response):
            result = extractor.fetch_latest(categories_filter)

        assert isinstance(result, set)
        assert result == set(expected_dtos)

    @pytest.mark.parametrize(
        ("response", "expected_exception", "match_message"),
        [
            ({"entries": [{}]}, PaperMissingFieldError, "Missing required field in entry"),
            (None, None, None),
        ],
        ids=["missing_fields", "no_response"],
    )
    def test_rss_fetch_latest_error_and_empty(
        self,
        categories_filter: set[model.Category],
        response: Any,
        expected_exception: type[Exception] | None,
        match_message: str | None,
    ) -> None:
        extractor = RSSPaperExtractor()
        mock_return = response if isinstance(response, dict) else {}

        with patch("feedparser.parse", return_value=mock_return):
            if expected_exception:
                with pytest.raises(expected_exception, match=match_message):
                    extractor.fetch_latest(categories_filter)
            else:
                result = extractor.fetch_latest(categories_filter)
                assert result == set()


class TestArXivCategoryExtractor:
    @pytest.mark.parametrize(
        ("category_data", "expected_dtos"),
        [
            (
                {
                    "Computer Science": [
                        {
                            "archive": "cs",
                            "subcategory": "AI",
                            "category_name": "Artificial Intelligence",
                            "description": "AI research.",
                        },
                        {
                            "archive": "cs",
                            "subcategory": "CV",
                            "category_name": "Computer Vision",
                            "description": "CV research.",
                        },
                    ],
                    "Physics": [
                        {
                            "archive": "astro-ph",
                            "subcategory": "GA",
                            "archive_name": "Astrophysics",
                            "category_name": "Astrophysics of Galaxies",
                            "description": "Galaxies research.",
                        },
                        {"archive": "nucl-th", "archive_name": "Nuclear Theory", "category_name": "Nuclear Theory"},
                    ],
                },
                {
                    CategoryDTO("cs", None, "Computer Science", None, None),
                    CategoryDTO("cs", "AI", "Computer Science", "Artificial Intelligence", "AI research."),
                    CategoryDTO("cs", "CV", "Computer Science", "Computer Vision", "CV research."),
                    CategoryDTO("astro-ph", None, "Astrophysics", None, None),
                    CategoryDTO("astro-ph", "GA", "Astrophysics", "Astrophysics of Galaxies", "Galaxies research."),
                    CategoryDTO("nucl-th", None, "Nuclear Theory", "Nuclear Theory", ""),
                },
            ),
        ],
        ids=["mixed_groups"],
    )
    def test_arxiv_fetch_categories_success(
        self, category_data: dict[str, list[dict[str, str]]], expected_dtos: set[CategoryDTO]
    ) -> None:
        extractor = ArXivCategoryExtractor()
        html = generate_category_html(category_data)
        mock_resp = MagicMock(status_code=200, text=html)

        with patch("requests.get", return_value=mock_resp):
            result = extractor.fetch_categories()

        assert isinstance(result, set)
        assert result == expected_dtos

    @pytest.mark.parametrize(
        ("html", "expected_exception", "expected_result", "match_message"),
        [
            ("<html></html>", CategoryParseError, None, "Failed to find the category taxonomy"),
            ("<html><div id='category_taxonomy_list'></div></html>", None, set(), None),
            (
                (
                    "<html><div id='category_taxonomy_list'><h2>Computer Science</h2>"
                    "<div><h4>MISSING_HEADER</h4><p>AI research.</p></div></div></html>"
                ),
                CategoryParseError,
                None,
                "Failed to parse category header",
            ),
            (
                (
                    "<html><div id='category_taxonomy_list'><h2>Computer Science</h2>"
                    "<div><p>AI research.</p></div></div></html>"
                ),
                CategoryParseError,
                None,
                "Missing archive for category None in group",
            ),
            (
                (
                    "<html><div id='category_taxonomy_list'><h2>Computer Science</h2>"
                    "<div><h4></h4><p>AI research.</p></div></div></html>"
                ),
                CategoryParseError,
                None,
                "Failed to parse category header",
            ),
            (
                (
                    "<html><div id='category_taxonomy_list'><h2>Computer Science</h2>"
                    "<div><h3></h3><p>AI research.</p></div></div></html>"
                ),
                CategoryParseError,
                None,
                "Missing archive for category None in group",
            ),
        ],
        ids=[
            "no_taxonomy",
            "empty_taxonomy",
            "faulty_category_header",
            "missing_category_header",
            "empty_category_header",
            "empty_archive_header",
        ],
    )
    def test_arxiv_fetch_categories_empty_and_no_categories(
        self,
        html: str,
        expected_exception: type[Exception] | None,
        expected_result: set[CategoryDTO] | None,
        match_message: str | None,
    ) -> None:
        extractor = ArXivCategoryExtractor()
        mock_resp = MagicMock(status_code=200, text=html)

        with patch("requests.get", return_value=mock_resp):
            if expected_exception:
                with pytest.raises(expected_exception, match=match_message):
                    extractor.fetch_categories()
            else:
                result = extractor.fetch_categories()
                assert result == expected_result

    def test_arxiv_fetch_categories_fetch_error(self) -> None:
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
            mock_get.return_value = mock_resp

            extractor = ArXivCategoryExtractor()
            with pytest.raises(CategoryFetchError, match="Failed to fetch the categories"):
                extractor.fetch_categories()


class TestJSONPaperExtractor:
    @pytest.mark.parametrize(
        ("entries", "from_date", "to_date", "expected_ids"),
        [
            (
                [
                    {"id": "a1", "title": "T1", "abstract": "A1", "update_date": "2025-01-01", "categories": "cs.AI"},
                    {
                        "id": "b2",
                        "title": "T2",
                        "abstract": "A2",
                        "update_date": "2025-02-01",
                        "categories": "math",
                    },
                    {
                        "id": "c3",
                        "title": "T3",
                        "abstract": "A3",
                        "update_date": "2025-03-01",
                        "categories": "cs.AI math",
                    },
                ],
                None,
                None,
                {"a1", "c3"},
            ),
            (
                [
                    {"id": "x1", "title": "X", "abstract": "AX", "update_date": "2025-01-10", "categories": "cs"},
                    {"id": "x2", "title": "X", "abstract": "AX", "update_date": "2025-02-10", "categories": "cs"},
                ],
                datetime.date(2025, 2, 1),
                None,
                {"x2"},
            ),
            (
                [
                    {"id": "y1", "title": "Y", "abstract": "AY", "update_date": "2025-01-05", "categories": "cs"},
                    {"id": "y2", "title": "Y", "abstract": "AY", "update_date": "2025-03-05", "categories": "cs"},
                ],
                None,
                datetime.date(2025, 2, 1),
                {"y1"},
            ),
        ],
        ids=["no_date_filter", "from_date_only", "to_date_only"],
    )
    def test_fetch_historical_filters(
        self,
        json_lines_file: Callable[[list[dict[str, Any]]], Path],
        categories_filter: set[model.Category],
        entries: list[dict[str, Any]],
        from_date: datetime.date | None,
        to_date: datetime.date | None,
        expected_ids: set[str],
    ) -> None:
        file_path = json_lines_file(entries)
        extractor = JSONPaperExtractor(file_path)
        result = extractor.fetch_historical(categories_filter, from_date, to_date)
        assert {dto.arxiv_id for dto in result} == expected_ids

    def test_fetch_historical_missing_field_raises(
        self, tmp_path: Path, categories_filter: set[model.Category]
    ) -> None:
        file_path = tmp_path / "bad.json"
        with file_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "z1", "title": "Z", "abstract": "AZ", "update_date": "2025-01-01"}) + "\n")
        extractor = JSONPaperExtractor(file_path)
        with pytest.raises(ValidationError):
            extractor.fetch_historical(categories_filter, None, None)
