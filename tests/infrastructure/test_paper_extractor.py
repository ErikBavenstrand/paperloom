import datetime
import time
from unittest.mock import MagicMock, patch

import pytest

from paperloom.application.ports.paper_extractor import CategoryDTO, CategoryParseError, PaperMissingFieldError
from paperloom.domain import model
from paperloom.infrastructure.paper_extractor import ArXivCategoryExtractor, PaperDTO, RSSPaperExtractor


class TestRSSPaperExtractor:
    def test_fetch_papers_success(self) -> None:
        mock_response = {
            "entries": [
                {
                    "id": "2025.12345",
                    "title": "Sample Paper",
                    "summary": "This is a sample abstract.",
                    "published_parsed": time.struct_time((2025, 1, 1, 4, 0, 0, 1, 1, 0)),
                    "tags": [{"term": "cs.CV"}, {"term": "cs.CL"}],
                },
                {
                    "id": "2025.67890",
                    "title": "Another Sample Paper",
                    "summary": "This is another sample abstract.",
                    "published_parsed": time.struct_time((2025, 1, 2, 4, 0, 0, 2, 2, 0)),
                    "tags": [{"term": "cs.NLP"}],
                },
            ],
        }

        client = RSSPaperExtractor()
        with patch("feedparser.parse", return_value=mock_response):
            papers = client.fetch_latest([model.Category(identifier=model.CategoryIdentifier("cs", "CV"))])

        assert len(papers) == 2
        paper_1, paper_2 = sorted(papers, key=lambda paper: paper.arxiv_id)
        assert isinstance(paper_1, PaperDTO)
        assert paper_1.arxiv_id == "2025.12345"
        assert paper_1.title == "Sample Paper"
        assert paper_1.abstract == "This is a sample abstract."
        assert paper_1.published_at == datetime.date(2025, 1, 1)
        assert paper_1.categories == ["cs.CV", "cs.CL"]

        assert isinstance(paper_2, PaperDTO)
        assert paper_2.arxiv_id == "2025.67890"
        assert paper_2.title == "Another Sample Paper"
        assert paper_2.abstract == "This is another sample abstract."
        assert paper_2.published_at == datetime.date(2025, 1, 2)
        assert paper_2.categories == ["cs.NLP"]

    def test_fetch_papers_missing_fields(self) -> None:
        mock_response = {"entries": [{}]}

        client = RSSPaperExtractor()
        with (
            patch("feedparser.parse", return_value=mock_response),
            pytest.raises(PaperMissingFieldError, match="Missing required field in entry"),
        ):
            client.fetch_latest([model.Category(identifier=model.CategoryIdentifier("cs", "CV"))])

    def test_fetch_papers_empty_entries(self) -> None:
        mock_response = {}

        client = RSSPaperExtractor()
        with patch("feedparser.parse", return_value=mock_response):
            papers = client.fetch_latest([model.Category(identifier=model.CategoryIdentifier("cs", "CV"))])

        assert papers == []


class TestArXivCategoryExtractor:
    def generate_category_html(self, category_data: dict[str, list[dict[str, str]]]) -> str:
        html_content = '<html><div id="category_taxonomy_list">'
        for group_name, categories in category_data.items():
            html_content += f"<h2>{group_name}</h2>"
            for category in categories:
                category_identifier = (
                    f"{category['archive']}.{category['subcategory']}"
                    if "subcategory" in category
                    else category["archive"]
                )
                archive_html = (
                    f"<h3>{category['archive_name']} ({category['archive']})</h3>" if "archive_name" in category else ""
                )

                html_content += f"""
                <div>
                    {archive_html}
                    <h4>{category_identifier} ({category["category_name"]})</h4>
                    <p>{category["description"]}</p>
                </div>
                """
        html_content += "</div></html>"
        return html_content

    def test_fetch_categories_success(self) -> None:
        category_data: dict[str, list[dict[str, str]]] = {
            "Computer Science": [
                {
                    "archive": "cs",
                    "subcategory": "AI",
                    "category_name": "Artificial Intelligence",
                    "description": "Research in artificial intelligence.",
                },
                {
                    "archive": "cs",
                    "subcategory": "CV",
                    "category_name": "Computer Vision",
                    "description": "Research in computer vision.",
                },
            ],
            "Physics": [
                {
                    "archive": "astro-ph",
                    "subcategory": "GA",
                    "archive_name": "Astrophysics",
                    "category_name": "Astrophysics of Galaxies",
                    "description": "Research in the astrophysics of galaxies.",
                },
                {
                    "archive": "nucl-th",
                    "archive_name": "Nuclear Theory",
                    "category_name": "Nuclear Theory",
                    "description": "Research in nuclear theory.",
                },
            ],
        }

        html_content = self.generate_category_html(category_data)
        mock_response = MagicMock(status_code=200, text=html_content)

        client = ArXivCategoryExtractor()
        with patch("requests.get", return_value=mock_response):
            categories = client.fetch_categories()

        expected_categories = [
            CategoryDTO(
                archive=cat["archive"],
                subcategory=cat.get("subcategory"),
                archive_name=cat.get("archive_name", group_name),
                category_name=cat.get("category_name"),
                description=cat.get("description"),
            )
            for group_name, cats in category_data.items()
            for cat in cats
        ] + list({
            CategoryDTO(
                archive=cat["archive"],
                subcategory=None,
                archive_name=cat.get("archive_name", group_name),
                category_name=None,
                description=None,
            )
            for group_name, cats in category_data.items()
            for cat in cats
            if cat.get("subcategory") is not None
        })

        assert categories == expected_categories

    def test_fetch_categories_empty_response(self) -> None:
        mock_response = MagicMock(status_code=200, text="<html></html>")

        client = ArXivCategoryExtractor()
        with (
            patch("requests.get", return_value=mock_response),
            pytest.raises(CategoryParseError, match="Failed to find the category taxonomy"),
        ):
            client.fetch_categories()

    def test_fetch_categories_no_categories(self) -> None:
        mock_response = MagicMock(status_code=200, text="<html><div id='category_taxonomy_list'></div></html>")

        client = ArXivCategoryExtractor()
        with patch("requests.get", return_value=mock_response):
            categories = client.fetch_categories()

        assert categories == []
