import datetime
import json
import re
import time
from pathlib import Path
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, ConfigDict, ValidationError
from tqdm.auto import tqdm

from paperloom.application.ports.paper_extractor import (
    AbstractCategoryExtractor,
    AbstractPaperExtractor,
    CategoryDTO,
    CategoryFetchError,
    CategoryParseError,
    PaperDTO,
    PaperMissingFieldError,
)
from paperloom.domain import model


class RSSPaperEntry(BaseModel):
    """A model representing a single RSS paper entry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    """The unique identifier for the paper, typically in the format 'arxiv:xxxx.xxxx'."""
    title: str
    """The title of the paper."""
    summary: str
    """The summary or abstract of the paper."""
    published_parsed: time.struct_time
    """The published date of the paper as a struct_time object."""
    tags: list[dict[str, str | None]] | None = None
    """Optional tags associated with the paper, each represented as a dictionary with 'term' key."""

    def to_paper_dto(self) -> PaperDTO:
        """Converts the RSS paper entry to a `PaperDTO`.

        Returns:
            A `PaperDTO` object containing the paper's details.
        """
        arxiv_id = self.id.split(":")[-1].split("v")[0].strip()
        abstract = self.summary.split("Abstract:")[-1].strip().replace("\n", " ")
        published_at = datetime.datetime.fromtimestamp(time.mktime(self.published_parsed)).date()
        categories = {tag["term"] for tag in self.tags or [] if "term" in tag and tag["term"] is not None}

        return PaperDTO(
            arxiv_id=arxiv_id,
            title=self.title.strip(),
            abstract=abstract,
            published_at=published_at,
            categories=categories,
        )


class RSSPaperExtractor(AbstractPaperExtractor):
    """A fetcher that extracts papers from the ArXiv RSS feed."""

    RSS_FEED_LIMIT = 2000

    def __init__(self, rss_url: str = "https://arxiv.org/rss/") -> None:
        """Initializes the `RSSPaperExtractor` with the given RSS client.

        Args:
            rss_url: The base URL for the ArXiv RSS feed.
        """
        self.rss_url = rss_url

    def fetch_latest(self, categories: set[model.Category]) -> set[PaperDTO]:
        """Fetch the latest papers from the ArXiv RSS feed for the given categories.

        Args:
            categories: The `Category` domain objects to filter the papers by.

        Raises:
            PaperMissingFieldError: If a required field is missing in the paper.

        Returns:
            A set of `PaperDTO` objects representing the papers.
        """
        categories_queue = [categories]
        pbar = tqdm(total=len(categories_queue), desc="Fetching latest papers from ArXiv")
        paper_dtos: set[PaperDTO] = set()

        while categories_queue:
            categories = categories_queue.pop(0)

            result = self._fetch_papers(categories)
            if self._should_split_categories(result, categories):
                new_categories = self._split_categories(categories)
                categories_queue.extend(new_categories)
                pbar.total += len(new_categories)
                pbar.refresh()

            paper_dtos.update(result)
            pbar.update(1)

        pbar.close()
        return paper_dtos

    def fetch_historical(
        self,
        categories: set[model.Category],
        from_date: datetime.date | None,
        to_date: datetime.date | None,
    ) -> set[PaperDTO]:
        """Historical fetching not supported for ArXiv RSS feed."""
        raise NotImplementedError

    def _fetch_papers(self, categories: set[model.Category]) -> set[PaperDTO]:
        """Fetch the latest papers from the ArXiv RSS feed for the given categories.

        Args:
            categories: The `Category` domain objects to filter the papers by.

        Raises:
            PaperMissingFieldError: If a required field is missing in the paper.

        Returns:
            A set of `PaperDTO` objects representing the papers.
        """
        paper_dtos: set[PaperDTO] = set()
        arxiv_rss_url = f"{self.rss_url}{'+'.join([str(category.identifier) for category in categories])}"
        entries: list[dict[str, Any]] = feedparser.parse(arxiv_rss_url).get("entries", [])  # type: ignore[no-untyped-call]

        for entry in entries:
            try:
                paper_dto = RSSPaperEntry(**entry).to_paper_dto()
            except ValidationError as e:
                raise PaperMissingFieldError(entry) from e
            paper_dtos.add(paper_dto)

        return paper_dtos

    def _should_split_categories(self, result: set[PaperDTO], categories: set[model.Category]) -> bool:
        """Checks if the categories should be split based on the result and category identifiers.

        Args:
            result: A set of `PaperDTO` objects representing the papers fetched from the extractor.
            categories: A set of `Category` domain objects representing the categories used
                to fetch the result.

        Returns:
            True if the categories should be split, False otherwise.
        """
        return len(result) == self.RSS_FEED_LIMIT and (
            len(categories) >= 2 or (len(categories) == 1 and next(iter(categories)).identifier.subcategory is None)
        )

    @staticmethod
    def _split_categories(categories: set[model.Category]) -> list[set[model.Category]]:
        """Splits the set of categories into two halves.

        If set contains a single category, it is expanded to all subcategories which are then
        split into two halves.

        Args:
            categories: A set of `Category` domain objects representing the categories used
                to fetch the result.

        Returns:
            A list of sets of `Category` domain objects representing the split categories.
        """
        if len(categories) <= 1:
            categories = categories.pop().subcategories

        mid = len(categories) // 2
        return [set(list(categories)[:mid]), set(list(categories)[mid:])]


class JSONPaperEntry(BaseModel):
    """A model representing a single JSON paper entry."""

    id: str
    """The unique identifier for the paper, typically in the format 'arxiv:xxxx.xxxx'."""
    title: str
    """The title of the paper."""
    abstract: str
    """The abstract of the paper."""
    update_date: str
    """The last update date of the paper in 'YYYY-MM-DD' format."""
    categories: str
    """A space-separated string of categories the paper belongs to."""

    def to_paper_dto(self) -> PaperDTO:
        """Converts the JSON paper entry to a `PaperDTO`.

        Returns:
            A `PaperDTO` object containing the paper's details.
        """
        return PaperDTO(
            arxiv_id=self.id.strip(),
            title=" ".join(self.title.split()).strip(),
            abstract=" ".join(self.abstract.split()).strip(),
            published_at=datetime.datetime.strptime(self.update_date, "%Y-%m-%d").date(),
            categories=set(self.categories.split()),
        )


class JSONPaperExtractor(AbstractPaperExtractor):
    """A fetcher that extracts papers from a JSON file."""

    def __init__(self, file_path: str | Path) -> None:
        """Initialize the `JSONPaperExtractor`.

        Args:
            file_path: The file path of the `json` file.
        """
        self._file_path = Path(file_path)

    def fetch_latest(self, categories: set[model.Category]) -> set[PaperDTO]:
        """Latest fetching not supported for json extractor."""
        raise NotImplementedError

    def fetch_historical(
        self,
        categories: set[model.Category],
        from_date: datetime.date | None,
        to_date: datetime.date | None,
    ) -> set[PaperDTO]:
        """Fetches historical papers for the given categories and time range.

        Args:
            categories: The `Cateogry` domain objects to filter the papers by.
            from_date: Optional from date filter (inclusive).
            to_date: Optional to date filter (inclusive).

        Raises:
            PaperMissingFieldError: If a required field is missing in the paper.

        Returns:
            A set of `PaperDTO` objects representing the papers.
        """
        with self._file_path.open("rb") as f:
            n_lines = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1024 * 1024), b""))  # type: ignore[arg-type]

        categories |= {subcategory for category in categories for subcategory in category.subcategories}
        entries: set[PaperDTO] = set()
        with self._file_path.open(encoding="utf-8") as f:
            for line in tqdm(
                f,
                total=n_lines,
                mininterval=0.5,
                miniters=1000,
                desc="Fetching historical papers from JSON",
                dynamic_ncols=True,
            ):
                entry = JSONPaperEntry(**json.loads(line)).to_paper_dto()

                if not any(str(category.identifier) in entry.categories for category in categories):
                    continue
                if from_date and entry.published_at < from_date:
                    continue
                if to_date and entry.published_at > to_date:
                    continue

                entries.add(entry)

        return entries


class ArXivCategoryExtractor(AbstractCategoryExtractor):
    """A category extractor that fetches categories from the ArXiv website."""

    CATEGORY_PATTERN = re.compile(r"([a-zA-Z\-]+)(?:\.([a-zA-Z\-]+))?\s*\(([^)]+)\)")
    ARCHIVE_PATTERN = re.compile(r"^(.*?)\s*\(")

    def __init__(self, url: str = "https://arxiv.org/category_taxonomy") -> None:
        """Initializes the `ArXivCategoryExtractor` with the given URL.

        Args:
            url: The URL for the ArXiv category taxonomy page.
        """
        self.url = url

    def fetch_categories(self) -> set[CategoryDTO]:
        """Fetches all categories from ArXiv.

        Raises:
            CategoryFetchError: If fetching categories fails.
            CategoryParseError: If parsing categories fails.

        Returns:
            A set of `CategoryDTO` objects representing the categories.
        """
        soup = self._fetch_and_parse_html()
        return self._extract_categories(soup)

    def _fetch_and_parse_html(self) -> Tag:
        """Fetches the HTML content from the ArXiv category taxonomy page and parses it.

        Raises:
            CategoryFetchError: If there is an error fetching the categories.
            CategoryParseError: If there is an error parsing the categories.

        Returns:
            A BeautifulSoup Tag object representing the category taxonomy list.
        """
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            msg = f"Failed to fetch the categories from {self.url}: {e}"
            raise CategoryFetchError(msg) from e

        soup = BeautifulSoup(response.text, "html.parser").find("div", id="category_taxonomy_list")
        if not isinstance(soup, Tag):
            msg = "Failed to find the category taxonomy list in the HTML response."
            raise CategoryParseError(msg)
        return soup

    def _extract_categories(self, soup: Tag) -> set[CategoryDTO]:
        """Extracts categories from the BeautifulSoup object.

        Args:
            soup: The BeautifulSoup object representing the category taxonomy list.

        Raises:
            CategoryParseError: If there is an error parsing the categories.

        Returns:
            A set of `CategoryDTO` objects representing the categories.
        """
        categories: set[CategoryDTO] = set()
        group_name: str | None = None
        archive_name: str | None = None
        archive: str | None = None
        subcategory: str | None = None
        category_name: str | None = None

        for element in soup.find_all(["h2", "h3", "h4", "p"]):
            if not isinstance(element, Tag):
                continue

            match element.name:
                case "h2":
                    group_name, archive_name, archive, subcategory, category_name = self._parse_group_header(element)
                case "h3":
                    archive_name, archive, subcategory, category_name = self._parse_archive_header(element)
                case "h4":
                    archive, subcategory, category_name = self._parse_category_header(element)
                case "p":
                    if archive is None:
                        msg = f"Missing archive for category {category_name!r} in group {group_name!r}"
                        raise CategoryParseError(msg)

                    categories.add(
                        self._create_category_dto(
                            element,
                            archive,
                            subcategory,
                            archive_name or group_name,
                            category_name,
                        ),
                    )

        archive_categories = {
            CategoryDTO(
                archive=category.archive,
                subcategory=None,
                archive_name=category.archive_name,
                category_name=None,
                description=None,
            )
            for category in categories
            if category.subcategory is not None
        }
        return categories | archive_categories

    @staticmethod
    def _parse_group_header(element: Tag) -> tuple[str, None, None, None, None]:
        """Parses the group header element.

        Args:
            element: The HTML element representing the group header.

        Returns:
            A tuple containing the group name and None values for other fields.
        """
        return str(element.text).strip(), None, None, None, None

    def _parse_archive_header(self, element: Tag) -> tuple[str | None, None, None, None]:
        """Parses the archive header element.

        Args:
            element: The HTML element representing the archive header.
            group_name: The name of the group to which the archive belongs.

        Returns:
            _description_
        """
        archive_name = self._extract_archive_name(element.text)
        return archive_name, None, None, None

    def _parse_category_header(self, element: Tag) -> tuple[str, str | None, str]:
        """Parses the category header element.

        Args:
            element: The HTML element representing the category header.

        Raises:
            CategoryParseError: If the category header format is invalid.

        Returns:
            A tuple containing the archive, subcategory, and category name.
        """
        category_data = self._extract_category_data(element.text)
        if category_data is None:
            msg = f"Failed to parse category header {element.text!r}"
            raise CategoryParseError(msg)

        return category_data

    @staticmethod
    def _create_category_dto(
        element: Tag,
        archive: str,
        subcategory: str | None,
        archive_name: str | None,
        category_name: str | None,
    ) -> CategoryDTO:
        """Creates a `CategoryDTO` object from the given parameters.

        Args:
            element: The HTML element representing the category description.
            archive: The archive to which the category belongs.
            subcategory: The subcategory of the category.
            archive_name: The name of the category archive.
            category_name: The name of the category.

        Returns:
            A `CategoryDTO` object representing the category.
        """
        return CategoryDTO(
            archive=archive,
            subcategory=subcategory,
            archive_name=archive_name,
            category_name=category_name,
            description=str(element.text).strip(),
        )

    def _extract_category_data(self, text: str | None) -> tuple[str, str | None, str] | None:
        """Extracts the category data from the given text.

        Looks for a pattern like "archive.subcategory (category name)".

        Args:
            text: The text to extract the category data from.

        Returns:
            A tuple containing the archive, subcategory, and category name, or None if not found.
        """
        if not text:
            return None

        match = self.CATEGORY_PATTERN.match(text)
        if match:
            before_period, after_period, inside_parentheses = match.groups()
            return before_period, after_period or None, inside_parentheses
        return None

    def _extract_archive_name(self, text: str | None) -> str | None:
        """Extracts the archive name from the given text.

        Looks for a pattern like "archive (description)".

        Args:
            text: The text to extract the archive name from.

        Returns:
            The extracted archive name, or None if not found.
        """
        if not text:
            return None

        match = self.ARCHIVE_PATTERN.match(text)
        return match.group(1).strip() if match else None
