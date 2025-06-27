import datetime
from typing import Literal

import click
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from paperloom import config
from paperloom.application.ports.paper_extractor import CategoryFetchError, CategoryParseError, PaperMissingFieldError
from paperloom.application.services import (
    fetch_and_store_categories,
    fetch_and_store_historical_papers,
    fetch_and_store_latest_papers,
)
from paperloom.infrastructure.paper_extractor import ArXivCategoryExtractor, JSONPaperExtractor, RSSPaperExtractor
from paperloom.infrastructure.persistence.orm import Base
from paperloom.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork

engine = create_engine(config.DATABASE_URL)
Base.metadata.create_all(engine)
session_factory = sessionmaker(bind=engine)

uow = SqlAlchemyUnitOfWork(session_factory)

arxiv_category_extractor = ArXivCategoryExtractor()


@click.group()
def cli() -> None:
    """The paperloom CLI."""


@cli.command()
def fetch_categories() -> None:
    """Fetch and store categories from ArXiv."""
    try:
        categories = fetch_and_store_categories(uow, arxiv_category_extractor)
    except CategoryFetchError as e:
        click.echo(f"Error fetching categories: {e}")
        return
    except CategoryParseError as e:
        click.echo(f"Error parsing categories: {e}")
        return

    click.echo(f"Fetched and stored {len(categories)} categories from ArXiv.")


@cli.command()
@click.option(
    "--extractor-type",
    "-t",
    type=click.Choice(["rss", "json"], case_sensitive=False),
    default="rss",
    help="The type of paper extractor to use. Defaults to 'rss'.",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help="Path to a file containing data to be processed. Only used if --type is 'json'.",
)
@click.option(
    "--categories",
    "-c",
    multiple=True,
    required=False,
    help="The categories to fetch papers from. (e.g. 'cs', 'cs.AI', ...)",
)
@click.option(
    "--all-categories",
    "-a",
    is_flag=True,
    help="Fetch papers from all available categories.",
)
def fetch_latest_papers(
    extractor_type: Literal["rss", "json"],
    input_file: str | None,
    categories: list[str],
    all_categories: bool,
) -> None:
    """Fetch and store the latest papers from ArXiv."""
    if all_categories:
        click.echo("Fetching papers from all categories.")
    elif not categories:
        click.echo("Error: You must specify either --categories or --all-categories.", err=True)
        return
    else:
        click.echo(f"Fetching papers from {len(categories)} categories: {categories!r}")

    if extractor_type == "json":
        if not input_file:
            click.echo("Error: --input-file is required when using --type json.", err=True)
            return
        click.echo(f"Using JSON extractor with input file: {input_file}")
        arxiv_paper_extractor = JSONPaperExtractor(file_path=input_file)
    elif extractor_type == "rss":
        click.echo("Using RSS extractor.")
        arxiv_paper_extractor = RSSPaperExtractor()

    try:
        papers = fetch_and_store_latest_papers(
            category_strings=categories or None,
            paper_extractor=arxiv_paper_extractor,
            uow=uow,
        )
    except PaperMissingFieldError as e:
        click.echo(f"Error fetching papers: {e}")
        return

    click.echo(f"Fetched {len(papers)} papers from ArXiv.")


@cli.command()
@click.option(
    "--extractor-type",
    "-t",
    type=click.Choice(["rss", "json"], case_sensitive=False),
    default="json",
    help="The type of paper extractor to use. Defaults to 'json'.",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help="Path to a file containing data to be processed. Only used if --type is 'json'.",
)
@click.option(
    "--categories",
    "-c",
    multiple=True,
    required=False,
    help="The categories to fetch papers from. (e.g. 'cs', 'cs.AI', ...)",
)
@click.option(
    "--all-categories",
    "-a",
    is_flag=True,
    help="Fetch papers from all available categories.",
)
@click.option(
    "--from-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=False,
    help="Fetch papers published after this date (inclusive).",
)
@click.option(
    "--to-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=False,
    help="Fetch papers published before this date (inclusive).",
)
def fetch_historical_papers(
    extractor_type: Literal["rss", "json"],
    input_file: str | None,
    categories: list[str],
    all_categories: bool,
    from_date: datetime.date | None,
    to_date: datetime.date | None,
) -> None:
    """Fetch and store historical papers from ArXiv."""
    if all_categories:
        click.echo("Fetching historical papers from all categories.")
    elif not categories:
        click.echo("Error: You must specify either --categories or --all-categories.", err=True)
        return
    else:
        click.echo(f"Fetching historical papers from {len(categories)} categories: {categories!r}")

    if extractor_type == "json":
        if not input_file:
            click.echo("Error: --input-file is required when using --type json.", err=True)
            return
        click.echo(f"Using JSON extractor with input file: {input_file}")
        arxiv_paper_extractor = JSONPaperExtractor(file_path=input_file)
    elif extractor_type == "rss":
        click.echo("Using RSS extractor.")
        arxiv_paper_extractor = RSSPaperExtractor()

    try:
        papers = fetch_and_store_historical_papers(
            category_strings=categories or None,
            paper_extractor=arxiv_paper_extractor,
            uow=uow,
            from_date=from_date,
            to_date=to_date,
        )
    except PaperMissingFieldError as e:
        click.echo(f"Error fetching historical papers: {e}")
        return

    click.echo(f"Fetched {len(papers)} historical papers from ArXiv.")


if __name__ == "__main__":
    cli()
