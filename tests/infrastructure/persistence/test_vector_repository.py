import datetime

import pymilvus
import pytest

from paperloom.application.ports.persistence.vector_repository import (
    VectorSearchFilter,
    VectoryRepositoryDeletionError,
    VectoryRepositoryInsertionError,
    VectoryRepositoryQueryError,
)
from paperloom.domain import model
from paperloom.infrastructure.persistence.vector_repository import (
    MilvusPaperVectorRepository,
    MilvusVectorSearchFilterTransformer,
)


class FakeMilvusClient(pymilvus.MilvusClient):
    def __init__(
        self,
        *,
        insert_error: Exception | None = None,
        query_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.collections = {}
        self.insert_error = insert_error
        self.query_error = query_error
        self.delete_error = delete_error

    def has_collection(self, collection_name: str, *args, **kwargs) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, schema: pymilvus.CollectionSchema, *args, **kwargs) -> None:
        self.collections[collection_name] = {"schema": schema, "data": []}

    def drop_collection(self, collection_name: str, *args, **kwargs) -> None:
        if collection_name in self.collections:
            del self.collections[collection_name]

    def create_index(self, *args, **kwargs) -> None:
        pass

    def insert(self, collection_name: str, data: dict | list[dict], *args, **kwargs) -> dict:
        if self.insert_error is not None:
            raise self.insert_error

        if collection_name in self.collections:
            data = [{"entity": item} for item in data] if isinstance(data, list) else {"entity": data}
            self.collections[collection_name]["data"].extend(data)
            return {}
        raise ValueError(f"Collection {collection_name} does not exist.")

    def delete(self, collection_name: str, ids: list, *args, **kwargs) -> dict[str, int]:
        if self.delete_error is not None:
            raise self.delete_error

        if collection_name in self.collections:
            self.collections[collection_name]["data"] = [
                item for item in self.collections[collection_name]["data"] if item["entity"]["arxiv_id"] not in ids
            ]
        return {}

    def search(self, collection_name: str, limit: int, filter: str, *args, **kwargs) -> list[list[dict]]:
        if self.query_error is not None:
            raise self.query_error

        if collection_name in self.collections:
            return [self.collections[collection_name]["data"][:limit]]
        return []


class TestMilvusVectorSearchFilterTransformer:
    def test_transform(self) -> None:
        transformer = MilvusVectorSearchFilterTransformer("categories", "published_at")

        filters = VectorSearchFilter(
            category_identifiers=[
                model.CategoryIdentifier.from_string("cs.AI"),
                model.CategoryIdentifier.from_string("econ"),
            ],
            published_after=datetime.date(2022, 1, 1),
            published_before=datetime.date(2023, 1, 1),
        )

        result = transformer.transform(filters)
        assert result == (
            '(ARRAY_CONTAINS(categories, "cs.AI") && ARRAY_CONTAINS(categories, "econ")) '
            "&& (published_at >= 20220101) && (published_at <= 20230101)"
        )

    def test_transform_empty(self) -> None:
        transformer = MilvusVectorSearchFilterTransformer("categories", "published_at")

        filters = VectorSearchFilter()

        result = transformer.transform(filters)
        assert not result

    def test_transform_no_categories(self) -> None:
        transformer = MilvusVectorSearchFilterTransformer("categories", "published_at")

        filters = VectorSearchFilter(
            published_after=datetime.date(2022, 1, 1),
            published_before=datetime.date(2023, 1, 1),
        )

        result = transformer.transform(filters)
        assert result == "(published_at >= 20220101) && (published_at <= 20230101)"

    def test_transform_no_published_dates(self) -> None:
        transformer = MilvusVectorSearchFilterTransformer("categories", "published_at")

        filters = VectorSearchFilter(
            category_identifiers=[
                model.CategoryIdentifier.from_string("cs.AI"),
                model.CategoryIdentifier.from_string("econ"),
                model.CategoryIdentifier.from_string("math"),
            ],
        )

        result = transformer.transform(filters)
        assert result == (
            '(ARRAY_CONTAINS(categories, "cs.AI") && ARRAY_CONTAINS(categories, "econ") '
            '&& ARRAY_CONTAINS(categories, "math"))'
        )


class TestMilvusVectorRepository:
    def test_insert_embeddings_success(self) -> None:
        fake_milvus_client = FakeMilvusClient()
        repository = MilvusPaperVectorRepository(fake_milvus_client, dimensions=3)

        embeddings = [[0.1, 0.2, 0.3]]
        papers = [
            model.Paper(
                arxiv_id="1234.5678",
                title="",
                abstract="",
                categories={model.Category(model.CategoryIdentifier.from_string("cs.AI"))},
                published_at=datetime.date(2022, 1, 1),
            ),
        ]

        repository.insert_embeddings(embeddings, papers)
        contents = fake_milvus_client.collections[repository.COLLECTION_NAME]["data"]

        assert len(contents) == 1
        assert contents == [
            {
                "entity": {
                    "arxiv_id": "1234.5678",
                    "embedding": [0.1, 0.2, 0.3],
                    "category_identifiers": ["cs.AI"],
                    "published_at": 20220101,
                },
            },
        ]

    def test_insert_embeddings_failure(self) -> None:
        fake_milvus_client = FakeMilvusClient(insert_error=Exception("Some error"))
        repository = MilvusPaperVectorRepository(fake_milvus_client, dimensions=3)

        embeddings = [[0.1, 0.2, 0.3]]
        papers = [
            model.Paper(
                arxiv_id="1234.5678",
                title="",
                abstract="",
                categories={model.Category(model.CategoryIdentifier.from_string("cs.AI"))},
                published_at=datetime.date(2022, 1, 1),
            ),
        ]

        with pytest.raises(
            VectoryRepositoryInsertionError,
            match=f"Error inserting embeddings into Milvus collection {repository.COLLECTION_NAME!r}.",
        ):
            repository.insert_embeddings(embeddings, papers)

    def test_delete_embeddings(self) -> None:
        fake_milvus_client = FakeMilvusClient()
        repository = MilvusPaperVectorRepository(fake_milvus_client, dimensions=3)

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        papers = [
            model.Paper(
                arxiv_id="1234.5678",
                title="",
                abstract="",
                categories={model.Category(model.CategoryIdentifier.from_string("cs.AI"))},
                published_at=datetime.date(2022, 1, 1),
            ),
            model.Paper(
                arxiv_id="9876.5432",
                title="",
                abstract="",
                categories={model.Category(model.CategoryIdentifier.from_string("econ"))},
                published_at=datetime.date(2023, 1, 1),
            ),
        ]
        repository.insert_embeddings(embeddings, papers)
        repository.delete_embeddings(["1234.5678"])

        contents = fake_milvus_client.collections[repository.COLLECTION_NAME]["data"]

        assert len(contents) == 1
        assert contents == [
            {
                "entity": {
                    "arxiv_id": "9876.5432",
                    "embedding": [0.4, 0.5, 0.6],
                    "category_identifiers": ["econ"],
                    "published_at": 20230101,
                },
            },
        ]

    def test_delete_embeddings_failure(self) -> None:
        fake_milvus_client = FakeMilvusClient(delete_error=Exception("Some error"))
        repository = MilvusPaperVectorRepository(fake_milvus_client, dimensions=3)

        with pytest.raises(
            VectoryRepositoryDeletionError,
            match=f"Error deleting embeddings from Milvus collection {repository.COLLECTION_NAME!r}.",
        ):
            repository.delete_embeddings(["1234.5678"])

    def test_query_embedding(self) -> None:
        fake_milvus_client = FakeMilvusClient()
        repository = MilvusPaperVectorRepository(fake_milvus_client, dimensions=3)

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        papers = [
            model.Paper(
                arxiv_id="1234.5678",
                title="",
                abstract="",
                categories={model.Category(model.CategoryIdentifier.from_string("cs.AI"))},
                published_at=datetime.date(2022, 1, 1),
            ),
            model.Paper(
                arxiv_id="9876.5432",
                title="",
                abstract="",
                categories={model.Category(model.CategoryIdentifier.from_string("econ"))},
                published_at=datetime.date(2023, 1, 1),
            ),
        ]
        repository.insert_embeddings(embeddings, papers)

        result = repository.query_embedding([0.1, 0.5, 0.1], top_k=1, filters=None)

        assert result == ["1234.5678"]

    def test_query_embedding_failure(self) -> None:
        fake_milvus_client = FakeMilvusClient(query_error=Exception("Some error"))
        repository = MilvusPaperVectorRepository(fake_milvus_client, dimensions=3)

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        papers = [
            model.Paper(
                arxiv_id="1234.5678",
                title="",
                abstract="",
                categories={model.Category(model.CategoryIdentifier.from_string("cs.AI"))},
                published_at=datetime.date(2022, 1, 1),
            ),
            model.Paper(
                arxiv_id="9876.5432",
                title="",
                abstract="",
                categories={model.Category(model.CategoryIdentifier.from_string("econ"))},
                published_at=datetime.date(2023, 1, 1),
            ),
        ]
        repository.insert_embeddings(embeddings, papers)

        with pytest.raises(
            VectoryRepositoryQueryError,
            match=f"Error querying Milvus collection {repository.COLLECTION_NAME!r}.",
        ):
            repository.query_embedding([0.1, 0.5, 0.1], top_k=1, filters=None)
