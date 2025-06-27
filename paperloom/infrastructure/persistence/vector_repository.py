from pymilvus import CollectionSchema, DataType, MilvusClient
from pymilvus.milvus_client import IndexParams

from paperloom.application.ports.persistence.vector_repository import (
    AbstractPaperVectorRepository,
    VectorSearchFilter,
    VectoryRepositoryDeletionError,
    VectoryRepositoryInsertionError,
    VectoryRepositoryQueryError,
)
from paperloom.domain.model import Paper


class MilvusVectorSearchFilterTransformer:
    """Transformer for converting a `VectorSearchFilter` to a Milvus filter string."""

    def __init__(self, category_identifiers_field_name: str, published_at_field_name: str) -> None:
        """Initializes the transformer with field names.

        Args:
            category_identifiers_field_name: The field name for category identifiers in the Milvus collection.
            published_at_field_name: The field name for published date in the Milvus collection.
        """
        self.categories_field_name = category_identifiers_field_name
        self.published_at_field_name = published_at_field_name

    def transform(self, filters: VectorSearchFilter) -> str:
        """Transform the `VectorSearchFilter` to a Milvus filter string.

        Args:
            filters: The `VectorSearchFilter` object containing the filters to apply.

        Returns:
            The generated Milvus filter string.
        """
        milvus_filter: list[str] = []
        if filters.category_identifiers:
            milvus_filter.append(
                " && ".join(
                    f'ARRAY_CONTAINS({self.categories_field_name}, "{category_identifier}")'
                    for category_identifier in filters.category_identifiers
                ),
            )
        if filters.published_after:
            milvus_filter.append(
                f"{self.published_at_field_name} >= {filters.published_after.strftime('%Y%m%d')}",
            )
        if filters.published_before:
            milvus_filter.append(
                f"{self.published_at_field_name} <= {filters.published_before.strftime('%Y%m%d')}",
            )
        milvus_filter = [f"({f})" for f in milvus_filter]
        return " && ".join(milvus_filter)


class MilvusPaperVectorRepository(AbstractPaperVectorRepository):
    """Milvus vector repository for storing and querying vector embeddings of `Paper`."""

    COLLECTION_NAME = "paper"
    """Name of the Milvus collection."""

    ARXIV_ID_FIELD_NAME = "arxiv_id"
    """Field name for ArXiv ID in the Milvus collection."""

    EMBEDDING_FIELD_NAME = "embedding"
    """Field name for embeddings in the Milvus collection."""

    CATEGORY_IDENTIFIERS_FIELD_NAME = "category_identifiers"
    """Field name for category identifiers in the Milvus collection."""

    PUBLISHED_AT_FIELD_NAME = "published_at"
    """Field name for published date in the Milvus collection."""

    def __init__(self, milvus_client: MilvusClient, dimensions: int) -> None:
        """Initializes the `MilvusVectorRepository` instance with the Milvus client and vector dimensions.

        Args:
            milvus_client: The Milvus client instance.
            dimensions: The number of dimensions for the vector embeddings.
        """
        self.client = milvus_client
        self.dimensions = dimensions
        self.filter_transformer = MilvusVectorSearchFilterTransformer(
            category_identifiers_field_name=self.CATEGORY_IDENTIFIERS_FIELD_NAME,
            published_at_field_name=self.PUBLISHED_AT_FIELD_NAME,
        )

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensure the Milvus collection exists, creating it if it does not."""
        if self.client.has_collection(self.COLLECTION_NAME):
            self.client.drop_collection(self.COLLECTION_NAME)

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            schema=self._define_schema(),
        )

        self.client.create_index(
            collection_name=self.COLLECTION_NAME,
            index_params=self._define_index(),
        )

    def _define_schema(self) -> CollectionSchema:
        """Define the schema for the Milvus collection.

        Returns:
            The `CollectionSchema` for the Milvus collection.
        """
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(self.ARXIV_ID_FIELD_NAME, DataType.VARCHAR, max_length=20, is_primary=True)
        schema.add_field(self.EMBEDDING_FIELD_NAME, DataType.FLOAT_VECTOR, dim=self.dimensions)
        schema.add_field(
            self.CATEGORY_IDENTIFIERS_FIELD_NAME,
            DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_length=20,
        )
        schema.add_field(self.PUBLISHED_AT_FIELD_NAME, DataType.INT64)
        return schema

    def _define_index(self) -> IndexParams:
        """Define the index for the Milvus collection.

        Returns:
            The `IndexParams` for the Milvus collection.
        """
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=self.EMBEDDING_FIELD_NAME,
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name=f"{self.EMBEDDING_FIELD_NAME}_index",
            params={"nlist": 128},
        )
        return index_params

    def insert_embeddings(self, embeddings: list[list[float]], papers: list[Paper]) -> None:
        """Insert embeddings and metadata into the vector repository.

        Args:
            embeddings: List of embeddings to insert.
            papers: List of `Paper` domain objects corresponding to the embeddings. Certain fields
                from the `Paper` object are used as metadata for the embeddings.

        Raises:
            VectoryRepositoryInsertionError: If there is an error inserting the embeddings into the Milvus collection.
        """
        data = [
            {
                self.ARXIV_ID_FIELD_NAME: paper.arxiv_id,
                self.EMBEDDING_FIELD_NAME: embedding,
                self.CATEGORY_IDENTIFIERS_FIELD_NAME: [str(category.identifier) for category in paper.categories],
                self.PUBLISHED_AT_FIELD_NAME: paper.published_at_int,
            }
            for embedding, paper in zip(embeddings, papers, strict=True)
        ]

        try:
            self.client.insert(collection_name=self.COLLECTION_NAME, data=data)
        except Exception as e:
            error_msg = f"Error inserting embeddings into Milvus collection {self.COLLECTION_NAME!r}."
            raise VectoryRepositoryInsertionError(error_msg) from e

    def delete_embeddings(self, arxiv_ids: list[str]) -> None:
        """Delete embeddings from the vector repository.

        Args:
            arxiv_ids: List of ArXiv IDs of the embeddings to delete.

        Raises:
            VectoryRepositoryDeletionError: If there is an error deleting the embeddings from the Milvus collection.
        """
        try:
            self.client.delete(collection_name=self.COLLECTION_NAME, ids=arxiv_ids)
        except Exception as e:
            error_msg = f"Error deleting embeddings from Milvus collection {self.COLLECTION_NAME!r}."
            raise VectoryRepositoryDeletionError(error_msg) from e

    def query_embedding(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        filters: VectorSearchFilter | None = None,
    ) -> list[str]:
        """Query the vector repository for similar embeddings.

        Args:
            query_embedding: The embedding to query against.
            top_k: The number of similar embeddings to return.
            filters: Optional filters to apply to the query.

        Raises:
            VectoryRepositoryQueryError: If there is an error querying the Milvus collection.

        Returns:
            List of metadata for the top_k similar embeddings.
        """
        milvus_filter_str = self.filter_transformer.transform(filters) if filters else ""
        try:
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                data=[query_embedding],
                limit=top_k,
                filter=milvus_filter_str,
                output_fields=[self.ARXIV_ID_FIELD_NAME],
            )
            return [result.get("entity", {})[self.ARXIV_ID_FIELD_NAME] for result in results[0]]
        except Exception as e:
            error_msg = f"Error querying Milvus collection {self.COLLECTION_NAME!r}."
            raise VectoryRepositoryQueryError(error_msg) from e
