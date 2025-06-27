from typing import Literal, overload

import openai
from sentence_transformers import SentenceTransformer

from paperloom.application.ports.embedding_model import AbstractEmbeddingModel, EmbeddingModelError


class OpenAIEmbeddingModel(AbstractEmbeddingModel):
    """OpenAI embedding model for converting strings to vectors using OpenAI API."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: Literal["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
    ) -> None:
        """Initializes the `OpenAIEmbeddingModel` instance with the OpenAI client and model name.

        Args:
            client: The OpenAI client instance.
            model: The name of the OpenAI model to use for embedding.
        """
        self.client = client
        self.model = model

    @property
    def dimensions(self) -> int:
        """Get the dimensions of the embedding model."""
        return len(self.embed_string(""))

    @overload
    def embed_string(self, text: str) -> list[float]: ...

    @overload
    def embed_string(self, text: list[str]) -> list[list[float]]: ...

    def embed_string(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embeds a string into a list of floats using the OpenAI model.

        Args:
            text: The string or list of strings to embed.

        Raises:
            EmbeddingModelError: If there is an error with the embedding request.

        Returns:
            A list of floats representing the embedded string or a list of lists of floats
            if multiple strings are provided.
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return [item.embedding for item in response.data] if isinstance(text, list) else response.data[0].embedding
        except Exception as e:
            error_msg = f"Error embedding string with OpenAI model {self.model!r}."
            raise EmbeddingModelError(error_msg) from e


class HuggingFaceEmbeddingModel(AbstractEmbeddingModel):
    """Hugging Face embedding model for converting strings to vectors using local models."""

    def __init__(self, model: SentenceTransformer) -> None:
        """Initialize the `HuggingFaceEmbeddingModel` instance with the specified model.

        Args:
            model: The Hugging Face model to use for embedding.
        """
        self.model = model

    @property
    def dimensions(self) -> int:
        """Get the dimensions of the embedding model."""
        return len(self.embed_string(""))

    @overload
    def embed_string(self, text: str) -> list[float]: ...

    @overload
    def embed_string(self, text: list[str]) -> list[list[float]]: ...

    def embed_string(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embeds a string into a list of floats using the Hugging Face model.

        Args:
            text: The string or list of strings to embed.

        Raises:
            EmbeddingModelError: If there is an error with the embedding request.

        Returns:
            A list of floats representing the embedded string or a list of lists of floats
            if multiple strings are provided.
        """
        try:
            return self.model.encode(text).tolist()  # type: ignore[return-type]
        except Exception as e:
            error_msg = f"Error embedding string with Hugging Face model {self.model._get_name()!r}."  # noqa: SLF001
            raise EmbeddingModelError(error_msg) from e
