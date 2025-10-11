"""
Embedding service - generate OpenAI embeddings for nodes and edges
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI
import time


class EmbeddingService:
    """Generate embeddings using OpenAI API"""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_retries: int = 3
    ):
        """
        Initialize embedding service

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            batch_size: Number of texts to embed in one API call
            max_retries: Maximum retry attempts for failed requests
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * 1536  # Default dimension for text-embedding-3-small

        # Truncate if too long (max 8191 tokens for OpenAI)
        text = text[:8000]

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"   Embedding failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    print(f"   ⚠️  Embedding failed after {self.max_retries} attempts: {e}")
                    # Return zero vector on failure
                    return [0.0] * 1536

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts but remember positions
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text[:8000])  # Truncate
                valid_indices.append(i)

        if not valid_texts:
            # All texts empty, return zero vectors
            return [[0.0] * 1536 for _ in texts]

        # Generate embeddings
        embeddings = [None] * len(texts)

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=valid_texts
                )

                # Map embeddings back to original positions
                for i, idx in enumerate(valid_indices):
                    embeddings[idx] = response.data[i].embedding

                # Fill empty positions with zero vectors
                for i in range(len(embeddings)):
                    if embeddings[i] is None:
                        embeddings[i] = [0.0] * 1536

                return embeddings

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"   Batch embedding failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    print(f"   ⚠️  Batch embedding failed: {e}")
                    # Return zero vectors
                    return [[0.0] * 1536 for _ in texts]

    def create_node_text(self, node_data: Dict[str, Any], fields: Optional[List[str]] = None) -> str:
        """
        Create text representation of node for embedding

        Args:
            node_data: Node properties
            fields: Specific fields to include (None = all)

        Returns:
            Text representation
        """
        if fields:
            # Use only specified fields
            parts = [f"{k}: {v}" for k, v in node_data.items() if k in fields and v is not None]
        else:
            # Use all non-null fields
            parts = [f"{k}: {v}" for k, v in node_data.items() if v is not None]

        return " | ".join(parts)

    def create_edge_text(
        self,
        from_node_text: str,
        to_node_text: str,
        relationship_type: str
    ) -> str:
        """
        Create text representation of edge for embedding

        Args:
            from_node_text: Text of source node
            to_node_text: Text of target node
            relationship_type: Relationship type

        Returns:
            Text representation
        """
        return f"{from_node_text} {relationship_type} {to_node_text}"

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model"""
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimension_map.get(self.model, 1536)
