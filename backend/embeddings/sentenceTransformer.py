from pydantic.v1 import BaseModel, Field, validator
from embeddings import BaseEmbedding, EmbeddingConfig
from sentence_transformers import SentenceTransformer
from typing import List


class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config.name)
        self.config = config
        self.embedding_model = SentenceTransformer(self.config.name)

    def encode(self, text: str):
        return self.embedding_model.encode(text)

    def __call__(self, input: List[str]) -> List[List[float]]:  
        embeddings = self.embedding_model.encode(input, show_progress_bar=False)
        return embeddings.tolist()

    
