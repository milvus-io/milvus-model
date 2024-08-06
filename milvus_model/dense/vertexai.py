from typing import List, Optional
import numpy as np
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from collections import defaultdict
import os

from google.auth import credentials as auth_credentials


class VertexAIEmbeddingFunction:
    def __init__(
        self,
        model_name: str = "text-embedding-004",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = 256,
        task: str = "SEMANTIC_SIMILARITY",
        project_id: Optional[str] = None,
        location: str = "us-central1",
        credentials: Optional[auth_credentials.Credentials] = None,
        **kwargs,
    ):
        self._vertexai_model_meta_info = defaultdict(dict)
        self._model_config = dict({"api_key": api_key, "base_url": base_url}, **kwargs)
        if dimensions is not None:
            self._vertexai_model_meta_info[model_name]["dim"] = dimensions
        if api_key:
            self.api_key = api_key
        elif "VERTEXAI_API_KEY" in os.environ and os.environ["VERTEXAI_API_KEY"]:
            self.api_key = os.environ["VERTEXAI_API_KEY"]
        elif credentials:
            self.credentials = credentials
        else:
            raise ValueError(
                "Did not find api_key or credentials, please add an environment variable"
                " `VERTEXAI_API_KEY` which contains it,"
                " pass `api_key` as a named parameter, or"
                " pass `credentials` as a named parameter."
            )

        self._encode_config = {"model": model_name, "dimensions": dimensions}
        self.task = task
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.client = TextEmbeddingModel.from_pretrained(model_name)

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        self.task = "RETRIEVAL_QUERY"
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        self.task = "RETRIEVAL_DOCUMENT"
        return self._encode(documents)

    @property
    def dim(self):
        return self._vertexai_model_meta_info[self.model_name]["dim"]

    def __call__(self, texts: List[str], task: str = "SEMANTIC_SIMILARITY") -> List[np.array]:
        self.task = task
        return self._encode(texts)

    def _encode_query(self, query: str) -> np.array:
        self.task = "RETRIEVAL_QUERY"
        return self._encode([query])[0]

    def _encode_document(self, document: str) -> np.array:
        self.task = "RETRIEVAL_DOCUMENT"
        return self._encode([document])[0]

    def _call_vertexai_api(self, texts: List[str]):
        inputs = [TextEmbeddingInput(text, self.task) for text in texts]
        kwargs = dict(output_dimensionality=self._vertexai_model_meta_info[self.model_name]["dim"])
        embeddings = self.client.get_embeddings(inputs, **kwargs)
        return [np.array(embedding.values) for embedding in embeddings]

    def _encode(self, texts: List[str]):
        return self._call_vertexai_api(texts)
