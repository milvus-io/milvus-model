from typing import List
import numpy as np
import os
from collections import defaultdict

from milvus_model.base import BaseEmbeddingFunction
from milvus_model.utils import import_nomic

import_nomic()
from nomic import embed

class NomicEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        api_key: str,
        model_name: str = "nomic-embed-text-v1.5",
        task_type: str = "search_document",
        dimensions: int = 768,
        **kwargs,
    ):
        self._nomic_model_meta_info = defaultdict(dict)
        self._nomic_model_meta_info[model_name]["dim"] = dimensions  # set the dimension

        if api_key is None:
            if "NOMIC_API_KEY" in os.environ and os.environ["NOMIC_API_KEY"]:
                self.api_key = os.environ["NOMIC_API_KEY"]
            else:
                error_message = (
                    "Did not find api_key, please add an environment variable"
                    " `NOMIC_API_KEY` which contains it, or pass"
                    "  `api_key` as a named parameter."
                )
                raise ValueError(error_message)
        else:
            self.api_key = api_key
        self.model_name = model_name
        self.task_type = task_type
        self.dimensionality = dimensions
        if "dimensionality" in kwargs: 
            self.dimensionality = kwargs["dimensionality"]
            kwargs.pop("dimensionality")

        self._encode_config = {
            "model": model_name,
            "task_type": task_type,
            "dimensionality": self.dimensionality,
            **kwargs,
        }

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries, task_type="search_query")

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents, task_type="search_document")

    @property
    def dim(self):
        return self._nomic_model_meta_info[self.model_name]["dim"]

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts, task_type=self.task_type)

    def _encode_query(self, query: str) -> np.array:
        return self._encode([query], task_type="search_query")[0]

    def _encode_document(self, document: str) -> np.array:
        return self._encode([document], task_type="search_document")[0]

    def _call_nomic_api(self, texts: List[str], task_type: str):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        embeddings_batch_response = embed.text(
            texts=texts,
            model=self.model_name,
            task_type=task_type,
            dimensionality=self.dimensionality,
        )
        return [np.array(embedding) for embedding in embeddings_batch_response["embeddings"]]

    def _encode(self, texts: List[str], task_type: str):
        return self._call_nomic_api(texts, task_type)
