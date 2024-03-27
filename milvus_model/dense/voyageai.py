from collections import defaultdict
from typing import List, Optional

import numpy as np

from milvus_model.base import BaseEmbeddingFunction


class VoyageEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(self, model_name: str = "voyage-2", api_key: Optional[str] = None, **kwargs):
        try:
            import voyageai
        except ImportError as err:
            error_message = "voyageai is not installed."
            raise ImportError(error_message) from err
        self.model_name = model_name
        self._voyageai_model_meta_info = defaultdict(dict)
        self._voyageai_model_meta_info["voyage-large-2"]["dim"] = 1536
        self._voyageai_model_meta_info["voyage-code-2"]["dim"] = 1536
        self._voyageai_model_meta_info["voyage-2"]["dim"] = 1024
        self._voyageai_model_meta_info["voyage-lite-02-instruct"]["dim"] = 1024
        self.client = voyageai.Client(api_key, **kwargs)

    @property
    def dim(self):
        return self._voyageai_model_meta_info[self.model_name]["dim"]

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._call_voyage_api(queries, input_type="query")

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._call_voyage_api(documents, input_type="document")

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._call_voyage_api(texts)

    def _call_voyage_api(self, texts: List[str], input_type: Optional[str] = None):
        results = self.client.embed(
            texts=texts, model=self.model_name, input_type=input_type
        ).embeddings
        return [np.array(data) for data in results]
