from typing import List, Optional
import os
import numpy as np
from collections import defaultdict

from milvus_model.base import BaseEmbeddingFunction
from milvus_model.utils import import_mistralai

import_mistralai()
from mistralai import Mistral

class MistralAIEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        api_key: str,
        model_name: str = "mistral-embed",
        **kwargs,
    ):
        self._mistral_model_meta_info = defaultdict(dict)
        self._mistral_model_meta_info[model_name]["dim"] = 1024  # fixed dimension

        if api_key is None:
            if "MISTRALAI_API_KEY" in os.environ and os.environ["MISTRALAI_API_KEY"]:
                self.api_key = os.environ["MISTRALAI_API_KEY"]
            else:
                error_message = (
                    "Did not find api_key, please add an environment variable"
                    " `MISTRALAI_API_KEY` which contains it, or pass"
                    "  `api_key` as a named parameter."
                )
                raise ValueError(error_message)
        else:
            self.api_key = api_key
        self.model_name = model_name
        self.client = Mistral(api_key=api_key)
        self._encode_config = {"model": model_name, **kwargs}

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents)

    @property
    def dim(self):
        return self._mistral_model_meta_info[self.model_name]["dim"]

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts)

    def _encode_query(self, query: str) -> np.array:
        return self._encode([query])[0]

    def _encode_document(self, document: str) -> np.array:
        return self._encode([document])[0]

    def _call_mistral_api(self, texts: List[str]):
        embeddings_batch_response = self.client.embeddings.create(
            inputs=texts,
            **self._encode_config
        )
        return [np.array(data.embedding) for data in embeddings_batch_response.data]

    def _encode(self, texts: List[str]):
        return self._call_mistral_api(texts)
