from collections import defaultdict
from typing import List, Optional


import numpy as np

from pymilvus.model.base import BaseEmbeddingFunction
from pymilvus.model.utils import import_google


class GeminiEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "gemini-embedding-exp-03-07",
        api_key: Optional[str] = None,
        config: Optional['types.EmbedContentConfig']=None,
        **kwargs,
    ):
        import_google()
        from google import genai
        from google.genai import types
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key, **kwargs)
        self.config: Optional[types.EmbedContentConfig] = config

        self._gemini_model_meta_info = defaultdict(dict)
        self._gemini_model_meta_info["gemini-embedding-exp-03-07"]["dim"] = 3072
        self._gemini_model_meta_info["models/embedding-001"]["dim"] = 768
        self._gemini_model_meta_info["models/text-embedding-004"]["dim"] = 768

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents)

    @property
    def dim(self):
        if self.config is None or self.config.output_dimensionality is None:
            return self._gemini_model_meta_info[self.model_name]["dim"]
        else:
            return self.config.output_dimensionality

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts)

    def _encode_query(self, query: str) -> np.array:
        return self._encode(query)[0]

    def _encode_document(self, document: str) -> np.array:
        return self._encode(document)[0]

    def _encode(self, texts: List[str]):
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=self.config
        )
        return [np.array(data.values) for data in result.embeddings]
