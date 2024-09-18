import os
from typing import List, Optional

import numpy as np
import requests

from milvus_model.base import BaseEmbeddingFunction

API_URL = "https://api.jina.ai/v1/embeddings"


class JinaEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "jina-embeddings-v3",
        api_key: Optional[str] = None,
        task: str = 'retrieval.passage',
        dimensions: Optional[int] = None,
        late_chunking: Optional[bool] = False,
        **kwargs,
    ):
        if api_key is None:
            if "JINAAI_API_KEY" in os.environ and os.environ["JINAAI_API_KEY"]:
                self.api_key = os.environ["JINAAI_API_KEY"]
            else:
                error_message = (
                    "Did not find api_key, please add an environment variable"
                    " `JINAAI_API_KEY` which contains it, or pass"
                    "  `api_key` as a named parameter."
                )
                raise ValueError(error_message)
        else:
            self.api_key = api_key
        self.model_name = model_name
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Accept-Encoding": "identity"}
        )
        self.task = task
        self._dim = dimensions
        self.late_chunking = late_chunking

    @property
    def dim(self):
        if self._dim is None:
            self._dim = self._call_jina_api([""])[0].shape[0]
        return self._dim

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._call_jina_api(queries, task='retrieval.query')

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._call_jina_api(documents, task='retrieval.passage')

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._call_jina_api(texts, task=self.task)

    def _call_jina_api(self, texts: List[str], task: Optional[str] = None):
        data = {
            "input": texts,
            "model": self.model_name,
            "task": task,
            "late_chunking": self.late_chunking,
        }
        if self._dim is not None:
            data["dimensions"] = self._dim
        resp = self._session.post(  # type: ignore[assignment]
            API_URL,
            json=data,
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        embeddings = resp["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore[no-any-return]
        return [np.array(result["embedding"]) for result in sorted_embeddings]
