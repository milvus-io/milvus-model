import os
from typing import List, Optional

import numpy as np
import requests

from milvus_model.base import BaseEmbeddingFunction

API_URL = "https://api.jina.ai/v1/embeddings"


class JinaEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "jina-embeddings-v2-base-en",
        api_key: Optional[str] = None,
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
        self.model_name = model_name
        self._dim = None

    @property
    def dim(self):
        if self._dim is None:
            self._dim = self._call_jina_api([""])[0].shape[0]
        return self._dim

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._call_jina_api(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._call_jina_api(documents)

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._call_jina_api(texts)

    def _call_jina_api(self, texts: List[str]):
        resp = self._session.post(  # type: ignore[assignment]
            API_URL, json={"input": texts, "model": self.model_name},
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        embeddings = resp["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore[no-any-return]
        return [np.array(result["embedding"]) for result in sorted_embeddings]
