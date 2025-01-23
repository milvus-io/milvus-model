from typing import List, Optional

import numpy as np
import requests

from pymilvus.model.base import BaseEmbeddingFunction


class TEIEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        api_url: str,
        dimensions: Optional[int] = None,
    ):
        self.api_url = api_url + "/v1/embeddings"
        self._session = requests.Session()
        self._dim = dimensions

    @property
    def dim(self):
        if self._dim is None:
            # This works by sending a dummy message to the API to retrieve the vector dimension,
            # as the original API does not directly provide this information
            self._dim = self._call_api(["get dim"])[0].shape[0]
        return self._dim

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._call_api(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._call_api(documents)

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._call_api(texts)

    def _call_api(self, texts: List[str]):
        data = {"input": texts}
        resp = self._session.post(  # type: ignore[assignment]
            self.api_url,
            json=data,
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["message"])

        embeddings = resp["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore[no-any-return]
        return [np.array(result["embedding"]) for result in sorted_embeddings]
