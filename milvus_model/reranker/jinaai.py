import os
from typing import List, Optional

import requests

from milvus_model.base import BaseRerankFunction, RerankResult

API_URL = "https://api.jina.ai/v1/rerank"


class JinaRerankFunction(BaseRerankFunction):
    def __init__(self, model_name: str = "jina-reranker-v1-base-en", api_key: Optional[str] = None):
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

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        resp = self._session.post(  # type: ignore[assignment]
            API_URL,
            json={
                "query": query,
                "documents": documents,
                "model": self.model_name,
                "top_n": top_k,
            },
        ).json()
        if "results" not in resp:
            raise RuntimeError(resp["detail"])

        results = []
        for res in resp["results"]:
            results.append(
                RerankResult(
                    text=res["document"]["text"], score=res["relevance_score"], index=res["index"]
                )
            )
        return results
