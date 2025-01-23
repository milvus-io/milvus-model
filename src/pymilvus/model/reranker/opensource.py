from typing import List

import requests

from pymilvus.model.base import BaseRerankFunction, RerankResult


class OpenSourceFunction(BaseRerankFunction):
    def __init__(self, api_url: str):
        self.api_url = api_url
        self._session = requests.Session()

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        resp = self._session.post(  # type: ignore[assignment]
            self.api_url,
            json={
                "query": query,
                "raw_scores": False,
                "return_text": True,
                "texts": documents,
                "truncate": False,
            },
        ).json()
        if "error" in resp:
            raise RuntimeError(resp["error"])

        results = []
        for res in resp:
            results.append(RerankResult(text=res["text"], score=res["score"], index=res["index"]))
        return results
