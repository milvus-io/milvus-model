import os
from typing import List, Optional

import requests

from pymilvus.model.base import BaseRerankFunction, RerankResult

API_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class DashscopeRerankFunction(BaseRerankFunction):
    def __init__(self, model_name: str = "qwen3-rerank", api_key: Optional[str] = None,  **kwargs):
        if api_key is None:
            if "DASHSCOPE_API_KEY" in os.environ and os.environ["DASHSCOPE_API_KEY"]:
                self.api_key = os.environ["DASHSCOPE_API_KEY"]
            else:
                error_message = (
                    "Did not find api_key, please add an environment variable"
                    " `DASHSCOPE_API_KEY` which contains it, or pass"
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
        self.rerank_config = {**kwargs} 

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        json_data = {
            "model": self.model_name,
            "input":{
                "query": query,
                "documents": documents,
            },
        }
        if self.rerank_config:
            json_data["parameters"] = self.rerank_config
        else:
            json_data["parameters"] = {}
        json_data["parameters"]["top_n"] = top_k
        resp = self._session.post(  # type: ignore[assignment]
            API_URL,
            json=json_data,
        ).json()
        if "output" not in resp:
            raise RuntimeError(resp["output"])

        results = []
        for res in resp["output"]["results"]:
            results.append(
                RerankResult(
                    text=res.get("document", {}).get("text", ""), 
                    score=res["relevance_score"], 
                    index=res["index"]
                )
            )
        return results