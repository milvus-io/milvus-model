import os
from typing import List, Optional

import requests

from pymilvus.model.base import BaseRerankFunction, RerankResult


API_URL = "https://api.contextual.ai/v1/rerank"


class ContextualAIRerankFunction(BaseRerankFunction):
    def __init__(
        self,
        model_name: str = "ctxl-rerank-v2-instruct-multilingual",
        api_key: Optional[str] = None,
        instruction: Optional[str] = None,
    ):
        if api_key is None:
            if "CONTEXTUAL_API_KEY" in os.environ and os.environ["CONTEXTUAL_API_KEY"]:
                self.api_key = os.environ["CONTEXTUAL_API_KEY"]
            else:
                error_message = (
                    "Did not find api_key, please add an environment variable"
                    " `CONTEXTUAL_API_KEY` which contains it, or pass"
                    "  `api_key` as a named parameter."
                )
                raise ValueError(error_message)
        else:
            self.api_key = api_key

        self.model_name = model_name
        self.instruction = instruction
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )

    def __call__(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        metadata: Optional[List[str]] = None,
    ) -> List[RerankResult]:
        payload = {
            "query": query,
            "documents": documents,
            "model": self.model_name,
            "top_n": top_k,
        }
        if self.instruction is not None:
            payload["instruction"] = self.instruction

        # metadata is optional; if provided, must match documents length;
        if metadata is None:
            payload["metadata"] = [""] * len(documents)
        else:
            if len(metadata) != len(documents):
                raise ValueError("metadata length must match documents length")
            payload["metadata"] = metadata

        resp = self._session.post(API_URL, json=payload).json()  

        if "results" not in resp:
            # API returns {"detail": ...} on error
            message = resp.get("detail", resp)
            raise RuntimeError(message)

        results: List[RerankResult] = []
        for item in resp["results"]:
            # Contextual AI response contains index and relevance_score; no document text returned
            results.append(
                RerankResult(
                    text="",  # API does not return the document text
                    score=item["relevance_score"],
                    index=item["index"],
                )
            )
        return results


