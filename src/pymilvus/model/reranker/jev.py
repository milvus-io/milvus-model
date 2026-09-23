import os
from typing import List, Optional

import requests

from pymilvus.model.base import BaseRerankFunction, RerankResult

API_URL = "https://api.typesafe.ai/v1/systemone"


class JevRerankFunction(BaseRerankFunction):
    """A reranker backed by TypeSafe's Jev model.

    Jev (``jev-latest``) is a structured-decision model that scores each
    candidate against a query using a ``noul`` (yes/no probability) question.
    One HTTP request scores all candidates in parallel.
    """

    def __init__(self, model_name: str = "jev-latest", api_key: Optional[str] = None):
        if api_key is None:
            if "TYPESAFE_API_KEY" in os.environ and os.environ["TYPESAFE_API_KEY"]:
                self.api_key = os.environ["TYPESAFE_API_KEY"]
            else:
                error_message = (
                    "Did not find api_key, please add an environment variable"
                    " `TYPESAFE_API_KEY` which contains it, or pass"
                    "  `api_key` as a named parameter."
                )
                raise ValueError(error_message)
        else:
            self.api_key = api_key
        self.model_name = model_name
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        questions = {}
        for i, doc in enumerate(documents):
            questions[f"d{i}"] = {
                "type": "noul",
                "instructions": (
                    f"Document: {doc} "
                    f"— Does this document provide evidence relevant to the claim?"
                ),
            }
        payload = {
            "state": f"Scientific claim: {query}",
            "model": self.model_name,
            "questions": questions,
        }
        resp = self._session.post(API_URL, json=payload).json()
        if "answers" not in resp:
            raise RuntimeError(f"Unexpected Jev response: {resp}")

        answers = resp["answers"]
        scored = []
        for i, doc in enumerate(documents):
            key = f"d{i}"
            if key not in answers:
                continue
            score = answers[key].get("noul")
            if score is None:
                continue
            scored.append(RerankResult(text=doc, score=float(score), index=i))
        scored.sort(key=lambda r: -r.score)
        return scored[:top_k]
