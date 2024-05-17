from typing import List, Optional

from milvus_model.base import BaseRerankFunction, RerankResult
from milvus_model.utils import import_voyageai

import_voyageai()
import voyageai

class VoyageRerankFunction(BaseRerankFunction):
    def __init__(self, model_name: str = "rerank-lite-1", api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = voyageai.Client(api_key=api_key)

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        vo_results = self.client.rerank(query, documents, model=self.model_name, top_k=top_k)
        results = []
        for vo_result in vo_results.results:
            results.append(
                RerankResult(
                    text=vo_result.document, score=vo_result.relevance_score, index=vo_result.index
                )
            )
        return results
