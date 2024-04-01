from typing import List, Optional

from milvus_model.base import BaseRerankFunction, RerankResult

try:
    import cohere
except ImportError:
    cohere = None


class CohereRerankFunction(BaseRerankFunction):
    def __init__(self, model_name: str = "rerank-english-v2.0", api_key: Optional[str] = None):
        if cohere is None:
            error_message = "cohere is not installed."
            raise ImportError(error_message)
        self.model_name = model_name
        self.client = cohere.Client(api_key)

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        co_results = self.client.rerank(
            query=query, documents=documents, top_n=top_k, model="rerank-english-v2.0"
        )
        results = []
        for co_result in co_results.results:
            results.append(
                RerankResult(
                    text=co_result.document["text"],
                    score=co_result.relevance_score,
                    index=co_result.index,
                )
            )
        return results
