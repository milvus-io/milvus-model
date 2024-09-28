from typing import List, Optional

from milvus_model.base import BaseRerankFunction, RerankResult
from milvus_model.utils import import_cohere

import_cohere()
import cohere

class CohereRerankFunction(BaseRerankFunction):
    def __init__(self, model_name: str = "rerank-english-v3.0", api_key: Optional[str] = None, return_documents=True, **kwargs):
        self.model_name = model_name
        self.client = cohere.ClientV2(api_key)
        self.rerank_config = {"return_documents": return_documents, **kwargs} 


    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        co_results = self.client.rerank(
            query=query, documents=documents, top_n=top_k, model=self.model_name, **self.rerank_config)
        results = []
        for co_result in co_results.results:
            document_text = ""
            if self.rerank_config["return_documents"] is True:
                document_text = co_result.document.text
            results.append(
                RerankResult(
                    text=document_text,
                    score=co_result.relevance_score,
                    index=co_result.index,
                )
            )
        return results
