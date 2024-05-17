from typing import Any, List

from milvus_model.base import BaseRerankFunction, RerankResult
from milvus_model.utils import import_sentence_transformers

import_sentence_transformers()
import sentence_transformers

class CrossEncoderRerankFunction(BaseRerankFunction):
    def __init__(
        self,
        model_name: str = "",
        device: str = "",
        batch_size: int = 32,
        activation_fct: Any = None,
        **kwargs,
    ):
        if sentence_transformers is None:
            error_message = "sentence_transformer is not installed."
            raise ImportError(error_message)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.activation_fct = activation_fct
        self.model = sentence_transformers.cross_encoder.CrossEncoder(
            model_name=model_name, device=self.device, default_activation_function=activation_fct
        )

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        query_document_pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(query_document_pairs, batch_size=self.batch_size)

        ranked_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if top_k:
            ranked_order = ranked_order[:top_k]

        results = []
        for index in ranked_order:
            results.append(RerankResult(text=documents[index], score=scores[index], index=index))
        return results
