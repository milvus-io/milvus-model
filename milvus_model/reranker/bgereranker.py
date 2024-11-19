from typing import Any, List, Optional, Union

import torch

from milvus_model.base import BaseRerankFunction, RerankResult
from milvus_model.utils import import_FlagEmbedding, import_transformers

import_FlagEmbedding()
import_transformers()
from FlagEmbedding import FlagAutoReranker
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    is_torch_npu_available,
)

class BGERerankFunction(BaseRerankFunction):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        batch_size: int = 32,
        normalize: bool = True,
        device: Optional[Union[str, List]] = None,
        query_max_length: int = 256,
        max_length: int = 512,
        **kwargs: Any,
    ):

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device

        if "devices" in kwargs:
            device = devices
            kwargs.pop("devices")

        _model_config = dict(
            {
                "model_name_or_path": model_name,
                "batch_size": batch_size,
                "use_fp16": use_fp16,
                "devices": device,
                "max_length": max_length,
                "query_max_length": query_max_length,
                "normalize": normalize,
            },
            **kwargs,
        )
        self.reranker = FlagAutoReranker.from_finetuned(**_model_config)


    def _batchify(self, texts: List[str], batch_size: int) -> List[List[str]]:
        return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]:
        query_document_pairs = [[query, doc] for doc in documents]
        batched_texts = self._batchify(documents, self.batch_size)
        scores = []
        for batched_text in batched_texts:
            query_document_pairs = [[query, text] for text in batched_text]
            batch_score = self.reranker.compute_score(
                query_document_pairs, normalize=self.normalize
            )
            scores.extend(batch_score)
        ranked_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        if top_k:
            ranked_order = ranked_order[:top_k]

        results = []
        for index in ranked_order:
            results.append(RerankResult(text=documents[index], score=scores[index], index=index))
        return results

