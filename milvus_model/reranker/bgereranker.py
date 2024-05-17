from typing import List, Optional, Union

import torch

from milvus_model.base import BaseRerankFunction, RerankResult
from milvus_model.utils import import_FlagEmbedding, import_transformers

import_FlagEmbedding()
import_transformers()
from FlagEmbedding import FlagReranker
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
        device: Optional[str] = None,
    ):

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.reranker = _FlagReranker(model_name, use_fp16=use_fp16, device=device)

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


class _FlagReranker(FlagReranker):
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        use_fp16: bool = False,
        cache_dir: Optional[str] = None,
        device: Optional[Union[str, int]] = None,
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, cache_dir=cache_dir
        )

        if device and isinstance(device, str):
            self.device = torch.device(device)
            if device == "cpu":
                use_fp16 = False
        elif torch.cuda.is_available():
            if device is not None:
                self.device = torch.device(f"cuda:{device}")
            else:
                self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16:
            self.model.half()

        self.model = self.model.to(self.device)

        self.model.eval()

        if device is None:
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)
        else:
            self.num_gpus = 1
