"""
The following code is adapted from/inspired by the 'neural-cherche' project:
https://github.com/raphaelsty/neural-cherche
Specifically, neural-cherche/neural_cherche/models/splade.py

MIT License

Copyright (c) 2023 Raphael Sourty

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
from typing import List, Optional

from scipy.sparse import csr_array

from pymilvus.model.base import BaseEmbeddingFunction


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SpladeEmbeddingFunction(BaseEmbeddingFunction):
    model_name: str

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        batch_size: int = 32,
        query_instruction: str = "",
        doc_instruction: str = "",
        device: Optional[str] = "cpu",
        k_tokens_query: Optional[int] = None,
        k_tokens_document: Optional[int] = None,
        **kwargs,
    ):
        from .splade_embedding.splade_impl import _SpladeImplementation
        self.model_name = model_name

        _model_config = dict(
            {"model_name_or_path": model_name, "batch_size": batch_size, "device": device},
            **kwargs,
        )
        self._model_config = _model_config
        self.model = _SpladeImplementation(**self._model_config)
        self.device = device
        self.k_tokens_query = k_tokens_query
        self.k_tokens_document = k_tokens_document
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction

    def __call__(self, texts: List[str]) -> csr_array:
        return self._encode(texts, None)

    def encode_documents(self, documents: List[str]) -> csr_array:
        return self._encode(
            [self.doc_instruction + document for document in documents], self.k_tokens_document,
        )

    def _encode(self, texts: List[str], k_tokens: int) -> csr_array:
        return self.model.forward(texts, k_tokens=k_tokens)

    def encode_queries(self, queries: List[str]) -> csr_array:
        return self._encode(
            [self.query_instruction + query for query in queries], self.k_tokens_query,
        )

    @property
    def dim(self) -> int:
        return len(self.model.tokenizer)

    def _encode_query(self, query: str) -> csr_array:
        return self.model.forward([self.query_instruction + query], k_tokens=self.k_tokens_query)[0]

    def _encode_document(self, document: str) -> csr_array:
        return self.model.forward(
            [self.doc_instruction + document], k_tokens=self.k_tokens_document
        )[0]

