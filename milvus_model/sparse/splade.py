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
from typing import Dict, List, Optional

import torch
from scipy.sparse import csr_array

from milvus_model.base import BaseEmbeddingFunction
from milvus_model.utils import import_transformers, import_scipy, import_torch
from milvus_model.sparse.utils import stack_sparse_embeddings

import_torch()
import_scipy()
import_transformers()
from transformers import AutoModelForMaskedLM, AutoTokenizer

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


class _SpladeImplementation:
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        **kwargs,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwargs)
        self.model.to(self.device)
        self.batch_size = batch_size

        self.relu = torch.nn.ReLU()
        self.relu.to(self.device)
        self.model.config.output_hidden_states = True

    def _encode(self, texts: List[str]):
        encoded_input = self.tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
        output = self.model(**encoded_input)
        return output.logits

    def _batchify(self, texts: List[str], batch_size: int) -> List[List[str]]:
        return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def forward(self, texts: List[str], k_tokens: int) -> csr_array:
        with torch.no_grad():
            batched_texts = self._batchify(texts, self.batch_size)
            sparse_embs = []
            for batch_texts in batched_texts:
                logits = self._encode(texts=batch_texts)
                activations = self._get_activation(logits=logits)
                if k_tokens is None:
                    nonzero_indices = torch.nonzero(activations["sparse_activations"])
                    activations["activations"] = nonzero_indices
                else:
                    activations = self._update_activations(**activations, k_tokens=k_tokens)
                batch_csr = self._convert_to_csr_array(activations)
                sparse_embs.extend(batch_csr)
        return stack_sparse_embeddings(sparse_embs).tocsr()

    def _get_activation(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"sparse_activations": torch.amax(torch.log1p(self.relu(logits)), dim=1)}

    def _update_activations(self, sparse_activations: torch.Tensor, k_tokens: int) -> torch.Tensor:
        activations = torch.topk(input=sparse_activations, k=k_tokens, dim=1).indices

        # Set value of max sparse_activations which are not in top k to 0.
        sparse_activations = sparse_activations * torch.zeros(
            (sparse_activations.shape[0], sparse_activations.shape[1]),
            dtype=int,
            device=self.device,
        ).scatter_(dim=1, index=activations.long(), value=1)

        activations = torch.cat(
            (
                torch.arange(activations.shape[0], device=activations.device)
                .repeat_interleave(activations.shape[1])
                .reshape(-1, 1),
                activations.reshape((-1, 1)),
            ),
            dim=1,
        )

        return {
            "activations": activations,
            "sparse_activations": sparse_activations,
        }

    def _filter_activations(
        self, activations: torch.Tensor, k_tokens: int, **kwargs
    ) -> torch.Tensor:
        _, activations = torch.topk(input=activations, k=k_tokens, dim=1, **kwargs)
        return activations

    def _convert_to_csr_array(self, activations: Dict):

        values = (
            activations["sparse_activations"][
                activations["activations"][:, 0], activations["activations"][:, 1]
            ]
            .cpu()
            .detach()
            .numpy()
        )

        row_indices = activations["activations"][:, 0].cpu().detach().numpy()
        col_indices = activations["activations"][:, 1].cpu().detach().numpy()
        return csr_array(
            (values.flatten(), (row_indices, col_indices)),
            shape=activations["sparse_activations"].shape,
        )
