"""
The following code is adapted from/inspired from :
https://huggingface.co/Alibaba-NLP/gte-multilingual-base/blob/main/scripts/gte_embedding.py

# Copyright 2024 The GTE Team Authors and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
import logging

from typing import Dict, List, Optional

from pymilvus.model.base import BaseEmbeddingFunction
from pymilvus.model.sparse.utils import stack_sparse_embeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class MGTEEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        batch_size: int = 16,
        device: str = "",
        normalize_embeddings: bool = True,
        dimensions: Optional[int] = None,
        use_fp16: bool = False,
        return_dense: bool = True,
        return_sparse: bool = True,
        **kwargs,
    ):
        from .mgte_embedding.gte_impl import _GTEEmbeddidng
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.use_fp16 = use_fp16
        self.dimensions = dimensions

        if "dimension" in kwargs: 
            self.dimensions = kwargs["dimension"]
            kwargs.pop("dimension")

        if device == "cpu" and use_fp16 is True:
            logger.warning(
                "Using fp16 with CPU can lead to runtime errors such as 'LayerNormKernelImpl', It's recommended to set 'use_fp16 = False' when using cpu. "
            )

        _model_config = dict(
            {
                "model_name": model_name,
                "device": device,
                "normalized": normalize_embeddings,
                "use_fp16": use_fp16,
            },
            **kwargs,
        )
        _encode_config = {
            "batch_size": batch_size,
            "return_dense": return_dense,
            "return_sparse": return_sparse,
        }
        self._model_config = _model_config
        self._encode_config = _encode_config

        self.model = _GTEEmbeddidng(**self._model_config)

        _encode_config["dimension"] = self.dimensions
        
        if self.dimensions is None:
            self.dimensions = self.model.model.config.hidden_size 

    def __call__(self, texts: List[str]) -> Dict:
        return self._encode(texts)

    @property
    def dim(self) -> Dict:
        return {
            "dense": self.dimensions,
            "sparse": len(self.model.tokenizer),
        }

    def _encode(self, texts: List[str]) -> Dict:
        from scipy.sparse import csr_array

        output = self.model.encode(texts=texts, **self._encode_config)
        results = {}
        if self._encode_config["return_dense"] is True:
            results["dense"] = list(output["dense_embeddings"])
        if self._encode_config["return_sparse"] is True:
            sparse_dim = self.dim["sparse"]
            results["sparse"] = []
            for sparse_vec in output["token_weights"]:
                indices = [int(k) for k in sparse_vec]
                values = list(sparse_vec.values())
                row_indices = [0] * len(indices)
                csr = csr_array((values, (row_indices, indices)), shape=(1, sparse_dim))
                results["sparse"].append(csr)
            results["sparse"] = stack_sparse_embeddings(results["sparse"]).tocsr()
        return results

    def encode_queries(self, queries: List[str]) -> Dict:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> Dict:
        return self._encode(documents)
