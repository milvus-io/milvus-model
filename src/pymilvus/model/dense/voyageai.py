from collections import defaultdict
from typing import List, Optional

import numpy as np
import struct

from pymilvus.model.base import BaseEmbeddingFunction
from pymilvus.model.utils import import_voyageai


class VoyageEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(self,
                 model_name: str = "voyage-3-large",
                 api_key: Optional[str] = None,
                 embedding_type: Optional[str] = None,
                 truncate: Optional[bool] = None,
                 dimension: Optional[int] = None,
                 **kwargs):
        import_voyageai()
        import voyageai

        self.model_name = model_name
        self.truncate = truncate
        self._voyageai_model_meta_info = defaultdict(dict)
        self._voyageai_model_meta_info["voyage-3-large"]["dim"] = [1024, 256, 512, 2048]
        self._voyageai_model_meta_info["voyage-code-3"]["dim"] = [1024, 256, 512, 2048]
        self._voyageai_model_meta_info["voyage-3"]["dim"] = [1024]
        self._voyageai_model_meta_info["voyage-3-lite"]["dim"] = [512]
        self._voyageai_model_meta_info["voyage-finance-2"]["dim"] = [1024]
        self._voyageai_model_meta_info["voyage-multilingual-2"]["dim"] = [1024]
        self._voyageai_model_meta_info["voyage-law-2"]["dim"] = [1024]
        #old model
        self._voyageai_model_meta_info["voyage-large-2"]["dim"] = [1536]
        self._voyageai_model_meta_info["voyage-code-2"]["dim"] = [1536]
        self._voyageai_model_meta_info["voyage-2"]["dim"] = [1024]
        self._voyageai_model_meta_info["voyage-lite-02-instruct"]["dim"] = [1024]

        if dimension is not None and dimension not in self._voyageai_model_meta_info[self.model_name]["dim"]:
            raise ValueError(f"The provided dimension ({dimension}) is not supported by the selected model ({self.model_name}). "
                             "Leave this parameter empty to use the default dimension for the model. "
                             "Please check the supported dimensions here: https://docs.voyageai.com/docs/embeddings"
                             )

        if embedding_type == "int8" or embedding_type == "uint8":
            raise ValueError("Currently int8 or uint8 is not supported with PyMilvus model library.")

        if self.model_name in ['voyage-3-large', 'voyage-code-3']:
            if embedding_type is not None and embedding_type not in ['float', 'binary', 'ubinary']:
                raise ValueError(f"The provided embedding_type ({embedding_type}) is not supported by the selected model "
                                 f"({self.model_name}). Leave this parameter empty for the default embedding_type (float). "
                                 f"Please check the supported embedding_type values here: https://docs.voyageai.com/docs/embeddings")
        else:
            if embedding_type is not None and embedding_type != 'float':
                raise ValueError(f"The provided embedding_type ({embedding_type}) is not supported by the selected model "
                                 f"({self.model_name}). Leave this parameter empty for the default embedding_type (float). "
                                 f"Please check the supported embedding_type values here: https://docs.voyageai.com/docs/embeddings")

        self.embedding_type = embedding_type
        self.dimension = dimension
        self.client = voyageai.Client(api_key, **kwargs)

    @property
    def dim(self):
        if self.dimension is None:
            return self._voyageai_model_meta_info[self.model_name]["dim"][0]
        return self.dimension

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._call_voyage_api(queries, input_type="query")

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._call_voyage_api(documents, input_type="document")

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._call_voyage_api(texts)

    def _call_voyage_api(self, texts: List[str], input_type: Optional[str] = None):
        embeddings = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type=input_type,
            truncation=self.truncate,
            output_dtype=self.embedding_type,
            output_dimension=self.dim,
        ).embeddings

        if self.embedding_type is None or self.embedding_type == "float":
            results = [np.array(data, dtype=np.float32) for data in embeddings]
        elif self.embedding_type == "binary":
            results = [
                np.unpackbits((np.array(result, dtype=np.int16) + 128).astype(np.uint8)).astype(bool)
                for result in embeddings
            ]
        elif self.embedding_type == "ubinary":
            results = [np.unpackbits(np.array(result, dtype=np.uint8)).astype(bool) for result in embeddings]
        else:
            raise ValueError(f"The provided embedding_type ({self.embedding_type}) is not supported by the selected model "
                             f"({self.model_name}). Leave this parameter empty for the default embedding_type (float). "
                             f"Please check the supported embedding_type values here: https://docs.voyageai.com/docs/embeddings")
        return results
