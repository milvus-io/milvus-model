from typing import List, Optional
import struct
from collections import defaultdict
import numpy as np

from milvus_model.base import BaseEmbeddingFunction
from milvus_model.utils import import_cohere

import_cohere()
import cohere

class CohereEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(self,
                 model_name: str = "embed-english-light-v3.0",
                 api_key: Optional[str] = None,
                 input_type: str = "search_document",
                 embedding_types: Optional[List[str]] = None,
                 truncate: Optional[str] = None,
                 **kwargs):
        self.model_name = model_name
        self.input_type = input_type
        self.embedding_types = embedding_types
        self.truncate = truncate

        if isinstance(embedding_types, list):
            if len(embedding_types) > 1:
                raise ValueError("Only one embedding type can be specified using current PyMilvus model library.")
            elif embedding_types[0] == "int8" or embedding_types[0] == "uint8":
                raise ValueError("Currently int8 or uint8 is not supported with PyMilvus model library.")
            else:
                pass
    
        self.client = cohere.Client(api_key, **kwargs)
        self._cohereai_model_meta_info = defaultdict(dict)
        self._cohereai_model_meta_info["embed-english-v3.0"]["dim"] = 1024
        self._cohereai_model_meta_info["embed-english-light-v3.0"]["dim"] = 384
        self._cohereai_model_meta_info["embed-english-v2.0"]["dim"] = 4096
        self._cohereai_model_meta_info["embed-english-light-v2.0"]["dim"] = 1024
        self._cohereai_model_meta_info["embed-multilingual-v3.0"]["dim"] = 1024
        self._cohereai_model_meta_info["embed-multilingual-light-v3.0"]["dim"] = 384
        self._cohereai_model_meta_info["embed-multilingual-v2.0"]["dim"] = 768

    def _call_cohere_api(self, texts: List[str], input_type: str) -> List[np.array]:
        embeddings = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type=input_type,
            embedding_types=self.embedding_types,
            truncate=self.truncate
        ).embeddings
        if self.embedding_types is None:
            results = [np.array(data, dtype=np.float32) for data in embeddings]
        else:
            results = getattr(embeddings, self.embedding_types[0], None)
            if self.embedding_types[0] == "binary":
                results = [struct.pack('b' * len(int8_vector), *int8_vector) for int8_vector in results] 
            elif self.embedding_types[0] == "ubinary":
                results = [struct.pack('B' * len(uint8_vector), *uint8_vector) for uint8_vector in results] 
            elif self.embedding_types[0] == "float":
                results = [np.array(result, dtype=np.float32) for result in results]
            else:
                pass
        return results

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._call_cohere_api(documents, input_type="search_document")

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._call_cohere_api(queries, input_type="search_query")

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._call_cohere_api(texts, self.input_type)

    @property
    def dim(self):
        return self._cohereai_model_meta_info[self.model_name]["dim"]

