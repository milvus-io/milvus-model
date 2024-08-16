from typing import List, Optional
import numpy as np

from milvus_model.base import BaseEmbeddingFunction
from milvus_model.utils import import_sentence_transformers, import_huggingface_hub

import_sentence_transformers()
import_huggingface_hub()

from .instructor_embedding.instructor_impl import Instructor

class InstructorEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "hkunlp/instructor-xl",
        batch_size: int = 32,
        query_instruction: str = "Represent the question for retrieval:",
        doc_instruction: str = "Represent the document for retrieval:",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        _model_config = dict({"model_name_or_path": model_name, "device": device}, **kwargs)
        self.model = Instructor(**_model_config)

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode([[self.doc_instruction, text] for text in texts])

    def _encode(self, texts: List[str]) -> List[np.array]:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        return list(embs)

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        instructed_queries = [[self.query_instruction, query] for query in queries]
        return self._encode(instructed_queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        instructed_documents = [[self.doc_instruction, document] for document in documents]
        return self._encode(instructed_documents)

    def _encode_query(self, query: str) -> np.array:
        instructed_query = self.query_instruction + query
        embs = self.model.encode(
            sentences=[instructed_query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embs[0]

    def _encode_document(self, document: str) -> np.array:
        instructed_document = self.doc_instruction + document
        embs = self.model.encode(
            sentences=[instructed_document],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embs[0]
 

