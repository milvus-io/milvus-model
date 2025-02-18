import numpy as np
from typing import List, Union
from pathlib import Path
from pymilvus.model.base import BaseEmbeddingFunction
from pymilvus.model.utils import import_model2vec


class Model2VecEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(self, model_source: Union[str, Path] = "minishlab/potion-base-8M", **kwargs):
        """
        Initialize the Model2VecEmbeddingFunction, which loads a Model2Vec model either from the Hugging Face Hub or from a local directory.
        Defaults to use the "minishlab/potion-base-8M" model and load from Hugging Face.

        Parameters:
            model_source (Union[str, Path]):
                - If a string is provided and it does not correspond to an existing local directory,
                  it is interpreted as a Hugging Face model identifier (e.g., "minishlab/potion-base-8M").
                - If the provided string (or Path) corresponds to an existing directory, the model is loaded locally.
            **kwargs:
                - Additional keyword arguments that will be passed to the StaticModel.from_pretrained method
                  when loading a remote model from the Hugging Face Hub, including parameters such as
                  huggingface authentication tokens.
        """
        import_model2vec()
        from model2vec import StaticModel

        self.model_source = model_source
        model_path = Path(model_source)

        if model_path.exists() and model_path.is_dir():
            self.model = StaticModel.load_local(model_path)
        else:
            self.model = StaticModel.from_pretrained(model_source, **kwargs)

        dummy_embedding = self.model.encode(["dummy"])
        self._dim = dummy_embedding[0].shape[0]

    @property
    def dim(self) -> int:
        return self._dim

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents)

    def _encode_query(self, query: str) -> np.array:
        return self._encode([query])[0]

    def _encode_document(self, document: str) -> np.array:
        return self._encode([document])[0]

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts)

    def _encode(self, texts: List[str]) -> List[np.array]:
        embeddings = self.model.encode(texts)
        return [embeddings[i] for i in range(embeddings.shape[0])]