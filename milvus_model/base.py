from abc import abstractmethod
from typing import List


class BaseEmbeddingFunction:
    @abstractmethod
    def __call__(self, texts: List[str]):
        """ """

    @abstractmethod
    def encode_queries(self, queries: List[str]):
        """ """


class BaseRerankFunction:
    @abstractmethod
    def __call__(self, query: str, documents: List[str], top_k: int):
        """ """


class RerankResult:
    def __init__(self, text: str, score: float, index: int):
        self.text = text
        self.score = score
        self.index = index

    def to_dict(self):
        return {"text": self.text, "score": self.score, "index": self.index}

    def __str__(self):
        return f"RerankResult(text={self.text!r}, score={self.score}, index={self.index})"

    def __repr__(self):
        return f"RerankResult(text={self.text!r}, score={self.score}, index={self.index})"
