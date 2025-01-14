__all__ = [
    "CohereRerankFunction",
    "BGERerankFunction",
    "VoyageRerankFunction",
    "CrossEncoderRerankFunction",
    "JinaRerankFunction",
]

from pymilvus.model.utils.lazy_import import LazyImport

bgegreranker = LazyImport("bgereranker", globals(), "pymilvus.model.reranker.bgereranker")
cohere = LazyImport("cohere", globals(), "pymilvus.model.reranker.cohere")
cross_encoder = LazyImport("cross_encoder", globals(), "pymilvus.model.reranker.cross_encoder")
jinaai = LazyImport("jinaai", globals(), "pymilvus.model.reranker.jinaai")
voyageai = LazyImport("voyageai", globals(), "pymilvus.model.reranker.voyageai")

def BGERerankFunction(*args, **kwargs):
    return bgegreranker.BGERerankFunction(*args, **kwargs)

def CohereRerankFunction(*args, **kwargs):
    return cohere.CohereRerankFunction(*args, **kwargs)

def CrossEncoderRerankFunction(*args, **kwargs):
    return cross_encoder.CrossEncoderRerankFunction(*args, **kwargs)

def JinaRerankFunction(*args, **kwargs):
    return jinaai.JinaRerankFunction(*args, **kwargs)

def VoyageRerankFunction(*args, **kwargs):
    return voyageai.VoyageRerankFunction(*args, **kwargs)
