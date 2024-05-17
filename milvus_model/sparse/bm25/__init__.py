__all__ = [
    "BM25EmbeddingFunction",
    "Analyzer",
    "build_analyzer_from_yaml",
    "build_default_analyzer",
]

from milvus_model.utils.lazy_import import LazyImport

bm25 = LazyImport("bm25", globals(), "milvus_model.sparse.bm25.bm25")
tokenizers = LazyImport("tokenizers", globals(), "milvus_model.sparse.bm25.tokenizers")

def BM25EmbeddingFunction(*args, **kwargs):
    return bm25.BM25EmbeddingFunction(*args, **kwargs)

def Analyzer(*args, **kwargs):
    return tokenizers.Analyzer(*args, **kwargs)

def build_analyzer_from_yaml(*args, **kwargs):
    return tokenizers.build_analzyer_from_yaml(*args, **kwargs)

def build_default_analyzer(*args, **kwargs):
    return tokenizers.build_default_analyzer(*args, **kwargs)

