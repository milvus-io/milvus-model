__all__ = ["SpladeEmbeddingFunction", "BM25EmbeddingFunction"]


from milvus_model.utils.lazy_import import LazyImport

bm25 = LazyImport("bm25", globals(), "milvus_model.sparse.bm25")
splade = LazyImport("openai", globals(), "milvus_model.sparse.splade")

def BM25EmbeddingFunction(*args, **kwargs):
    return bm25.BM25EmbeddingFunction(*args, **kwargs)

def SpladeEmbeddingFunction(*args, **kwargs):
    return splade.SpladeEmbeddingFunction(*args, **kwargs)

