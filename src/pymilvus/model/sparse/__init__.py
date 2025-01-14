__all__ = ["SpladeEmbeddingFunction", "BM25EmbeddingFunction"]


from pymilvus.model.utils.lazy_import import LazyImport

bm25 = LazyImport("bm25", globals(), "pymilvus.model.sparse.bm25")
splade = LazyImport("openai", globals(), "pymilvus.model.sparse.splade")

def BM25EmbeddingFunction(*args, **kwargs):
    return bm25.BM25EmbeddingFunction(*args, **kwargs)

def SpladeEmbeddingFunction(*args, **kwargs):
    return splade.SpladeEmbeddingFunction(*args, **kwargs)

