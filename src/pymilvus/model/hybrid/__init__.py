__all__ = ["BGEM3EmbeddingFunction", "MGTEEmbeddingFunction"]

from pymilvus.model.utils.lazy_import import LazyImport

bge_m3 = LazyImport("bge_m3", globals(), "pymilvus.model.hybrid.bge_m3")
mgte = LazyImport("mgte", globals(), "pymilvus.model.hybrid.mgte")

def BGEM3EmbeddingFunction(*args, **kwargs):
    return bge_m3.BGEM3EmbeddingFunction(*args, **kwargs)

def MGTEEmbeddingFunction(*args, **kwargs):
    return mgte.MGTEEmbeddingFunction(*args, **kwargs)
