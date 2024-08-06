__all__ = ["BGEM3EmbeddingFunction", "MGTEEmbeddingFunction"]

from milvus_model.utils.lazy_import import LazyImport

bge_m3 = LazyImport("bge_m3", globals(), "milvus_model.hybrid.bge_m3")
mgte = LazyImport("mgte", globals(), "milvus_model.hybrid.mgte")

def BGEM3EmbeddingFunction(*args, **kwargs):
    return bge_m3.BGEM3EmbeddingFunction(*args, **kwargs)

def MGTEEmbeddingFunction(*args, **kwargs):
    return mgte.MGTEEmbeddingFunction(*args, **kwargs)
