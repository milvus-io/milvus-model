__all__ = [
    "OpenAIEmbeddingFunction",
    "SentenceTransformerEmbeddingFunction",
    "VoyageEmbeddingFunction",
    "JinaEmbeddingFunction",
    "OnnxEmbeddingFunction",
]

from milvus_model.utils.lazy_import import LazyImport

jinaai = LazyImport("jinaai", globals(), "milvus_model.dense.jinaai")
openai = LazyImport("openai", globals(), "milvus_model.dense.openai")
sentence_transformer = LazyImport("sentence_transformer", globals(), "milvus_model.dense.sentence_transformer")
voyageai = LazyImport("voyageai", globals(), "milvus_model.dense.voyageai")
onnx = LazyImport("onnx", globals(), "milvus_model.dense.onnx")

def JinaEmbeddingFunction(*args, **kwargs):
    return jinaai.JinaEmbeddingFunction(*args, **kwargs)

def OpenAIEmbeddingFunction(*args, **kwargs):
    return openai.OpenAIEmbeddingFunction(*args, **kwargs)

def SentenceTransformerEmbeddingFunction(*args, **kwargs):
    return sentence_transformer.SentenceTransformerEmbeddingFunction(*args, **kwargs)

def VoyageEmbeddingFunction(*args, **kwargs):
    return voyageai.VoyageEmbeddingFunction(*args, **kwargs)

def OnnxEmbeddingFunction(*args, **kwargs):
    return onnx.OnnxEmbeddingFunction(*args, **kwargs)
