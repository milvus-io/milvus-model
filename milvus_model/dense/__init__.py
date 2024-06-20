__all__ = [
    "OpenAIEmbeddingFunction",
    "SentenceTransformerEmbeddingFunction",
    "VoyageEmbeddingFunction",
    "JinaEmbeddingFunction",
    "OnnxEmbeddingFunction",
    "CohereEmbeddingFunction",
    "VertexAIEmbeddingFunction",
    "MistralAIEmbeddingFunction",
    "NomicEmbeddingFunction",
]

from milvus_model.utils.lazy_import import LazyImport

jinaai = LazyImport("jinaai", globals(), "milvus_model.dense.jinaai")
openai = LazyImport("openai", globals(), "milvus_model.dense.openai")
sentence_transformer = LazyImport(
    "sentence_transformer", globals(), "milvus_model.dense.sentence_transformer"
)
voyageai = LazyImport("voyageai", globals(), "milvus_model.dense.voyageai")
onnx = LazyImport("onnx", globals(), "milvus_model.dense.onnx")
cohere = LazyImport("cohere", globals(), "milvus_model.dense.cohere")
vertexai = LazyImport("vertexai", globals(), "milvus_model.dense.vertexai")
mistralai = LazyImport("mistralai", globals(), "milvus_model.dense.mistralai")
nomic = LazyImport("nomic", globals(), "milvus_model.dense.nomic")


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


def CohereEmbeddingFunction(*args, **kwargs):
    return cohere.CohereEmbeddingFunction(*args, **kwargs)


def VertexAIEmbeddingFunction(*args, **kwargs):
    return vertexai.VertexAIEmbeddingFunction(*args, **kwargs)


def MistralAIEmbeddingFunction(*args, **kwargs):
    return mistralai.MistralAIEmbeddingFunction(*args, **kwargs)


def NomicEmbeddingFunction(*args, **kwargs):
    return nomic.NomicEmbeddingFunction(*args, **kwargs)
