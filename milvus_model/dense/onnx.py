
import onnxruntime

from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
import numpy as np
from typing import List

from milvus_model.base import BaseEmbeddingFunction

class OnnxEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(self, model_name: str = "GPTCache/paraphrase-albert-onnx", tokenizer_name: str = "GPTCache/paraphrase-albert-small-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model_name = model_name
        onnx_model_path = hf_hub_download(repo_id=model_name, filename="model.onnx")
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        config = AutoConfig.from_pretrained(
           tokenizer_name
        )
        self.__dimension = config.hidden_size

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts)

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        return self._encode(documents)

    def _encode(self, texts: List[str]) -> List[np.array]:
        return [self._to_embedding(text) for text in texts]

    def _to_embedding(self, data: str, **_):
        encoded_text = self.tokenizer.encode_plus(data, padding="max_length")

        ort_inputs = {
            "input_ids": np.array(encoded_text["input_ids"]).astype("int64").reshape(1, -1),
            "attention_mask": np.array(encoded_text["attention_mask"]).astype("int64").reshape(1, -1),
            "token_type_ids": np.array(encoded_text["token_type_ids"]).astype("int64").reshape(1, -1),
        }

        ort_outputs = self.ort_session.run(None, ort_inputs)
        ort_feat = ort_outputs[0]
        emb = self._post_proc(ort_feat, ort_inputs["attention_mask"])
        emb =  emb.flatten()
        return emb / np.linalg.norm(emb)

    def _post_proc(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], -1)
            .astype(float)
        )
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    @property
    def dim(self):
        return self.__dimension

