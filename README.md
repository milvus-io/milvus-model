# Milvus Model Lib

The `milvus-model` library provides the integration with common embedding and reranker models for Milvus, a high performance open-source vector database built for AI applications.  `milvus-model` lib is included as a dependency in `pymilvus`, the Python SDK of Milvus.

`milvus-model` supports embedding and reranker models from service providers like OpenAI, Voyage AI, Cohere, and open-source models through SentenceTransformers or Hugging Face [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) .

`milvus-model` supports Python 3.8 and above.

## Installation

If you use `pymilvus`, you can install `milvus-model` through its alias `pymilvus[model]`:
```bash
pip install pymilvus[model] 
# or pip install "pymilvus[model]" for zsh.
```

You can also install it directly:
```bash
pip install pymilvus.model
```

To upgrade milvus-model to the latest version, use:
```
pip install pymilvus.model --upgrade
```
If milvus-model was initially installed as part of the PyMilvus optional components, you should also upgrade PyMilvus to ensure compatibility. This can be done with:
```
pip install pymilvus[model] --upgrade
```
If you need to install a specific version of milvus-model, specify the version number:
```bash
pip install pymilvus.model==0.3.0
```
This command installs version 0.3.0 of milvus-model.




