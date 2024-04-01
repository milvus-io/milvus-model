# Milvus Model Lib

The milvus-model library provides model components for PyMilvus, the official Python SDK for Milvus, an open-source vector database built for AI applications.

## Installation

`milvus-model` supports Python 3.8 and above. Install it via pip:
```bash
pip install milvus-model
```
This command installs milvus-model directly. If you're working within the PyMilvus ecosystem and want to include milvus-model as an optional component, you can install it using:
```bash
pip install pymilvus[model] 
# or pip install "pymilvus[model]" for zsh.
```
To upgrade milvus-model to the latest version, use:
```
pip install milvus-model --upgrade
```
If milvus-model was initially installed as part of the PyMilvus optional components, you should also upgrade PyMilvus to ensure compatibility. This can be done with:
```
pip install pymilvus[model] --upgrade
```
If you need to install a specific version of milvus-model, specify the version number:
```bash
pip install milvus-model==0.2.0
```
This command installs version 0.2.0 of milvus-model.




