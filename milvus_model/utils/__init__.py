__all__ = [
    "import_openai",
    "import_sentence_transformers",
    "import_FlagEmbedding",
    "import_nltk",
    "import_transformers",
    "import_jieba",
    "import_konlpy",
    "import_mecab",
    "import_scipy",
    "import_protobuf",
    "import_unidic_lite",
    "import_cohere",
    "import_voyageai",
    "import_torch",
    "import_huggingface_hub",
    "import_mistralai",
    "import_nomic",
    "import_instructor",
    "import_datasets",
    ]

import importlib.util
from typing import Optional

from milvus_model.utils.dependency_control import prompt_install

def import_openai():
    _check_library("openai", package="openai>=1.12.0")

def import_sentence_transformers():
    _check_library("sentence_transformers", package="sentence-transformers")

def import_FlagEmbedding():
    _check_library("peft", package="peft")
    _check_library("FlagEmbedding", package="FlagEmbedding>=1.2.2")

def import_nltk():
    _check_library("nltk", package="nltk>=3.9.1")

def import_transformers():
    _check_library("transformers", package="transformers>=4.36.0")

def import_jieba():
    _check_library("jieba", package="jieba")

def import_konlpy():
    _check_library("konlpy", package="konlpy")

def import_mecab():
    _check_library("konlpy", package="mecab-python3")

def import_scipy():
    _check_library("scipy", package="scipy>=1.10.0")

def import_protobuf():
    _check_library("protobuf", package="protobuf==3.20.2")

def import_unidic_lite():
    _check_library("unidic-lite", package="unidic-lite")

def import_cohere():
    _check_library("cohere", "cohere")

def import_voyageai():
    _check_library("voyageai", "voyageai>=0.2.0")

def import_torch():
    _check_library("torch", "torch")

def import_huggingface_hub():
    _check_library("huggingface_hub", package="huggingface-hub")

def import_mistralai():
    _check_library("mistralai", package="mistralai")

def import_nomic():
    _check_library("nomic", package="nomic")

def import_instructor():
    _check_library("InstructorEmbedding", package="InstructorEmbedding")

def import_datasets():
    _check_library("datasets", package="datasets")

def _check_library(libname: str, prompt: bool = True, package: Optional[str] = None):
    is_avail = False
    if importlib.util.find_spec(libname):
        is_avail = True
    if not is_avail and prompt:
        prompt_install(package if package else libname)
    return is_avail
