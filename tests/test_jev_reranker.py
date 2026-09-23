import importlib.util
import os
import sys
import unittest
from unittest import mock

# 直接加载 jev.py 源码，绕过 pymilvus.model 包的 __init__（它会触发 onnxruntime 等重依赖导入）
_HERE = os.path.dirname(os.path.abspath(__file__))
_JEV_PATH = os.path.join(_HERE, "..", "src", "pymilvus", "model", "reranker", "jev.py")


def _load_jev_module():
    spec = importlib.util.spec_from_file_location("jev_module_under_test", _JEV_PATH)
    module = importlib.util.module_from_spec(spec)
    # 用假的 base 模块替代真实 base（避免 import pymilvus.model.base 拉起重依赖）
    fake_base = mock.MagicMock()
    fake_base.BaseRerankFunction = object
    fake_base.RerankResult = mock.Mock(side_effect=lambda **kw: _FakeResult(**kw))
    sys.modules["pymilvus.model.base"] = fake_base
    spec.loader.exec_module(module)
    return module


class _FakeResult:
    def __init__(self, text="", score=0.0, index=0):
        self.text = text
        self.score = score
        self.index = index


class TestJevRerankFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_jev_module()

    def setUp(self):
        self.query = "0-dimensional biomaterials lack inductive properties."
        self.documents = [
            "We study 0-dimensional biomaterials and find they lack inductive properties.",
            "We study 3-dimensional biomaterials for tissue engineering.",
            "This paper reviews inductive properties of various materials.",
        ]

    def _sample_response(self):
        return {
            "model": "jev-latest",
            "answers": {
                "d0": {"type": "noul", "noul": 0.96},
                "d1": {"type": "noul", "noul": 0.06},
                "d2": {"type": "noul", "noul": 0.30},
            },
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

    def test_missing_api_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                self.module.JevRerankFunction(api_key=None)

    def test_api_key_from_constructor(self):
        fn = self.module.JevRerankFunction(api_key="test-key")
        self.assertEqual(fn.api_key, "test-key")
        self.assertEqual(fn.model_name, "jev-latest")

    def test_ranking_sorts_by_noul_descending(self):
        fn = self.module.JevRerankFunction(api_key="test-key")
        mock_resp = mock.Mock()
        mock_resp.json.return_value = self._sample_response()
        with mock.patch.object(fn._session, "post", return_value=mock_resp) as post_mock:
            results = fn(self.query, self.documents, top_k=3)

        # 断言请求体结构正确
        payload = post_mock.call_args[1]["json"]
        self.assertEqual(payload["model"], "jev-latest")
        self.assertIn("0-dimensional biomaterials", payload["state"])
        self.assertEqual(set(payload["questions"].keys()), {"d0", "d1", "d2"})
        self.assertEqual(payload["questions"]["d0"]["type"], "noul")

        # 断言按 noul 降序：d0(0.96) > d2(0.30) > d1(0.06)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].index, 0)
        self.assertEqual(results[0].score, 0.96)
        self.assertEqual(results[1].index, 2)
        self.assertEqual(results[2].index, 1)

    def test_top_k_truncates(self):
        fn = self.module.JevRerankFunction(api_key="test-key")
        mock_resp = mock.Mock()
        mock_resp.json.return_value = self._sample_response()
        with mock.patch.object(fn._session, "post", return_value=mock_resp):
            results = fn(self.query, self.documents, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual([r.index for r in results], [0, 2])


if __name__ == "__main__":
    unittest.main()
