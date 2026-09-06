import json
import unittest

from agentforge.llm import LLMConfig, LLMError, OpenAICompatibleClient


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


class LLMTests(unittest.TestCase):
    def test_sends_auth_and_decodes_json(self):
        seen = {}

        def opener(req, timeout):
            seen["authorization"] = req.headers["Authorization"]
            seen["timeout"] = timeout
            return Response({"choices": [{"message": {"content": '{"answer": "ok"}'}}]})

        client = OpenAICompatibleClient(
            LLMConfig("secret-from-test", "https://example.test/v1", "test-model", 4),
            opener,
        )
        self.assertEqual(client.complete_json("system", "user"), {"answer": "ok"})
        self.assertEqual(seen["authorization"], "Bearer secret-from-test")
        self.assertEqual(seen["timeout"], 4)

    def test_rejects_invalid_response(self):
        client = OpenAICompatibleClient(
            LLMConfig("key", "https://example.test/v1", "model"),
            lambda req, timeout: Response({"choices": []}),
        )
        with self.assertRaises(LLMError):
            client.complete_json("system", "user")


if __name__ == "__main__":
    unittest.main()
