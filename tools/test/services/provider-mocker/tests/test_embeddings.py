"""The embedding fixture pins similarity by topic marker and by words."""

import math
import unittest

import httpx
from provider_mocker.app import app
from provider_mocker.embeddings import DIMENSIONS, embed


def cosine(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


class EmbeddingFixtureTest(unittest.IsolatedAsyncioTestCase):
    async def post(self, body):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            return await client.post("/v1/embeddings", json=body)

    def test_topic_marker_makes_texts_identical(self):
        on = embed("__EMBED_TOPIC__ lights How do I turn on the lights?")
        off = embed("__EMBED_TOPIC__ lights How do I turn off the lights?")
        self.assertAlmostEqual(cosine(on, off), 1.0)

    def test_unmarked_text_repeats_and_unrelated_text_stays_apart(self):
        text = "Explain why the sky is blue."
        self.assertAlmostEqual(cosine(embed(text), embed(text)), 1.0)
        other = embed("How do I enable two-factor authentication?")
        self.assertLess(cosine(embed(text), other), 0.5)
        self.assertAlmostEqual(math.sqrt(cosine(other, other)), 1.0)

    async def test_api_answers_each_input_in_order(self):
        response = await self.post(
            {"model": "e2e/fixture-embedding", "input": ["a", "b"]}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual([item["index"] for item in data], [0, 1])
        self.assertEqual(len(data[0]["embedding"]), DIMENSIONS)

    async def test_api_rejects_bad_input_and_dimensions(self):
        self.assertEqual((await self.post({"input": 3})).status_code, 400)
        response = await self.post({"input": "a", "dimensions": 4})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
