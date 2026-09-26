"""Request-level regression for the zero memory threshold fallback (#4174)."""

import json
import math
import os
import time
import uuid

import requests

from memory_tests.base import HTTP_OK, MemoryFeaturesTest

MIN_ORTHOGONAL_RESIDUAL_NORM = 1e-6


class MemoryDefaultThresholdTest(MemoryFeaturesTest):
    """A score between 0.60 and 0.70 must be rejected when the plugin says 0."""

    MARKER = "MEMORY_ZERO_MARKER"
    SCORE = 0.65

    @staticmethod
    def _unit(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise AssertionError("Embedding API returned a zero-norm vector")
        return [value / norm for value in vector]

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        health_url = os.environ.get("ROUTER_HEALTH_ENDPOINT", "")
        self.assertTrue(
            health_url.endswith("/ready"),
            "The memory integration runner must supply the router API readiness URL",
        )
        endpoint = health_url.removesuffix("/ready") + "/api/v1/diagnostics/embeddings"
        response = requests.post(
            endpoint,
            json={"model": "mmbert", "dimension": 768, "texts": texts},
            timeout=self.timeout,
        )
        self.assertEqual(response.status_code, HTTP_OK, response.text)
        embeddings = response.json().get("embeddings", [])
        self.assertEqual(len(embeddings), len(texts))
        for item in embeddings:
            self.assertEqual(item.get("model_used"), "mmbert")
            self.assertEqual(len(item.get("embedding", [])), 768)
        return [self._unit(item["embedding"]) for item in embeddings]

    def _score_between_defaults(
        self, zero_query: list[float], control_query: list[float]
    ) -> tuple[list[float], float]:
        """Build a unit row at 0.65 to zero_query, aligned with the control."""
        projection = self._dot(control_query, zero_query)
        residual = [
            control - projection * zero
            for control, zero in zip(control_query, zero_query, strict=True)
        ]
        residual_norm = math.sqrt(sum(value * value for value in residual))
        if residual_norm < MIN_ORTHOGONAL_RESIDUAL_NORM:
            axis = min(range(len(zero_query)), key=lambda index: abs(zero_query[index]))
            residual = [-zero_query[axis] * value for value in zero_query]
            residual[axis] += 1
        orthogonal = self._unit(residual)
        tail = math.sqrt(1 - self.SCORE * self.SCORE)
        memory_vector = [
            self.SCORE * zero + tail * other
            for zero, other in zip(zero_query, orthogonal, strict=True)
        ]
        self.assertAlmostEqual(
            self._dot(memory_vector, zero_query), self.SCORE, delta=1e-5
        )
        control_score = self._dot(memory_vector, control_query)
        self.assertGreater(
            control_score,
            0.20,
            "The low-threshold control must retrieve the engineered memory row",
        )
        return memory_vector, control_score

    def _insert_memory(
        self, user_id: str, content: str, embedding: list[float]
    ) -> None:
        now = int(time.time())
        self.milvus.client.insert(
            collection_name=self.milvus.collection,
            data=[
                {
                    "id": uuid.uuid4().hex,
                    "user_id": user_id,
                    "project_id": "default",
                    "memory_type": "semantic",
                    "content": content,
                    "source": "e2e",
                    "metadata": json.dumps(
                        {
                            "user_id": user_id,
                            "project_id": "default",
                            "source": "e2e",
                            "importance": 1.0,
                            "access_count": 0,
                            "last_accessed": now,
                        }
                    ),
                    "embedding": embedding,
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "importance": 1.0,
                }
            ],
        )
        self.flush_and_wait(8)
        stored = self.milvus.search_memories(user_id, content)
        self.assertEqual(len(stored), 1, "Engineered memory was not visible in Milvus")

    def test_zero_plugin_threshold_uses_documented_default(self):
        """0.65 memory passes the 0.10 control and fails the 0→0.70 route."""
        self.assertTrue(self.milvus.is_available(), "Milvus is required for this E2E")
        user_id = f"{self.test_user}_threshold_{uuid.uuid4().hex[:8]}"
        question = "Which city do I live in now?"
        zero_route_query = f"{self.MARKER} {question}"
        zero_vector, control_vector = self._embed([zero_route_query, question])
        memory_vector, control_score = self._score_between_defaults(
            zero_vector, control_vector
        )
        canary = f"threshold-canary-{uuid.uuid4().hex[:12]}"
        self._insert_memory(
            user_id, f"Q: I live in {canary}. A: {canary}", memory_vector
        )

        # Probe the zero route first. A positive control response echoes the
        # canary and could otherwise create another memory on an auto-store route.
        zero_route = self.send_memory_request(
            zero_route_query, auto_store=False, user_id=user_id
        )
        self.assertIsNotNone(zero_route, "Zero-threshold route request failed")
        self.assertNotIn(
            canary,
            zero_route.get("_output_text", ""),
            "A 0.65 memory passed the zero plugin threshold; expected the 0.70 default",
        )

        # The low-threshold default route proves retrieval and prompt echo work.
        for attempt in range(3):
            control = self.send_memory_request(
                question, auto_store=False, user_id=user_id, verbose=attempt == 0
            )
            self.assertIsNotNone(control, "Low-threshold control request failed")
            if canary in control.get("_output_text", ""):
                break
            self.flush_and_wait(5)
        else:
            self.fail(
                f"Low-threshold control did not inject a memory at cosine {control_score:.3f}"
            )
