"""Milvus verification and base test class for memory feature E2E tests.

Provides MilvusVerifier for direct storage checks and MemoryFeaturesTest
as the base class with shared helpers (send_memory_request, wait_for_storage,
query_with_retry, etc.).
"""

import json
import os
import time

import requests
from test_base import SemanticRouterTestBase

try:
    from pymilvus import Collection, MilvusClient, connections

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("⚠️  pymilvus not installed - Milvus verification disabled")

HTTP_OK = 200
PREVIEW_LENGTH = 200
MSG_PREVIEW_LENGTH = 100
MIN_CONTENT_MATCHES = 3


class MilvusVerifier:
    """Helper to verify memory storage in Milvus directly."""

    def __init__(
        self, address: str = "localhost:19530", collection: str = "memory_test_ci"
    ):
        self.address = address
        self.collection = collection
        self.client = None
        if MILVUS_AVAILABLE:
            try:
                self.client = MilvusClient(uri=f"http://{address}")
            except Exception as e:
                print(f"⚠️  Failed to connect to Milvus: {e}")

    def flush(self) -> bool:
        """Flush the collection to make data searchable."""
        if not self.client:
            return False
        try:
            connections.connect(uri=f"http://{self.address}")
            collection = Collection(self.collection)
            collection.flush()
            return True
        except Exception as e:
            print(f"⚠️  Milvus flush failed: {e}")
            return False

    def count_memories(self, user_id: str) -> int:
        """Count memories stored for a user."""
        if not self.client:
            return -1
        try:
            self.flush()
            results = self.client.query(
                collection_name=self.collection,
                filter=f'user_id == "{user_id}"',
                output_fields=["id"],
            )
            return len(results)
        except Exception as e:
            print(f"⚠️  Milvus query failed: {e}")
            return -1

    def search_memories(
        self, user_id: str, keyword: str, max_retries: int = 3
    ) -> list[dict]:
        """Search for memories containing a keyword (in content field).

        Uses retry logic with flush between attempts to handle Milvus
        consistency delays.
        """
        if not self.client:
            return []

        for attempt in range(max_retries):
            try:
                self.flush()

                if attempt > 0:
                    time.sleep(2)

                results = self.client.query(
                    collection_name=self.collection,
                    filter=f'user_id == "{user_id}"',
                    output_fields=["id", "content", "created_at"],
                )

                matches = [
                    r
                    for r in results
                    if keyword.lower() in r.get("content", "").lower()
                ]

                if matches:
                    return matches

                if results and not matches:
                    print(
                        f"   ⚠️  Found {len(results)} memories but none contain '{keyword}'"
                    )
                    for r in results[:3]:
                        print(f"      - {r.get('content', '')[:50]}...")

                if not results and attempt < max_retries - 1:
                    print(
                        f"   ⏳ Milvus search attempt {attempt + 1}/{max_retries}: "
                        f"no results, retrying..."
                    )
                    time.sleep(2)
                    continue

                return matches

            except Exception as e:
                print(f"⚠️  Milvus search failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return []

        return []

    @staticmethod
    def _parse_metadata_field(record: dict) -> dict:
        """Parse the metadata JSON field from a Milvus record."""
        meta_raw = record.get("metadata", "{}")
        if not isinstance(meta_raw, str):
            return meta_raw or {}
        try:
            return json.loads(meta_raw)
        except json.JSONDecodeError:
            return {}

    def get_memory_metadata(
        self, user_id: str, keyword: str | None = None, max_retries: int = 3
    ) -> dict | None:
        """Get a memory's metadata for a user.

        If keyword is provided, filters by keyword in content.
        If keyword is None, returns the first (most recent) memory for the user.
        Each test uses a unique user_id, so no keyword is needed for isolation.
        """
        if not self.client:
            return None

        for attempt in range(max_retries):
            try:
                self.flush()
                if attempt > 0:
                    time.sleep(2)

                results = self.client.query(
                    collection_name=self.collection,
                    filter=f'user_id == "{user_id}"',
                    output_fields=[
                        "id",
                        "content",
                        "metadata",
                        "access_count",
                        "created_at",
                        "updated_at",
                    ],
                )

                candidates = results
                if keyword:
                    candidates = [
                        r
                        for r in results
                        if keyword.lower() in r.get("content", "").lower()
                    ]

                for r in candidates:
                    r["_parsed_metadata"] = self._parse_metadata_field(r)
                    return r

                if not results and attempt < max_retries - 1:
                    print(
                        f"   ⏳ Milvus metadata attempt {attempt + 1}/{max_retries}: "
                        f"no results for user, retrying..."
                    )
                    time.sleep(2)
                    continue

                if results and not candidates:
                    print(
                        f"   ⚠️  Found {len(results)} memories for user but "
                        f"keyword '{keyword}' not in content. "
                        f"Stored: {results[0].get('content', '')[:80]}..."
                    )
                return None

            except Exception as e:
                print(f"⚠️  Milvus metadata query failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

        return None

    def is_available(self) -> bool:
        """Check if Milvus verification is available."""
        return self.client is not None


class MemoryFeaturesTest(SemanticRouterTestBase):
    """Base test class for memory features with shared helpers."""

    def setUp(self):
        """Set up test configuration."""
        self.router_endpoint = os.environ.get(
            "ROUTER_ENDPOINT", "http://localhost:8888"
        )
        self.responses_url = f"{self.router_endpoint}/v1/responses"
        self.timeout = 120

        self.test_user = f"memory_features_test_{int(time.time())}"

        # Milvus standalone needs time for new segments to become searchable
        self.storage_wait = 10

        milvus_address = os.environ.get("MILVUS_ADDRESS", "localhost:19530")
        milvus_collection = os.environ.get("MILVUS_COLLECTION", "memory_test_ci")
        self.milvus = MilvusVerifier(
            address=milvus_address, collection=milvus_collection
        )

    def send_memory_request(
        self,
        message: str,
        auto_store: bool = False,
        user_id: str | None = None,
        retrieval_limit: int = 5,
        similarity_threshold: float = 0.7,
        verbose: bool = True,
        previous_response_id: str | None = None,
    ) -> dict | None:
        """Send a request with memory context."""
        user = user_id or self.test_user

        payload = {
            "model": "MoM",
            "input": message,
            "instructions": (
                "You are a helpful assistant with memory. "
                "Use retrieved memories to answer questions accurately."
            ),
            "metadata": {"user_id": user},
        }

        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        if verbose:
            preview = message[:MSG_PREVIEW_LENGTH]
            ellipsis = "..." if len(message) > MSG_PREVIEW_LENGTH else ""
            print(f"\n📤 Request (user: {user}):")
            print(f"   Message: {preview}{ellipsis}")
            print(f"   Auto-store: {auto_store}, Retrieval limit: {retrieval_limit}")

        try:
            response = requests.post(
                self.responses_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-authz-user-id": user,
                },
                timeout=self.timeout,
            )

            if response.status_code != HTTP_OK:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                return None

            result = response.json()
            output_text = self._extract_output_text(result)
            result["_output_text"] = output_text

            if verbose:
                print(f"📥 Response status: {result.get('status', 'unknown')}")
                output_preview = (
                    output_text[:PREVIEW_LENGTH] + "..."
                    if len(output_text) > PREVIEW_LENGTH
                    else output_text
                )
                print(f"   Output: {output_preview}")

            return result

        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return None

    def _extract_output_text(self, response: dict) -> str:
        """Extract text from Response API output."""
        output_text = response.get("output_text", "")
        if output_text:
            return output_text

        output = response.get("output", [])
        if output and isinstance(output, list):
            first_output = output[0]
            content = first_output.get("content", [])
            if content and isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        return part["text"]
            if "text" in first_output:
                return first_output["text"]

        return ""

    def wait_for_storage(self, seconds: int | None = None):
        """Wait for memory storage (embedding + Milvus flush) to complete."""
        wait_time = seconds or self.storage_wait
        print(f"\n⏳ Waiting {wait_time}s for memory storage...")
        time.sleep(wait_time)

    def flush_and_wait(self, wait_seconds: int = 5):
        """Flush Milvus and wait for vectors to become searchable.

        After flush, sealed segments need to be indexed before vector search
        finds them. CI environments with slower I/O need more time than local.
        """
        if self.milvus.is_available():
            self.milvus.flush()
            time.sleep(wait_seconds)

    def query_with_retry(
        self, message: str, keywords: list, max_attempts: int = 3, wait_between: int = 5
    ) -> tuple:
        """Query the router and check for keywords, retrying on miss.

        Returns (output_text, found_keywords). Retries with flush between
        attempts to handle Milvus segment visibility delays in CI.
        """
        output = ""
        for attempt in range(max_attempts):
            result = self.send_memory_request(
                message=message, auto_store=False, verbose=(attempt == 0)
            )
            if not result:
                continue
            output = result.get("_output_text", "").lower()
            found = [kw for kw in keywords if kw in output]
            if found:
                return output, found
            if attempt < max_attempts - 1:
                print(
                    f"   ⏳ Retry {attempt + 1}/{max_attempts}: "
                    f"keywords {keywords} not in response, flushing..."
                )
                self.flush_and_wait(wait_between)
        return output, []
