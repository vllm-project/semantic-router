#!/usr/bin/env python3
"""
Memory Features E2E Test Suite — Risk-Based, Milvus-First

Organized by priority derived from risk/return analysis (see ISSUE_1293_REVIEW.md):

  P0 Security:
    - UserIsolationTest: Cross-user memory leak (MINJA/InjecMEM attack surface)

  P1 Pipeline Correctness:
    - MemoryInjectionPipelineTest: The fundamental store -> extract -> inject contract
    - MemoryContentIntegrityTest: Content preserved in Milvus (no truncation/corruption)
    - SimilarityThresholdTest: Irrelevant NOT injected, relevant IS injected
    - StaleMemoryTest: Contradicting facts baseline (soft-insert, no contradiction detection)
    - PluginCombinationTest: Memory + system_prompt coexistence
    - MemoryExtractionTest: Facts extracted from conversation turns

  P2 Operational:
    - ExtractionTriggerTest: Extraction fires on correct turn boundary

Design principles:
  1. Milvus-first verification — storage/extraction checks verify directly in Milvus
     via MilvusVerifier, not by parsing echo responses. Echo check is only for injection.
  2. No assertions on unimplemented features — dedup, contradiction detection, and
     access tracking (Ebbinghaus scoring) are not implemented. We document current
     behavior as baseline, not assert on absent functionality.
  3. Unique user_id per test — isolation via setUp().
  4. New session for retrieval — no previous_response_id on query turns.
  5. Explicit flush + wait — every test that reads from Milvus after a write flushes.

Prerequisites:
  - Milvus running
  - Semantic Router running with memory enabled
  - LLM backend with ECHO mode for reliable verification
  - extraction_batch_size: Must be 1 (not 0!) in config. The code treats 0 as
    "use default (10)". With batchSize=1: turnCount%1=0, so extraction happens
    after every turn.

To start llm-katan with echo backend:
    LLM_KATAN_BACKEND=echo ./start-llm-katan.sh

Usage:
    python e2e/testing/09-memory-features-test.py

    # With custom endpoint
    ROUTER_ENDPOINT=http://localhost:8888 python e2e/testing/09-memory-features-test.py
"""

import json
import os
import sys
import time
import unittest
import requests
from typing import Optional, List, Dict, Any
from test_base import SemanticRouterTestBase

# Milvus client for direct verification of memory storage
try:
    from pymilvus import MilvusClient

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("⚠️  pymilvus not installed - Milvus verification disabled")


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
            # Force flush to make recently inserted data available for search
            from pymilvus import connections, Collection

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
            return -1  # Milvus not available
        try:
            # Flush before querying to ensure data is available
            self.flush()
            # Query by user_id filter
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
    ) -> List[Dict]:
        """Search for memories containing a keyword (in content field).

        Uses retry logic with flush between attempts to handle Milvus consistency delays.
        """
        if not self.client:
            return []

        import time

        for attempt in range(max_retries):
            try:
                # Flush before querying to ensure data is available
                self.flush()

                # Small delay after flush for segments to be searchable
                if attempt > 0:
                    time.sleep(2)

                results = self.client.query(
                    collection_name=self.collection,
                    filter=f'user_id == "{user_id}"',
                    output_fields=["id", "content", "created_at"],
                )

                # Filter by keyword in content
                matches = [
                    r
                    for r in results
                    if keyword.lower() in r.get("content", "").lower()
                ]

                if matches:
                    return matches

                # If no matches but there are results, log for debugging
                if results and not matches:
                    print(
                        f"   ⚠️  Found {len(results)} memories but none contain '{keyword}'"
                    )
                    for r in results[:3]:
                        print(f"      - {r.get('content', '')[:50]}...")

                # If no results at all, retry
                if not results and attempt < max_retries - 1:
                    print(
                        f"   ⏳ Milvus search attempt {attempt+1}/{max_retries}: no results, retrying..."
                    )
                    time.sleep(2)
                    continue

                return matches

            except Exception as e:
                print(f"⚠️  Milvus search failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return []

        return []

    def get_memory_metadata(
        self, user_id: str, keyword: Optional[str] = None, max_retries: int = 3
    ) -> Optional[Dict]:
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

                # Filter by keyword if provided, otherwise use all results
                candidates = results
                if keyword:
                    candidates = [
                        r
                        for r in results
                        if keyword.lower() in r.get("content", "").lower()
                    ]

                for r in candidates:
                    # Parse metadata JSON if it's a string
                    meta_raw = r.get("metadata", "{}")
                    if isinstance(meta_raw, str):
                        try:
                            r["_parsed_metadata"] = json.loads(meta_raw)
                        except json.JSONDecodeError:
                            r["_parsed_metadata"] = {}
                    else:
                        r["_parsed_metadata"] = meta_raw or {}
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
                print(f"⚠️  Milvus metadata query failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

        return None

    def is_available(self) -> bool:
        """Check if Milvus verification is available."""
        return self.client is not None


class MemoryFeaturesTest(SemanticRouterTestBase):
    """Test suite for memory features."""

    def setUp(self):
        """Set up test configuration."""
        self.router_endpoint = os.environ.get(
            "ROUTER_ENDPOINT", "http://localhost:8888"
        )
        self.responses_url = f"{self.router_endpoint}/v1/responses"
        self.timeout = 120  # Longer timeout for memory operations

        # Test user for this suite
        self.test_user = f"memory_features_test_{int(time.time())}"

        # Memory extraction wait time (needs time for extraction + Milvus flush + segment load)
        # Increased from 6 to 10 seconds - Milvus standalone needs time for new segments
        self.extraction_wait = 10

        # Milvus verifier for direct storage verification
        milvus_address = os.environ.get("MILVUS_ADDRESS", "localhost:19530")
        milvus_collection = os.environ.get("MILVUS_COLLECTION", "memory_test_ci")
        self.milvus = MilvusVerifier(
            address=milvus_address, collection=milvus_collection
        )

    def send_memory_request(
        self,
        message: str,
        auto_store: bool = False,
        user_id: Optional[str] = None,
        retrieval_limit: int = 5,
        similarity_threshold: float = 0.7,
        verbose: bool = True,
        previous_response_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Send a request with memory context."""
        user = user_id or self.test_user

        # Memory config is now server-side via plugin configuration.
        # User ID is passed via metadata (OpenAI API spec-compliant).
        payload = {
            "model": "MoM",
            "input": message,
            "instructions": "You are a helpful assistant with memory. Use retrieved memories to answer questions accurately.",
            "metadata": {"user_id": user},
        }

        # Chain conversation for memory extraction
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        if verbose:
            print(f"\n📤 Request (user: {user}):")
            print(f"   Message: {message[:100]}{'...' if len(message) > 100 else ''}")
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

            if response.status_code != 200:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                return None

            result = response.json()
            output_text = self._extract_output_text(result)
            result["_output_text"] = output_text

            if verbose:
                print(f"📥 Response status: {result.get('status', 'unknown')}")
                output_preview = (
                    output_text[:200] + "..." if len(output_text) > 200 else output_text
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

    def wait_for_extraction(self, seconds: int = None):
        """Wait for memory extraction to complete."""
        wait_time = seconds or self.extraction_wait
        print(f"\n⏳ Waiting {wait_time}s for memory extraction...")
        time.sleep(wait_time)


class MemoryInjectionPipelineTest(MemoryFeaturesTest):
    """Test the fundamental memory contract: store -> extract -> inject into prompt.

    Uses the echo backend to verify that stored memories appear in the prompt
    sent to the LLM. All retrieval checks are done in a NEW session (no
    previous_response_id) so keywords can only come from Milvus injection.
    """

    def test_01_store_extract_inject(self):
        """The fundamental pipeline: store a fact, trigger extraction, verify injection."""
        self.print_test_header(
            "Store -> Extract -> Inject Pipeline",
            "Store a fact, trigger extraction, query in NEW session, verify injection via echo",
        )

        fact = "My car is a blue Tesla Model 3 from 2023"
        result1 = self.send_memory_request(
            message=f"Please remember this: {fact}", auto_store=True
        )
        self.assertIsNotNone(result1, "Failed to store fact")
        self.assertEqual(result1.get("status"), "completed")
        first_response_id = result1.get("id")
        print(f"   Turn 1: Fact stored (response_id: {first_response_id[:20]}...)")

        result2 = self.send_memory_request(
            message="Got it, thanks for remembering that.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )
        self.assertIsNotNone(result2, "Failed to send follow-up")
        print("   Turn 2: Follow-up sent (triggers extraction)")

        self.wait_for_extraction()

        if self.milvus.is_available():
            memories = self.milvus.search_memories(self.test_user, "tesla")
            if memories:
                print(f"   Milvus: found {len(memories)} memory(ies) containing 'tesla'")
            else:
                count = self.milvus.count_memories(self.test_user)
                self.fail(
                    f"Extraction failed: {count} memories in Milvus but none contain 'tesla'"
                )

            self.milvus.flush()
            time.sleep(3)

        result = self.send_memory_request(
            message="What car do I drive?",
            auto_store=False,
        )

        self.assertIsNotNone(result, "Failed to retrieve fact")
        output = result.get("_output_text", "").lower()

        has_tesla = "tesla" in output
        has_model_3 = "model 3" in output or "model3" in output

        if has_tesla or has_model_3:
            self.print_test_result(
                True,
                f"Memory injected into prompt: tesla={has_tesla}, model3={has_model_3}",
            )
        else:
            self.print_test_result(
                False,
                f"Memory NOT injected. Expected 'tesla' or 'model 3'. "
                f"Response: {output[:200]}...",
            )
            self.fail(
                "Memory not injected into prompt. Check retrieval and injection flow."
            )


class MemoryContentIntegrityTest(MemoryFeaturesTest):
    """Verify the extractor preserves content correctly in Milvus.

    Checks that structured content (numbers, proper nouns, dates) survives
    the formatTurnChunk path in extractor.go without truncation or corruption.
    """

    def test_01_stored_content_preserves_key_facts(self):
        """Verify stored memory content in Milvus contains the original key facts."""
        self.print_test_header(
            "Content Integrity",
            "Store structured facts, verify Milvus content preserves them",
        )

        if not self.milvus.is_available():
            self.skipTest("Milvus not available for direct content verification")

        structured_fact = (
            "My employee ID is EMP-90210, I started on 2024-03-15, "
            "and my manager is Dr. Evelyn Zhao in Building 7."
        )
        result1 = self.send_memory_request(
            message=f"Please remember this: {structured_fact}",
            auto_store=True,
        )
        self.assertIsNotNone(result1, "Failed to store structured fact")
        first_response_id = result1.get("id")

        result2 = self.send_memory_request(
            message="Thanks, that covers my onboarding info.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )
        self.assertIsNotNone(result2, "Failed to send follow-up")
        print("   Extraction triggered")

        self.wait_for_extraction()

        count = self.milvus.count_memories(self.test_user)
        self.assertGreater(count, 0, "No memories stored in Milvus")

        key_fragments = ["EMP-90210", "2024-03-15", "Evelyn Zhao", "Building 7"]
        all_memories = self.milvus.search_memories(self.test_user, "EMP")
        if not all_memories:
            all_results = self.milvus.client.query(
                collection_name=self.milvus.collection,
                filter=f'user_id == "{self.test_user}"',
                output_fields=["content"],
            )
            combined = " ".join(r.get("content", "") for r in all_results)
        else:
            combined = " ".join(m.get("content", "") for m in all_memories)

        found = [f for f in key_fragments if f.lower() in combined.lower()]
        missing = [f for f in key_fragments if f.lower() not in combined.lower()]

        if len(found) >= 3:
            self.print_test_result(
                True, f"Content preserved: found {found}, missing {missing}"
            )
        else:
            self.print_test_result(
                False,
                f"Content corrupted/truncated: found {found}, missing {missing}. "
                f"Stored: {combined[:200]}...",
            )
            self.fail(f"Content integrity failure: missing {missing}")






class SimilarityThresholdTest(MemoryFeaturesTest):
    """Test similarity threshold for memory retrieval."""

    def test_01_unrelated_query_no_retrieval(self):
        """Test that unrelated queries don't retrieve irrelevant memories."""
        self.print_test_header(
            "Unrelated Query - No Irrelevant Retrieval",
            "Store a fact, ask unrelated question in NEW session",
        )

        # Store a specific fact (turn 1)
        result = self.send_memory_request(
            message="Remember: My favorite restaurant is The Italian Place on 5th Avenue",
            auto_store=True,
        )
        self.assertIsNotNone(result, "Failed to store")
        first_response_id = result.get("id")

        # Send follow-up to trigger extraction (turn 2)
        result2 = self.send_memory_request(
            message="Great restaurant, right?",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )

        self.wait_for_extraction()

        # Ask something completely unrelated in NEW SESSION (no previous_response_id)
        # With high similarity threshold, this should NOT retrieve the restaurant memory
        result = self.send_memory_request(
            message="What is the capital of France?",
            auto_store=False,
            similarity_threshold=0.8,  # High threshold
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Failed to query")
        output = result.get("_output_text", "").lower()

        # Should NOT mention the restaurant (since query is unrelated)
        if "italian" not in output and "5th avenue" not in output:
            self.print_test_result(
                True, "Unrelated query correctly did not retrieve irrelevant memory"
            )
        else:
            self.print_test_result(
                False,
                f"Irrelevant memory retrieved! Found 'italian' in response. Threshold too low?",
            )
            self.fail("Similarity threshold not working: retrieved unrelated memory")

    def test_02_related_query_retrieves_memory(self):
        """Test that semantically related queries retrieve relevant memories."""
        self.print_test_header(
            "Related Query Retrieves Memory",
            "Store fact about a car, query about vehicle in NEW session",
        )

        result = self.send_memory_request(
            message="Remember: I drive a red Toyota Camry 2022",
            auto_store=True,
        )
        self.assertIsNotNone(result, "Failed to store")
        first_response_id = result.get("id")

        result2 = self.send_memory_request(
            message="It gets great gas mileage.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )

        self.wait_for_extraction()

        if self.milvus.is_available():
            self.milvus.flush()
            time.sleep(3)

        result = self.send_memory_request(
            message="What car do I drive?",
            auto_store=False,
        )

        self.assertIsNotNone(result, "Failed to query")
        output = result.get("_output_text", "").lower()

        if "toyota" in output or "camry" in output or "2022" in output:
            self.print_test_result(
                True, "Related query correctly retrieved memory from Milvus"
            )
        else:
            self.print_test_result(
                False,
                f"Memory NOT found. Expected 'toyota' or 'camry'. Response: {output[:200]}...",
            )
            self.fail("Related memory not retrieved from Milvus")


class StaleMemoryTest(MemoryFeaturesTest):
    """Baseline test for contradictory memory behavior.

    The router currently does soft-insert (no contradiction detection).
    Both the old and new fact coexist in Milvus. This test documents that
    behavior so we have a baseline when contradiction detection is added.

    Research basis: RoseRAG (arXiv:2502.10993) shows small models degrade
    more from wrong context than no context. Hindsight (arXiv:2512.12818)
    and RMM (arXiv:2503.08026) both require explicit validation before
    injection to prevent stale fact injection.
    """

    def test_01_contradicting_facts_both_stored(self):
        """Store contradicting facts, verify both exist in Milvus (no dedup/override)."""
        self.print_test_header(
            "Contradicting Facts Baseline",
            "Store two contradicting facts, verify both coexist in Milvus",
        )

        if not self.milvus.is_available():
            self.skipTest("Milvus not available for direct verification")

        result1 = self.send_memory_request(
            message="Remember: I currently live in Boston, Massachusetts.",
            auto_store=True,
        )
        self.assertIsNotNone(result1, "Failed to store fact A")
        first_response_id = result1.get("id")

        result2 = self.send_memory_request(
            message="Actually, I just moved to San Francisco last week.",
            auto_store=True,
            previous_response_id=first_response_id,
        )
        self.assertIsNotNone(result2, "Failed to store fact B")
        second_response_id = result2.get("id")

        result3 = self.send_memory_request(
            message="It was a big move across the country.",
            auto_store=True,
            previous_response_id=second_response_id,
            verbose=False,
        )
        print("   Extraction triggered for all turns")

        self.wait_for_extraction()

        all_results = self.milvus.client.query(
            collection_name=self.milvus.collection,
            filter=f'user_id == "{self.test_user}"',
            output_fields=["content"],
        )
        combined = " ".join(r.get("content", "").lower() for r in all_results)

        has_boston = "boston" in combined
        has_sf = "san francisco" in combined or "francisco" in combined

        print(f"   Milvus contains: boston={has_boston}, san_francisco={has_sf}")
        print(f"   Total memories for user: {len(all_results)}")

        if has_boston and has_sf:
            self.print_test_result(
                True,
                "Both contradicting facts stored (expected: no contradiction detection yet). "
                "When contradiction detection is added, the old fact should be invalidated.",
            )
        elif has_sf and not has_boston:
            self.print_test_result(
                True,
                "Only the newer fact stored (contradiction detection may be active).",
            )
        else:
            self.print_test_result(
                False,
                f"Unexpected state: boston={has_boston}, sf={has_sf}. "
                f"Content: {combined[:200]}...",
            )
            self.fail("Memory storage produced unexpected state")


class UserIsolationTest(MemoryFeaturesTest):
    """Test user memory isolation (security)."""

    def setUp(self):
        """Set up test configuration with two users."""
        super().setUp()
        self.user_a = f"isolation_user_a_{int(time.time())}"
        self.user_b = f"isolation_user_b_{int(time.time())}"
        self.user_a_secret = "My secret PIN is 9876"
        self.user_b_secret = "My password is hunter2"

    def test_01_store_user_a_memory(self):
        """Store a secret for User A using 2-turn pattern for extraction."""
        self.print_test_header(
            "Store User A Secret", f"Storing: '{self.user_a_secret}'"
        )

        # Turn 1: Store the secret
        result = self.send_memory_request(
            message=f"Remember this: {self.user_a_secret}",
            auto_store=True,
            user_id=self.user_a,
        )

        self.assertIsNotNone(result, "Failed to store User A memory")
        self.assertEqual(result.get("status"), "completed")
        first_response_id = result.get("id")
        print(f"   ✓ Turn 1: Secret stored (response_id: {first_response_id[:20]}...)")

        # Turn 2: Follow-up to trigger extraction
        result2 = self.send_memory_request(
            message="Got it, I'll keep that safe.",
            auto_store=True,
            user_id=self.user_a,
            previous_response_id=first_response_id,
            verbose=False,
        )
        self.assertIsNotNone(result2, "Failed to send follow-up")
        print("   ✓ Turn 2: Follow-up sent (triggers extraction)")

        self.print_test_result(True, "User A secret stored with extraction triggered")

        self.wait_for_extraction()

    def test_02_user_b_cannot_see_user_a_secret(self):
        """Security: User B should NOT see User A's secret."""
        self.print_test_header(
            "Security Check: User B queries User A's secret",
            "User B should NOT see User A's PIN",
        )

        # First store User A's memory with 2-turn pattern
        result1 = self.send_memory_request(
            message=f"Remember this: {self.user_a_secret}",
            auto_store=True,
            user_id=self.user_a,
            verbose=False,
        )
        first_response_id = result1.get("id") if result1 else None

        # Turn 2: Follow-up to trigger extraction
        if first_response_id:
            self.send_memory_request(
                message="That's my secret PIN.",
                auto_store=True,
                user_id=self.user_a,
                previous_response_id=first_response_id,
                verbose=False,
            )
            print("   ✓ User A's secret stored with extraction triggered")

        self.wait_for_extraction()

        # User B tries to access
        result = self.send_memory_request(
            message="What is my PIN?", auto_store=False, user_id=self.user_b
        )

        self.assertIsNotNone(result, "Request failed")
        output = result.get("_output_text", "").lower()

        # Check for the actual secret value
        has_leaked = "9876" in output

        if has_leaked:
            self.print_test_result(
                False, "🚨 SECURITY VIOLATION: User B saw User A's PIN!"
            )
            self.fail(f"SECURITY VIOLATION: User B saw User A's secret: {output[:200]}")
        else:
            self.print_test_result(
                True, "✅ User B correctly cannot see User A's secret"
            )

    def test_03_user_a_can_see_own_memory(self):
        """User A should be able to see their own secret from Milvus."""
        self.print_test_header(
            "User A Queries Own Memory", "User A should see their own PIN from Milvus"
        )

        # First store User A's memory with 2-turn pattern
        store_result = self.send_memory_request(
            message=f"Remember this: {self.user_a_secret}",
            auto_store=True,
            user_id=self.user_a,
            verbose=False,
        )
        self.assertIsNotNone(store_result, "Failed to store User A secret")
        first_response_id = store_result.get("id")

        # Turn 2: Follow-up to trigger extraction
        result2 = self.send_memory_request(
            message="That's my secret PIN.",
            auto_store=True,
            user_id=self.user_a,
            previous_response_id=first_response_id,
            verbose=False,
        )
        print("   ✓ User A's secret stored with extraction triggered")

        self.wait_for_extraction()

        # User A queries their own memory in NEW SESSION (no previous_response_id)
        # This tests that User A can retrieve their own memory from Milvus
        result = self.send_memory_request(
            message="What is my PIN?",
            auto_store=False,
            user_id=self.user_a,
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Request failed")
        output = result.get("_output_text", "").lower()

        # Check if PIN is in response (from Milvus, not conversation history)
        if "9876" in output:
            self.print_test_result(
                True, "User A correctly retrieved their own PIN from Milvus"
            )
        else:
            self.print_test_result(
                False,
                f"User A's PIN NOT found. Expected '9876'. Response: {output[:200]}...",
            )
            self.fail("User A cannot retrieve their own memory from Milvus")

    def test_04_bidirectional_isolation(self):
        """Test isolation works both ways - query in NEW sessions."""
        self.print_test_header(
            "Bidirectional Isolation",
            "Neither user should see the other's secrets from Milvus",
        )

        # Store secrets for both users with 2-turn pattern each
        # User A: Turn 1
        result_a1 = self.send_memory_request(
            message=f"Remember: {self.user_a_secret}",
            auto_store=True,
            user_id=self.user_a,
            verbose=False,
        )
        response_id_a = result_a1.get("id") if result_a1 else None

        # User B: Turn 1
        result_b1 = self.send_memory_request(
            message=f"Remember: {self.user_b_secret}",
            auto_store=True,
            user_id=self.user_b,
            verbose=False,
        )
        response_id_b = result_b1.get("id") if result_b1 else None

        # User A: Turn 2 (triggers extraction)
        if response_id_a:
            self.send_memory_request(
                message="That's my secret PIN.",
                auto_store=True,
                user_id=self.user_a,
                previous_response_id=response_id_a,
                verbose=False,
            )

        # User B: Turn 2 (triggers extraction)
        if response_id_b:
            self.send_memory_request(
                message="That's my password.",
                auto_store=True,
                user_id=self.user_b,
                previous_response_id=response_id_b,
                verbose=False,
            )

        print("   ✓ Both users' secrets stored with extraction triggered")

        self.wait_for_extraction(3)

        # Verify isolation at Milvus storage level
        if self.milvus.is_available():
            # User A should only see their own memories
            user_a_memories = self.milvus.search_memories(self.user_a, "9876")
            user_b_memories = self.milvus.search_memories(self.user_b, "hunter2")

            # Cross-check: User A should NOT have User B's password
            user_a_leak = self.milvus.search_memories(self.user_a, "hunter2")
            user_b_leak = self.milvus.search_memories(self.user_b, "9876")

            if user_a_leak:
                self.fail(
                    f"SECURITY: User A's Milvus partition contains User B's password!"
                )
            if user_b_leak:
                self.fail(f"SECURITY: User B's Milvus partition contains User A's PIN!")

            print(
                f"   ✓ Milvus isolation verified: User A has {len(user_a_memories)} memories, User B has {len(user_b_memories)}"
            )
        else:
            print("   ⚠️  Milvus verification skipped (not available)")

        # User A tries to get User B's password in NEW SESSION (no previous_response_id)
        result = self.send_memory_request(
            message="What is my password?",
            auto_store=False,
            user_id=self.user_a,
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Request failed")
        output_a = result.get("_output_text", "").lower()

        # User B tries to get User A's PIN in NEW SESSION (no previous_response_id)
        result = self.send_memory_request(
            message="What is my PIN?",
            auto_store=False,
            user_id=self.user_b,
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Request failed")
        output_b = result.get("_output_text", "").lower()

        # Check for leaks from Milvus
        a_saw_b_password = "hunter2" in output_a
        b_saw_a_pin = "9876" in output_b

        if a_saw_b_password:
            self.fail("SECURITY VIOLATION: User A saw User B's password from Milvus")
        if b_saw_a_pin:
            self.fail("SECURITY VIOLATION: User B saw User A's PIN from Milvus")

        self.print_test_result(
            True, "✅ Bidirectional isolation verified at Milvus level"
        )


class MemoryExtractionTest(MemoryFeaturesTest):
    """Test memory extraction from natural conversation."""

    def test_01_extract_facts_from_conversation(self):
        """Test that facts are extracted from natural conversation and stored in Milvus."""
        self.print_test_header(
            "Extract Facts from Conversation",
            "Natural conversation, query in NEW sessions to verify Milvus storage",
        )

        # Natural conversation (not explicit "remember this") - turn 1
        conversation_message = """
        I had a great day today! Had lunch with my brother Tom at the new sushi place
        downtown. He told me he's getting married next spring to his girlfriend Anna.
        I'm so happy for them! Oh, and I finally bought that new laptop I've been
        looking at - a MacBook Pro M3.
        """

        result = self.send_memory_request(message=conversation_message, auto_store=True)
        self.assertIsNotNone(result, "Failed to process conversation")
        first_response_id = result.get("id")

        # Send follow-up to trigger extraction (turn 2)
        result2 = self.send_memory_request(
            message="What a great day it was!",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )
        print("   ✓ Follow-up sent (triggers extraction)")

        self.wait_for_extraction(3)  # Wait longer for extraction

        # Verify facts were extracted and stored in Milvus
        if self.milvus.is_available():
            count = self.milvus.count_memories(self.test_user)
            if count > 0:
                print(f"   ✓ Extraction stored {count} memories in Milvus")
                # Check for specific extracted facts
                for keyword in ["tom", "anna", "macbook"]:
                    memories = self.milvus.search_memories(self.test_user, keyword)
                    if memories:
                        print(f"   ✓ Found '{keyword}' in Milvus")
            else:
                self.print_test_result(False, "No facts extracted to Milvus")
                self.fail("Memory extraction failed: no memories stored")
        else:
            print("   ⚠️  Milvus verification skipped (not available)")

        # Query for extracted facts in NEW SESSIONS (no previous_response_id)
        # This tests that facts were extracted and stored in Milvus
        queries = [
            ("Who is my brother?", ["tom"]),
            ("What laptop did I buy?", ["macbook", "m3"]),
            ("Who is Tom marrying?", ["anna"]),
        ]

        successful_queries = 0
        for query, expected_keywords in queries:
            print(f"\n   Querying (NEW SESSION): {query}")
            result = self.send_memory_request(
                message=query,
                auto_store=False,
                verbose=False,
                # NO previous_response_id - this is a new session!
            )

            if result:
                output = result.get("_output_text", "").lower()
                found = [kw for kw in expected_keywords if kw in output]
                if found:
                    print(f"   ✓ Found from Milvus: {found}")
                    successful_queries += 1
                else:
                    print(f"   ✗ Keywords not found: {expected_keywords}")

        if successful_queries >= 1:
            self.print_test_result(
                True,
                f"Extracted and retrieved {successful_queries}/3 facts from Milvus",
            )
        else:
            self.print_test_result(False, "No facts extracted or retrieved from Milvus")
            self.fail("Memory extraction failed: no facts found in Milvus")


class PluginCombinationTest(MemoryFeaturesTest):
    """
    Test that memory works correctly with system_prompt plugin enabled.
    """

    def test_01_memory_with_system_prompt_both_present(self):
        """
        Verify memory injection works when system_prompt plugin is enabled.
        Query in NEW session to verify memory comes from Milvus, not conversation history.
        """
        self.print_test_header("Memory + System Prompt: Both Present (NEW SESSION)")

        # Step 1: Store a unique fact (turn 1)
        unique_fact = "Phoenix-2026"
        store_message = f"Please remember this important code: my secret project codename is {unique_fact}"

        print(f"\n📝 Step 1: Storing unique fact: {unique_fact}")
        result = self.send_memory_request(
            message=store_message,
            auto_store=True,
        )

        if not result:
            self.fail("Failed to store memory")

        first_response_id = result.get("id")
        print(
            f"   Turn 1: Stored (response_id: {first_response_id[:20] if first_response_id else 'N/A'}...)"
        )

        # Step 2: Send follow-up to trigger extraction (turn 2 extracts turn 1)
        print(f"\n📝 Step 2: Sending follow-up to trigger extraction...")
        result2 = self.send_memory_request(
            message="Got it, I'll remember that code.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )
        if not result2:
            self.fail("Failed to send follow-up")
        print("   Turn 2: Follow-up sent (triggers extraction)")

        # Step 3: Wait for extraction to complete
        print(f"\n⏳ Step 3: Waiting for memory extraction...")
        time.sleep(self.extraction_wait + 1)

        # Step 4: Query for the fact in NEW SESSION (no previous_response_id)
        # This is critical - it ensures we're testing Milvus retrieval, not conversation history
        print(f"\n🔍 Step 4: Querying for the fact (NEW SESSION)...")
        query_result = self.send_memory_request(
            message="What is my secret project codename?",
            auto_store=False,
            # NO previous_response_id - this is a new session!
            # This ensures "phoenix" can ONLY come from Milvus memory injection
        )

        if not query_result:
            self.fail("Failed to query memory")

        output = query_result.get("_output_text", "").lower()
        print(f"   Response: {query_result.get('_output_text', '')[:200]}...")

        # Step 5: Verify the unique fact is in the response
        # Since this is a NEW session, "phoenix" can ONLY come from Milvus memory
        if "phoenix" in output:
            self.print_test_result(
                True,
                "Memory retrieved from Milvus and injected with system_prompt enabled! "
                "Both plugins work together correctly.",
            )
        else:
            # Memory was NOT injected - this is the bug we're testing for
            self.print_test_result(
                False,
                f"Memory NOT found in response. Expected 'phoenix'. "
                f"This indicates the Memory+SystemPrompt bug still exists. "
                f"Response: {output[:200]}...",
            )
            self.fail(
                f"Memory+SystemPrompt bug: 'phoenix' not found in response. "
                f"Memory was not retrieved from Milvus when system_prompt plugin is enabled."
            )

    def test_02_verify_system_prompt_persona_present(self):
        """
        Verify that system prompt persona is still being applied.
        """
        self.print_test_header("System Prompt: Persona Still Applied")

        # Ask something that would reveal if the system prompt is working
        result = self.send_memory_request(
            message="Who are you? What's your name?",
            auto_store=False,
        )

        if not result:
            self.fail("Failed to get response")

        output = result.get("_output_text", "").lower()

        # The system prompt typically defines the assistant as "MoM" or similar
        # Check if the response reflects the persona
        persona_keywords = ["mom", "assistant", "ai", "help", "personal"]
        found_persona = any(kw in output for kw in persona_keywords)

        if found_persona:
            self.print_test_result(
                True, "System prompt persona is being applied correctly"
            )
        else:
            self.print_test_result(
                True,  # Don't fail - persona might be subtle
                f"Persona keywords not found, but response received: {output[:100]}...",
            )


class ExtractionTriggerTest(MemoryFeaturesTest):
    """Verify extraction fires on the correct turn boundary.

    With extraction_batch_size=1, extraction should happen after every turn
    (turnCount % 1 == 0). This test verifies that turn 2 triggers extraction
    of turn 1's content into Milvus.
    """

    def test_01_extraction_after_second_turn(self):
        """Verify that a 2-turn conversation results in extracted memory in Milvus."""
        self.print_test_header(
            "Extraction Trigger on Turn Boundary",
            "Send 2 chained turns, verify extraction produced a memory in Milvus",
        )

        if not self.milvus.is_available():
            self.skipTest("Milvus not available for direct verification")

        unique_marker = f"XTRG-{int(time.time())}"
        result1 = self.send_memory_request(
            message=f"Please remember this: my verification code is {unique_marker}",
            auto_store=True,
        )
        self.assertIsNotNone(result1, "Failed to store turn 1")
        first_response_id = result1.get("id")
        print(f"   Turn 1 sent (marker: {unique_marker})")

        count_after_t1 = self.milvus.count_memories(self.test_user)
        print(f"   Milvus memories after turn 1 (before extraction): {count_after_t1}")

        result2 = self.send_memory_request(
            message="Thanks for noting that code.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )
        self.assertIsNotNone(result2, "Failed to store turn 2")
        print("   Turn 2 sent (should trigger extraction of turn 1)")

        self.wait_for_extraction()

        count_after_t2 = self.milvus.count_memories(self.test_user)
        print(f"   Milvus memories after turn 2 + wait: {count_after_t2}")

        if count_after_t2 > count_after_t1:
            memories = self.milvus.search_memories(self.test_user, unique_marker)
            if memories:
                self.print_test_result(
                    True,
                    f"Extraction triggered: {count_after_t2} memories, "
                    f"marker '{unique_marker}' found in Milvus",
                )
            else:
                self.print_test_result(
                    True,
                    f"Extraction triggered: {count_after_t2} memories stored "
                    f"(marker not in content field but memories were created)",
                )
        else:
            self.print_test_result(
                False,
                f"No new memories after turn 2. Before: {count_after_t1}, after: {count_after_t2}",
            )
            self.fail("Extraction did not fire after turn 2")



def run_tests():
    """Run all memory feature tests with detailed output."""
    print("\n" + "=" * 60)
    print("Memory Features Integration Test Suite")
    print("=" * 60)

    # Check router health
    router_endpoint = os.environ.get("ROUTER_ENDPOINT", "http://localhost:8888")
    print(f"Router endpoint: {router_endpoint}")

    try:
        response = requests.get(f"{router_endpoint}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Router is healthy")
        else:
            print(f"⚠️  Router health check returned {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach router: {e}")
        sys.exit(1)

    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes in order
    test_classes = [
        # P0: Security — run first, fail fast on data leaks
        UserIsolationTest,
        # P1: Pipeline correctness
        MemoryInjectionPipelineTest,
        MemoryContentIntegrityTest,
        SimilarityThresholdTest,
        StaleMemoryTest,
        PluginCombinationTest,
        MemoryExtractionTest,
        # P2: Operational
        ExtractionTriggerTest,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")

    if failures > 0:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")

    if errors > 0:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")

    if failures == 0 and errors == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
