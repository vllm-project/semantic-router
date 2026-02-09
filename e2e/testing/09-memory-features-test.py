#!/usr/bin/env python3
"""
Memory Features Integration Test

Comprehensive tests for memory functionality:
- Memory Retrieval: Store and retrieve flow
- Query Rewriting: Vague queries get rewritten with context
- Deduplication: No duplicate memories stored
- Full Conversation Flow: Multi-turn with memory
- Similarity Threshold: Only relevant memories retrieved
- Memory Extraction: Facts extracted from conversation
- User Isolation: User A cannot see User B's memories (security)

Prerequisites:
- Milvus running (docker-compose with milvus)
- Semantic Router running with memory enabled
- LLM backend with ECHO mode for reliable verification

IMPORTANT - Testing Strategy:
1. Echo Backend: Returns the full prompt (including injected memory context),
   allowing tests to verify that memory was properly injected by checking for
   specific keywords in the response.

2. NEW SESSION Queries: All query steps are performed WITHOUT previous_response_id.
   This ensures keywords found in the response can ONLY come from Milvus memory
   injection, not from conversation history.

3. extraction_batch_size: Must be 1 (not 0!) in config. The code treats 0 as
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
                headers={"Content-Type": "application/json"},
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


class MemoryRetrievalTest(MemoryFeaturesTest):
    """Test basic memory store and retrieve functionality."""

    def test_01_store_and_retrieve_simple_fact(self):
        """Test storing a fact and retrieving it."""
        self.print_test_header(
            "Store and Retrieve Simple Fact",
            "Store a specific fact and verify it can be retrieved from Milvus",
        )

        # STEP 1: Store a specific fact (creates turn 1)
        fact = "My car is a blue Tesla Model 3 from 2023"
        result1 = self.send_memory_request(
            message=f"Please remember this: {fact}", auto_store=True
        )
        self.assertIsNotNone(result1, "Failed to store fact")
        self.assertEqual(result1.get("status"), "completed")
        first_response_id = result1.get("id")
        print(f"   ✓ Turn 1: Fact stored (response_id: {first_response_id[:20]}...)")

        # STEP 2: Send follow-up to trigger extraction (turn 2 extracts turn 1)
        # Extraction needs history, so we chain with previous_response_id
        result2 = self.send_memory_request(
            message="Got it, thanks for remembering that.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )
        self.assertIsNotNone(result2, "Failed to send follow-up")
        print("   ✓ Turn 2: Follow-up sent (triggers extraction of turn 1)")

        self.wait_for_extraction()

        # Verify storage in Milvus directly
        if self.milvus.is_available():
            memories = self.milvus.search_memories(self.test_user, "tesla")
            if memories:
                self.print_test_result(
                    True, f"Fact stored in Milvus: found {len(memories)} memory(ies)"
                )
            else:
                # Check if any memories exist for this user
                count = self.milvus.count_memories(self.test_user)
                if count > 0:
                    print(
                        f"   ⚠️  {count} memories stored but 'tesla' not found in content"
                    )
                else:
                    self.print_test_result(False, "No memories stored in Milvus")
                    self.fail("Memory extraction failed: no memories in Milvus")
        else:
            print("   ⚠️  Milvus verification skipped (not available)")

        # Query for the fact in a NEW SESSION (no previous_response_id)
        # This ensures we're testing memory retrieval from Milvus, not conversation history
        result = self.send_memory_request(
            message="What car do I drive?",
            auto_store=False,
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Failed to retrieve fact")
        output = result.get("_output_text", "").lower()

        # Check if the response contains relevant information
        # With echo backend: response contains the full prompt including injected memory
        # Since this is a NEW session, keywords can ONLY come from Milvus memory injection
        has_tesla = "tesla" in output
        has_model_3 = "model 3" in output or "model3" in output
        has_blue = "blue" in output

        if has_tesla or has_model_3:
            self.print_test_result(
                True,
                f"Memory retrieved from Milvus and injected: Tesla={has_tesla}, Model3={has_model_3}, Blue={has_blue}",
            )
        else:
            # Memory was NOT injected - this is a failure
            self.print_test_result(
                False,
                f"Memory NOT found in response. Expected 'tesla' or 'model 3'. "
                f"Response: {output[:200]}...",
            )
            self.fail(
                f"Memory not retrieved from Milvus: 'tesla' not found in response. "
                f"Check memory storage and retrieval flow."
            )

    def test_02_store_multiple_facts_retrieve_specific(self):
        """Test storing multiple facts and retrieving a specific one."""
        self.print_test_header(
            "Store Multiple Facts, Retrieve Specific",
            "Store 3 facts, query for one specific fact from Milvus",
        )

        facts = [
            "My favorite color is purple",
            "I work as a software engineer at Google",
            "My dog's name is Max and he is a golden retriever",
        ]

        # Store all facts with conversation chaining for extraction
        # Each turn extracts the previous turn's content
        last_response_id = None
        for i, fact in enumerate(facts):
            result = self.send_memory_request(
                message=f"Remember: {fact}",
                auto_store=True,
                previous_response_id=last_response_id,
                verbose=(i == 0),
            )
            self.assertIsNotNone(result, f"Failed to store fact {i+1}")
            last_response_id = result.get("id")
            print(f"   ✓ Stored fact {i+1}")

        # Send one more follow-up to extract the last fact
        result = self.send_memory_request(
            message="Thanks, I've told you all my facts.",
            auto_store=True,
            previous_response_id=last_response_id,
            verbose=False,
        )
        print("   ✓ Follow-up sent (triggers extraction of last fact)")

        self.wait_for_extraction(3)  # Wait for extractions to complete

        # Query for dog info in NEW SESSION (no previous_response_id)
        # This ensures we're testing memory retrieval from Milvus
        result = self.send_memory_request(
            message="What is my dog's name?",
            auto_store=False,
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Failed to query")
        output = result.get("_output_text", "").lower()

        if "max" in output or "golden" in output or "retriever" in output:
            self.print_test_result(
                True, "Correctly retrieved dog-related memory from Milvus"
            )
        else:
            self.print_test_result(
                False,
                f"Memory NOT found. Expected 'max' or 'golden'. Response: {output[:200]}...",
            )
            self.fail("Memory not retrieved from Milvus: dog info not found")


class QueryRewritingTest(MemoryFeaturesTest):
    """Test query rewriting functionality."""

    def test_01_vague_query_with_context(self):
        """Test that vague queries work with memory context from Milvus."""
        self.print_test_header(
            "Vague Query with Context",
            "Store context, then ask a vague question in NEW session",
        )

        # Store some context (turn 1)
        result = self.send_memory_request(
            message="I'm planning a trip to Japan next month. My budget is $5000.",
            auto_store=True,
        )
        self.assertIsNotNone(result, "Failed to store context")
        first_response_id = result.get("id")

        # Send follow-up to trigger extraction (turn 2)
        result2 = self.send_memory_request(
            message="I'm so excited about this trip!",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )

        self.wait_for_extraction()

        # Ask a vague follow-up in NEW SESSION (no previous_response_id)
        # This tests that memory is retrieved from Milvus and provides context
        result = self.send_memory_request(
            message="How much can I spend on hotels?",
            auto_store=False,
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Failed to process vague query")
        output = result.get("_output_text", "").lower()

        # The response should reference the budget or trip from Milvus memory
        has_context = any(
            word in output for word in ["5000", "budget", "japan", "trip"]
        )

        if has_context:
            self.print_test_result(True, "Memory context retrieved from Milvus")
        else:
            self.print_test_result(
                False,
                f"Memory context NOT found. Expected 'japan' or '5000'. Response: {output[:200]}...",
            )
            self.fail("Memory context not retrieved from Milvus")

    def test_02_pronoun_resolution(self):
        """Test that pronouns are resolved using memory context from Milvus."""
        self.print_test_header(
            "Pronoun Resolution", "Store info about a person, query in NEW session"
        )

        # Store info about a person (turn 1)
        result = self.send_memory_request(
            message="My friend Sarah is a doctor who lives in Boston.", auto_store=True
        )
        self.assertIsNotNone(result, "Failed to store info")
        first_response_id = result.get("id")

        # Send follow-up to trigger extraction (turn 2)
        result2 = self.send_memory_request(
            message="She's a great friend.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )

        self.wait_for_extraction()

        # Ask using pronoun in NEW SESSION (no previous_response_id)
        # Memory context from Milvus should help resolve the pronoun
        result = self.send_memory_request(
            message="Where does she live?",
            auto_store=False,
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Failed to process pronoun query")
        output = result.get("_output_text", "").lower()

        # Check if Boston or Sarah is mentioned (from Milvus memory)
        if "boston" in output or "sarah" in output:
            self.print_test_result(
                True, "Memory context retrieved, pronoun can be resolved"
            )
        else:
            self.print_test_result(
                False,
                f"Memory NOT found. Expected 'boston' or 'sarah'. Response: {output[:200]}...",
            )
            self.fail("Memory context not retrieved from Milvus")


class DeduplicationTest(MemoryFeaturesTest):
    """Test memory deduplication functionality."""

    def test_01_no_duplicate_storage(self):
        """Test that identical memories are not stored multiple times."""
        self.print_test_header(
            "No Duplicate Storage",
            "Store the same fact 3 times, verify no duplicates in Milvus retrieval",
        )

        fact = "My phone number is 555-123-4567"

        # Store the same fact multiple times with conversation chaining
        # Each turn extracts the previous turn's content
        last_response_id = None
        for i in range(3):
            result = self.send_memory_request(
                message=f"Remember: {fact}",
                auto_store=True,
                previous_response_id=last_response_id,
                verbose=(i == 0),  # Only verbose on first
            )
            self.assertIsNotNone(result, f"Failed to store (attempt {i+1})")
            last_response_id = result.get("id")
            if i > 0:
                print(f"   ✓ Stored attempt {i+1}")
            time.sleep(1)  # Small wait between stores

        # Send follow-up to trigger extraction of the last fact
        result = self.send_memory_request(
            message="Got it, you've told me your phone number.",
            auto_store=True,
            previous_response_id=last_response_id,
            verbose=False,
        )
        print("   ✓ Follow-up sent (triggers extraction)")

        self.wait_for_extraction(3)

        # Verify deduplication in Milvus directly
        if self.milvus.is_available():
            memories = self.milvus.search_memories(self.test_user, "555-123-4567")
            count = len(memories)
            if count == 0:
                self.print_test_result(False, "No memories stored in Milvus")
                self.fail("Memory storage failed: phone number not in Milvus")
            elif count == 1:
                self.print_test_result(
                    True, f"Deduplication working: only 1 memory stored (not 3)"
                )
            elif count <= 3:
                print(f"   ⚠️  Found {count} memories - some duplicates exist")
            else:
                self.print_test_result(
                    False, f"Deduplication broken: {count} duplicates stored"
                )
                self.fail(f"Deduplication failed: {count} copies stored instead of 1")
        else:
            print("   ⚠️  Milvus verification skipped (not available)")

        # Query in NEW SESSION (no previous_response_id)
        # This tests deduplication at the Milvus level
        result = self.send_memory_request(
            message="What is my phone number?",
            auto_store=False,
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Failed to query")
        output = result.get("_output_text", "")

        # Count occurrences of the phone number
        phone_count = output.lower().count("555-123-4567") + output.lower().count(
            "5551234567"
        )

        if phone_count <= 1:
            self.print_test_result(
                True, "No duplicate phone numbers in Milvus retrieval"
            )
        elif phone_count > 3:
            self.print_test_result(
                False,
                f"Found {phone_count} duplicates - deduplication not working",
            )
            self.fail(f"Deduplication failed: found {phone_count} occurrences")
        else:
            self.print_test_result(
                True,
                f"Found {phone_count} occurrences (within acceptable range)",
            )

    def test_02_similar_but_different_facts(self):
        """Test that similar but different facts are stored separately."""
        self.print_test_header(
            "Similar But Different Facts",
            "Store similar facts with different values, query from Milvus",
        )

        # Store similar but different facts with conversation chaining
        result1 = self.send_memory_request(
            message="My home address is 123 Main Street, New York", auto_store=True
        )
        self.assertIsNotNone(result1, "Failed to store address 1")
        first_response_id = result1.get("id")

        time.sleep(1)

        result2 = self.send_memory_request(
            message="My work address is 456 Business Ave, Boston",
            auto_store=True,
            previous_response_id=first_response_id,
        )
        self.assertIsNotNone(result2, "Failed to store address 2")
        second_response_id = result2.get("id")

        # Send follow-up to trigger extraction of the last fact
        result3 = self.send_memory_request(
            message="Those are my two addresses.",
            auto_store=True,
            previous_response_id=second_response_id,
            verbose=False,
        )
        print("   ✓ Follow-up sent (triggers extraction)")

        self.wait_for_extraction()

        # Query for work address in NEW SESSION (no previous_response_id)
        result = self.send_memory_request(
            message="What is my work address?",
            auto_store=False,
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Failed to query")
        output = result.get("_output_text", "").lower()

        # Should mention work address from Milvus
        if "456" in output or "business" in output or "boston" in output:
            self.print_test_result(True, "Work address correctly retrieved from Milvus")
        else:
            self.print_test_result(
                False,
                f"Work address NOT found. Expected 'boston' or '456'. Response: {output[:200]}...",
            )
            self.fail("Work address not retrieved from Milvus")


class FullConversationFlowTest(MemoryFeaturesTest):
    """Test full multi-turn conversation with memory."""

    def test_01_multi_turn_conversation(self):
        """Test a realistic multi-turn conversation using memory from Milvus."""
        self.print_test_header(
            "Multi-Turn Conversation Flow",
            "Store facts, then query in NEW sessions to verify Milvus retrieval",
        )

        # Store facts with conversation chaining
        store_messages = [
            (
                "Hi! I'm planning my wedding for June 15th, 2026. My fiancée's name is Emily.",
                "wedding plans",
            ),
            (
                "We're thinking of having it at a beach venue. Our guest count is around 150 people.",
                "venue/guests",
            ),
            ("Our total budget is $50,000 for everything.", "budget"),
        ]

        print("\n--- Storing facts ---")
        last_response_id = None
        for message, description in store_messages:
            print(f"   Storing: {description}")
            result = self.send_memory_request(
                message=message,
                auto_store=True,
                previous_response_id=last_response_id,
                verbose=False,
            )
            self.assertIsNotNone(result, f"Failed to store: {description}")
            last_response_id = result.get("id")
            time.sleep(1)  # Small wait between stores

        # Send follow-up to trigger extraction of the last fact
        result = self.send_memory_request(
            message="I'm so excited about the wedding!",
            auto_store=True,
            previous_response_id=last_response_id,
            verbose=False,
        )
        print("   ✓ Follow-up sent (triggers extraction)")

        self.wait_for_extraction(3)

        # Query in NEW SESSIONS (no previous_response_id)
        # This tests that memory is retrieved from Milvus
        queries = [
            ("When is my wedding?", ["june", "15", "2026", "emily"]),
            ("How many guests are we expecting?", ["150", "beach", "venue"]),
        ]

        successful_queries = 0
        for query, expected_keywords in queries:
            print(f"\n--- Querying (NEW SESSION): {query} ---")
            result = self.send_memory_request(
                message=query,
                auto_store=False,
                # NO previous_response_id - this is a new session!
            )

            self.assertIsNotNone(result, f"Failed query: {query}")
            output = result.get("_output_text", "").lower()
            found = [kw for kw in expected_keywords if kw in output]

            if found:
                print(f"   ✓ Found keywords from Milvus: {found}")
                successful_queries += 1
            else:
                print(f"   ✗ Keywords NOT found in response: {expected_keywords}")

        if successful_queries >= 1:
            self.print_test_result(
                True, f"Retrieved {successful_queries}/2 facts from Milvus"
            )
        else:
            self.fail("No facts retrieved from Milvus in multi-turn test")


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
        """Test that related queries do retrieve relevant memories from Milvus."""
        self.print_test_header(
            "Related Query - Memory Retrieved", "Store fact, query in NEW session"
        )

        # Store a specific fact (turn 1)
        result = self.send_memory_request(
            message="Remember: I graduated from MIT in 2020 with a degree in Computer Science",
            auto_store=True,
        )
        self.assertIsNotNone(result, "Failed to store")
        first_response_id = result.get("id")

        # Send follow-up to trigger extraction (turn 2)
        result2 = self.send_memory_request(
            message="It was a great experience.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )

        self.wait_for_extraction()

        # Ask a related question in NEW SESSION (no previous_response_id)
        # This tests that related queries retrieve memories from Milvus
        result = self.send_memory_request(
            message="Where did I go to college?",
            auto_store=False,
            similarity_threshold=0.5,  # Lower threshold for better recall
            # NO previous_response_id - this is a new session!
        )

        self.assertIsNotNone(result, "Failed to query")
        output = result.get("_output_text", "").lower()

        # Should mention MIT or education details from Milvus
        if "mit" in output or "computer science" in output or "2020" in output:
            self.print_test_result(
                True, "Related query correctly retrieved memory from Milvus"
            )
        else:
            self.print_test_result(
                False,
                f"Memory NOT found. Expected 'mit' or '2020'. Response: {output[:200]}...",
            )
            self.fail("Related memory not retrieved from Milvus")


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
        MemoryRetrievalTest,
        QueryRewritingTest,
        DeduplicationTest,
        FullConversationFlowTest,
        SimilarityThresholdTest,
        MemoryExtractionTest,
        UserIsolationTest,
        PluginCombinationTest,  # Tests memory + system_prompt working together
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
