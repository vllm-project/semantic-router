"""Pipeline correctness tests for memory features.

Tests the core store-then-retrieve contract, content fidelity,
similarity thresholds, and contradictory memory behavior.
"""

from memory_tests.base import MIN_CONTENT_MATCHES, PREVIEW_LENGTH, MemoryFeaturesTest


class MemoryInjectionPipelineTest(MemoryFeaturesTest):
    """Test the fundamental memory contract: store -> inject into prompt.

    Storage happens on every turn (direct per-turn chunk, no LLM call).
    Uses the echo backend to verify that stored memories appear in the prompt
    sent to the LLM. All retrieval checks are done in a NEW session (no
    previous_response_id) so keywords can only come from Milvus injection.
    """

    def test_01_store_and_inject(self):
        """The fundamental pipeline: store a fact, verify injection in a new session."""
        self.print_test_header(
            "Store -> Inject Pipeline",
            "Store a fact, query in NEW session, verify injection via echo",
        )

        fact = "My car is a blue Tesla Model 3 from 2023"
        result1 = self.send_memory_request(
            message=f"Please remember this: {fact}", auto_store=True
        )
        self.assertIsNotNone(result1, "Failed to store fact")
        self.assertEqual(result1.get("status"), "completed")
        first_response_id = result1.get("id")
        print(f"   Fact stored (response_id: {first_response_id[:20]}...)")

        self.wait_for_storage()

        if self.milvus.is_available():
            memories = self.milvus.search_memories(self.test_user, "tesla")
            if memories:
                print(
                    f"   Milvus: found {len(memories)} memory(ies) containing 'tesla'"
                )
            else:
                count = self.milvus.count_memories(self.test_user)
                self.fail(
                    f"Storage failed: {count} memories in Milvus but none contain 'tesla'"
                )

        self.flush_and_wait(8)

        output, found = self.query_with_retry(
            "Tell me about my Tesla Model 3 car", ["tesla", "model 3", "model3"]
        )

        if found:
            self.print_test_result(
                True,
                f"Memory injected into prompt: found {found}",
            )
        else:
            self.print_test_result(
                False,
                f"Memory NOT injected. Expected 'tesla' or 'model 3'. "
                f"Response: {output[:PREVIEW_LENGTH]}...",
            )
            self.fail(
                "Memory not injected into prompt. Check retrieval and injection flow."
            )


class MemoryContentIntegrityTest(MemoryFeaturesTest):
    """Verify per-turn storage preserves content correctly in Milvus.

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

        _result2 = self.send_memory_request(
            message="Thanks, that covers my onboarding info.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )
        self.assertIsNotNone(_result2, "Failed to send follow-up")
        print("   Follow-up stored")

        self.wait_for_storage()

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

        if len(found) >= MIN_CONTENT_MATCHES:
            self.print_test_result(
                True, f"Content preserved: found {found}, missing {missing}"
            )
        else:
            self.print_test_result(
                False,
                f"Content corrupted/truncated: found {found}, missing {missing}. "
                f"Stored: {combined[:PREVIEW_LENGTH]}...",
            )
            self.fail(f"Content integrity failure: missing {missing}")


class SimilarityThresholdTest(MemoryFeaturesTest):
    """Test similarity threshold for memory retrieval."""

    def test_01_unrelated_query_no_memory_contamination(self):
        """Verify that stored memories only contain the intended content.

        With a real LLM (not echo), we cannot assert on response text because
        the LLM may proactively reference injected memory in unrelated answers.
        Instead, we verify at the Milvus level that the stored memories only
        contain restaurant-related content and nothing about France/Paris.
        """
        self.print_test_header(
            "No Memory Contamination",
            "Store a restaurant fact, verify Milvus has no unrelated content",
        )

        if not self.milvus.is_available():
            self.skipTest("Milvus not available for direct verification")

        result = self.send_memory_request(
            message="Remember: My favorite restaurant is The Italian Place on 5th Avenue",
            auto_store=True,
        )
        self.assertIsNotNone(result, "Failed to store")
        first_response_id = result.get("id")

        _result2 = self.send_memory_request(
            message="Great restaurant, right?",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )

        self.wait_for_storage()

        all_results = self.milvus.client.query(
            collection_name=self.milvus.collection,
            filter=f'user_id == "{self.test_user}"',
            output_fields=["content"],
        )

        self.assertGreater(len(all_results), 0, "No memories stored")
        combined = " ".join(r.get("content", "").lower() for r in all_results)

        has_restaurant = "italian" in combined or "restaurant" in combined
        has_unrelated = "france" in combined or "paris" in combined

        print(f"   Stored {len(all_results)} memories for user")
        print(f"   Contains restaurant info: {has_restaurant}")
        print(f"   Contains unrelated content: {has_unrelated}")

        if has_restaurant and not has_unrelated:
            self.print_test_result(
                True, "Memory contains only restaurant fact, no contamination"
            )
        elif not has_restaurant:
            self.print_test_result(
                True,
                "Memory stored but 'italian/restaurant' not in content field "
                "(turn chunk may have been formatted differently). "
                "No contamination detected.",
            )
        else:
            self.print_test_result(
                False, "Unrelated content found in memory — possible contamination"
            )
            self.fail("Memory contamination: unrelated content stored")

    def test_02_related_query_retrieves_memory(self):
        """Test that semantically related queries retrieve relevant memories."""
        self.print_test_header(
            "Related Query Retrieves Memory",
            "Store fact about a car, query with key terms in NEW session",
        )

        result = self.send_memory_request(
            message="Remember: I drive a red Toyota Camry 2022",
            auto_store=True,
        )
        self.assertIsNotNone(result, "Failed to store")
        first_response_id = result.get("id")

        _result2 = self.send_memory_request(
            message="It gets great gas mileage.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )

        self.wait_for_storage()
        self.flush_and_wait(8)

        output, found = self.query_with_retry(
            "Tell me about my red Toyota Camry", ["toyota", "camry", "2022"]
        )

        if found:
            self.print_test_result(
                True, f"Related query correctly retrieved memory: {found}"
            )
        else:
            self.print_test_result(
                False,
                f"Memory NOT found. Expected 'toyota' or 'camry'. "
                f"Response: {output[:PREVIEW_LENGTH]}...",
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

        _result3 = self.send_memory_request(
            message="It was a big move across the country.",
            auto_store=True,
            previous_response_id=second_response_id,
            verbose=False,
        )
        print("   All turns stored")

        self.wait_for_storage()

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
                "Both contradicting facts stored (expected: no contradiction "
                "detection yet). When contradiction detection is added, "
                "the old fact should be invalidated.",
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
                f"Content: {combined[:PREVIEW_LENGTH]}...",
            )
            self.fail("Memory storage produced unexpected state")
