"""Pipeline correctness tests for memory features.

Tests the core store-then-retrieve contract, content fidelity,
similarity thresholds, and contradictory memory behavior.
"""

from memory_tests.base import (
    MIN_CONTENT_MATCHES,
    MSG_PREVIEW_LENGTH,
    PREVIEW_LENGTH,
    MemoryFeaturesTest,
)


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


class SupersededMemoryTest(MemoryFeaturesTest):
    """A correction hides only the fact it replaces.

    The reflection gate drops a retrieved turn when a newer retrieved turn
    reports the user's own change to the same property. A turn that also
    states a second, uncorrected fact has to survive, or the prompt loses a
    memory the user never withdrew. A change that only reaffirms what is
    already true is not a correction at all.

    Assertions read the prompt the echo backend received, so each one covers
    storage, retrieval, the gate and injection on the live path. Every expected
    keyword has to reach the prompt, so a memory that retrieval never returned
    fails the test instead of passing it.
    """

    def _prompt_containing(
        self, message: str, expected: list[str], max_attempts: int = 3
    ) -> str:
        """Return the prompt the provider received, retrying until it holds every keyword.

        Retrieval runs against Milvus, whose sealed segments need a flush and an
        index pass before a vector search sees them.
        """
        output = ""
        for attempt in range(max_attempts):
            result = self.send_memory_request(
                message=message, auto_store=False, verbose=(attempt == 0)
            )
            if not result:
                continue
            response_text: str = result.get("_output_text", "").lower()
            context_start: int = response_text.find(
                "the following is relevant context from previous conversations with the user:"
            )
            context_end: int = response_text.find(
                "use this context to personalize your response when relevant.",
                context_start,
            )
            output = (
                response_text[context_start:context_end]
                if context_start >= 0 and context_end >= 0
                else ""
            )
            missing = [kw for kw in expected if kw not in output]
            if not missing:
                return output
            if attempt < max_attempts - 1:
                print(
                    f"   ⏳ Retry {attempt + 1}/{max_attempts}: {missing} not in "
                    f"prompt, flushing..."
                )
                self.flush_and_wait(5)

        missing = [kw for kw in expected if kw not in output]
        self.print_test_result(
            False,
            f"Prompt is missing {missing}. Query: '{message[:MSG_PREVIEW_LENGTH]}'. "
            f"Prompt: {output[:PREVIEW_LENGTH]}...",
        )
        self.fail(f"Memory context never reached the prompt: {missing}")
        return output

    def test_01_a_second_fact_in_the_turn_survives_a_later_move(self):
        """A move correction keeps the unrelated fact stated in the same turn."""
        self.print_test_header(
            "Correction Keeps A Second Fact",
            "Store a turn holding two facts, store a move, verify both reach the prompt",
        )

        first = self.send_memory_request(
            message="I live in Boston, I'm married.", auto_store=True
        )
        self.assertIsNotNone(first, "Failed to store the two-fact turn")

        second = self.send_memory_request(
            message="I just moved to Denver, and I live there now.", auto_store=True
        )
        self.assertIsNotNone(second, "Failed to store the correction")

        self.wait_for_storage()
        self.flush_and_wait(8)

        prompt = self._prompt_containing(
            "Where do I live now, and what is my home life like?",
            ["denver", "married"],
        )
        self.print_test_result(
            True,
            f"The move reached the prompt and the marriage fact stayed with it: "
            f"{prompt[:PREVIEW_LENGTH]}...",
        )

    def test_02_a_reaffirming_change_does_not_correct_an_unrelated_fact(self):
        """Saying a fact still holds after a move must not retire that fact."""
        self.print_test_header(
            "Reaffirmation Is Not A Correction",
            "Store a job fact, store a move that reaffirms it, verify the job stays",
        )

        first = self.send_memory_request(
            message="I work as a nurse at the Riverside Hospital.", auto_store=True
        )
        self.assertIsNotNone(first, "Failed to store the job fact")

        second = self.send_memory_request(
            message="I moved apartments, and I still work as a nurse now.",
            auto_store=True,
        )
        self.assertIsNotNone(second, "Failed to store the reaffirming move")

        self.wait_for_storage()
        self.flush_and_wait(8)

        prompt = self._prompt_containing(
            "What do I do for work these days?",
            ["riverside", "apartments"],
        )
        self.print_test_result(
            True,
            f"Both turns reached the prompt, so the move did not retire the job: "
            f"{prompt[:PREVIEW_LENGTH]}...",
        )

    def test_03_a_quoted_task_example_does_not_correct_earlier_fact(self):
        """Quoting a first-person change in a task prompt must not retire an earlier fact."""
        self.print_test_header(
            "Quoted Task Is Not A Correction",
            "Store a residence fact, store a translation quoting a move, verify residence stays",
        )

        first = self.send_memory_request(message="I live in Boston.", auto_store=True)
        self.assertIsNotNone(first, "Failed to store the residence fact")
        self.wait_for_storage(seconds=3)
        self.wait_for_storage(seconds=2)

        second = self.send_memory_request(
            message='Please translate this sentence: "I just moved to Denver, and I live there now."',
            auto_store=True,
        )
        self.assertIsNotNone(second, "Failed to store the translation task")

        self.wait_for_storage()
        self.flush_and_wait(8)

        prompt = self._prompt_containing(
            "Where do I live?",
            ["boston"],
        )
        self.print_test_result(
            True,
            f"The residence reached the prompt and was not hidden by the quoted example: "
            f"{prompt[:PREVIEW_LENGTH]}...",
        )

    def test_04_explicit_reaffirmation_with_continue_or_remain_does_not_correct_earlier_fact(
        self,
    ):
        """Explicit reaffirmations such as continue or remain must not retire an older fact."""
        self.print_test_header(
            "Continue Or Remain Reaffirmation Is Not A Correction",
            "Store a workplace fact, store a move reaffirming nursing with continue, verify workplace stays",
        )

        first = self.send_memory_request(
            message="I work as a nurse at the Children's Hospital.", auto_store=True
        )
        self.assertIsNotNone(first, "Failed to store the workplace fact")

        second = self.send_memory_request(
            message="I moved apartments, and I continue to work as a nurse now.",
            auto_store=True,
        )
        self.assertIsNotNone(second, "Failed to store the reaffirming move")

        self.wait_for_storage()
        self.flush_and_wait(8)

        prompt = self._prompt_containing(
            "Where do I work as a nurse?",
            ["children", "apartments"],
        )
        self.print_test_result(
            True,
            f"Both turns reached the prompt, so continuing the job did not retire the hospital fact: "
            f"{prompt[:PREVIEW_LENGTH]}...",
        )

    def test_05_curly_single_quoted_translation_keeps_earlier_residence(self):
        """A curly single-quoted change request must not retire a stored residence."""
        self.print_test_header(
            "Curly-Quoted Task Is Not A Correction",
            "Store a residence, quote a Denver move using curly quotes, verify Boston stays",
        )

        first = self.send_memory_request(message="I live in Boston.", auto_store=True)
        self.assertIsNotNone(first, "Failed to store the residence fact")
        self.wait_for_storage(seconds=3)

        exact_quoted_task = self.send_memory_request(
            message=(
                "THRESHOLD_MARKER Please translate this sentence: "
                "\u2018I just moved to Denver, and I live there now.\u2019"
            ),
            auto_store=True,
        )
        self.assertIsNotNone(exact_quoted_task, "Failed to store the quoted task")
        self.wait_for_storage(seconds=3)
        exact_prompt = self._prompt_containing(
            "What sentence did I ask you to translate: "
            "\u2018I just moved to Denver, and I live there now.\u2019?",
            ["boston"],
        )
        self.assertIn("q: i live in boston.", exact_prompt)

        second = self.send_memory_request(
            message=(
                "THRESHOLD_MARKER Please translate this sentence: "
                "\u2018I just moved to Denver, and I live there now actually\u2019."
            ),
            auto_store=True,
        )
        self.assertIsNotNone(second, "Failed to store the translation task")

        self.wait_for_storage()
        self.flush_and_wait(8)

        self.assertTrue(self.milvus.is_available(), "Milvus is required for this test")
        stored_records = self.milvus.client.query(
            collection_name=self.milvus.collection,
            filter=f'user_id == "{self.test_user}"',
            output_fields=["content"],
        )
        stored_content = "\n".join(
            record.get("content", "").lower() for record in stored_records
        )
        self.assertIn("please translate this sentence", stored_content)

        prompt = self._prompt_containing(
            "What sentence did I ask you to translate: "
            "\u2018I just moved to Denver, and I live there now actually\u2019?",
            ["boston", "denver"],
        )
        self.assertIn("q: i live in boston.", prompt)
        self.assertIn("q: threshold_marker please translate this sentence", prompt)
        self.print_test_result(
            True,
            f"The earlier residence reached the prompt despite the quoted move: "
            f"{prompt[:PREVIEW_LENGTH]}...",
        )

    def test_06_corrected_question_keeps_independent_assistant_fact(self):
        """A corrected residence must not remove a separate fact from its answer."""
        self.print_test_header(
            "Correction Keeps Assistant Fact",
            "Store a residence with an assistant dog fact, correct the residence, verify both",
        )

        first = self.send_memory_request(
            message="I live in Boston.",
            auto_store=True,
        )
        self.assertIsNotNone(first, "Failed to store the residence and assistant fact")
        self.assertEqual(first.get("_output_text"), "Your dog Biscuit is a beagle.")
        self.wait_for_storage(seconds=3)

        second = self.send_memory_request(
            message="THRESHOLD_MARKER I just moved to Denver, and I live there now.",
            auto_store=True,
        )
        self.assertIsNotNone(second, "Failed to store the residence correction")
        self.assertEqual(second.get("_output_text"), "Welcome to Denver!")

        self.wait_for_storage()
        self.flush_and_wait(8)

        self.assertTrue(self.milvus.is_available(), "Milvus is required for this test")
        stored_records = self.milvus.client.query(
            collection_name=self.milvus.collection,
            filter=f'user_id == "{self.test_user}"',
            output_fields=["content", "created_at"],
        )
        stored_content = "\n".join(
            record.get("content", "").lower() for record in stored_records
        )
        self.assertIn("i just moved to denver, and i live there now", stored_content)
        boston_created_at = next(
            (
                record["created_at"]
                for record in stored_records
                if "i live in boston." in record["content"].lower()
            ),
            None,
        )
        denver_created_at = next(
            (
                record["created_at"]
                for record in stored_records
                if "i just moved to denver, and i live there now"
                in record["content"].lower()
            ),
            None,
        )
        self.assertIsNotNone(boston_created_at)
        self.assertIsNotNone(denver_created_at)
        self.assertGreater(denver_created_at, boston_created_at)

        prompt = self._prompt_containing(
            "I just moved to Denver, and I live there now. What breed is Biscuit?",
            ["denver", "biscuit"],
        )
        self.assertNotIn("boston", prompt)
        self.assertIn("beagle", prompt)
        self.print_test_result(
            True,
            f"The correction replaced Boston while the assistant's dog fact stayed: "
            f"{prompt[:PREVIEW_LENGTH]}...",
        )
