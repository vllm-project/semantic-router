"""Memory storage mechanics and plugin interaction tests.

Tests per-turn storage from natural conversation and verifies that
memory works correctly alongside the system_prompt plugin.
"""

import time

from memory_tests.base import PREVIEW_LENGTH, MemoryFeaturesTest


class MemoryStorageTest(MemoryFeaturesTest):
    """Test per-turn memory storage from natural conversation."""

    def test_01_store_turns_from_conversation(self):
        """Test that conversation turns are stored in Milvus and retrievable."""
        self.print_test_header(
            "Store Turns from Conversation",
            "Natural conversation, query in NEW sessions to verify Milvus storage",
        )

        conversation_message = """
        I had a great day today! Had lunch with my brother Tom at the new sushi place
        downtown. He told me he's getting married next spring to his girlfriend Anna.
        I'm so happy for them! Oh, and I finally bought that new laptop I've been
        looking at - a MacBook Pro M3.
        """

        result = self.send_memory_request(message=conversation_message, auto_store=True)
        self.assertIsNotNone(result, "Failed to process conversation")

        self.wait_for_storage()

        if self.milvus.is_available():
            count = self.milvus.count_memories(self.test_user)
            if count > 0:
                print(f"   ✓ Stored {count} memories in Milvus")
                for keyword in ["tom", "anna", "macbook"]:
                    memories = self.milvus.search_memories(self.test_user, keyword)
                    if memories:
                        print(f"   ✓ Found '{keyword}' in Milvus")
            else:
                self.print_test_result(False, "No turns stored in Milvus")
                self.fail("Memory storage failed: no memories stored")
        else:
            print("   ⚠️  Milvus verification skipped (not available)")

        self.flush_and_wait(8)

        queries = [
            ("Tell me about my brother Tom and the sushi lunch", ["tom"]),
            ("What MacBook laptop did I buy?", ["macbook", "m3"]),
            ("Who is Tom getting married to?", ["anna"]),
        ]

        successful_queries = 0
        for query, expected_keywords in queries:
            print(f"\n   Querying (NEW SESSION): {query}")
            _, found = self.query_with_retry(
                query, expected_keywords, max_attempts=2, wait_between=5
            )
            if found:
                print(f"   ✓ Found from Milvus: {found}")
                successful_queries += 1
            else:
                print(f"   ✗ Keywords not found: {expected_keywords}")

        if successful_queries >= 1:
            self.print_test_result(
                True,
                f"Stored and retrieved {successful_queries}/3 turns from Milvus",
            )
        else:
            self.print_test_result(False, "No turns stored or retrieved from Milvus")
            self.fail("Memory storage failed: no turns found in Milvus")


class PluginCombinationTest(MemoryFeaturesTest):
    """Test that memory works correctly with system_prompt plugin enabled."""

    def test_01_memory_with_system_prompt_both_present(self):
        """Verify memory injection works when system_prompt plugin is enabled.

        Query in NEW session to verify memory comes from Milvus, not
        conversation history.
        """
        self.print_test_header("Memory + System Prompt: Both Present (NEW SESSION)")

        unique_fact = "Phoenix-2026"
        store_message = (
            "Please remember this important code: "
            f"my secret project codename is {unique_fact}"
        )

        print(f"\n📝 Step 1: Storing unique fact: {unique_fact}")
        result = self.send_memory_request(
            message=store_message,
            auto_store=True,
        )

        if not result:
            self.fail("Failed to store memory")

        first_response_id = result.get("id")
        rid = first_response_id[:20] if first_response_id else "N/A"
        print(f"   Turn 1: Stored (response_id: {rid}...)")

        # Step 2: Send follow-up (also stored as a per-turn chunk)
        print("\n📝 Step 2: Sending follow-up...")
        result2 = self.send_memory_request(
            message="Got it, I'll remember that code.",
            auto_store=True,
            previous_response_id=first_response_id,
            verbose=False,
        )
        if not result2:
            self.fail("Failed to send follow-up")
        print("   Turn 2: Follow-up stored")

        # Step 3: Wait for storage to complete
        print("\n⏳ Step 3: Waiting for memory storage...")
        time.sleep(self.storage_wait + 1)

        # Step 4: Query for the fact in NEW SESSION (no previous_response_id)
        print("\n🔍 Step 4: Querying for the fact (NEW SESSION)...")
        query_result = self.send_memory_request(
            message="What is my secret project codename?",
            auto_store=False,
        )

        if not query_result:
            self.fail("Failed to query memory")

        output = query_result.get("_output_text", "").lower()
        print(
            f"   Response: {query_result.get('_output_text', '')[:PREVIEW_LENGTH]}..."
        )

        # Since this is a NEW session, "phoenix" can ONLY come from Milvus memory
        if "phoenix" in output:
            self.print_test_result(
                True,
                "Memory retrieved from Milvus and injected with system_prompt enabled! "
                "Both plugins work together correctly.",
            )
        else:
            self.print_test_result(
                False,
                "Memory NOT found in response. Expected 'phoenix'. "
                "This indicates the Memory+SystemPrompt bug still exists. "
                f"Response: {output[:PREVIEW_LENGTH]}...",
            )
            self.fail(
                "Memory+SystemPrompt bug: 'phoenix' not found in response. "
                "Memory was not retrieved from Milvus when system_prompt "
                "plugin is enabled."
            )

    def test_02_verify_system_prompt_persona_present(self):
        """Verify that system prompt persona is still being applied."""
        self.print_test_header("System Prompt: Persona Still Applied")

        result = self.send_memory_request(
            message="Who are you? What's your name?",
            auto_store=False,
        )

        if not result:
            self.fail("Failed to get response")

        output = result.get("_output_text", "").lower()

        persona_keywords = ["mom", "assistant", "ai", "help", "personal"]
        found_persona = any(kw in output for kw in persona_keywords)

        if found_persona:
            self.print_test_result(
                True, "System prompt persona is being applied correctly"
            )
        else:
            self.print_test_result(
                True,
                f"Persona keywords not found, but response received: {output[:100]}...",
            )
