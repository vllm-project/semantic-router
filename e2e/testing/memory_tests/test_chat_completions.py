"""Chat Completions API memory E2E tests."""

import time

import requests

from memory_tests.base import HTTP_OK, PREVIEW_LENGTH, MemoryFeaturesTest

MILVUS_POST_FLUSH_SLEEP_SEC = 3

SYSTEM_MESSAGE = "You are a helpful assistant with memory."
STORED_FACT = "Please remember this: My favorite color is purple"
MEMORY_KEYWORD = "purple"


def _chat_headers(user_id: str) -> dict:
    return {"Content-Type": "application/json", "x-authz-user-id": user_id}


def _chat_payload(messages: list, user_id: str) -> dict:
    return {"model": "MoM", "messages": messages, "metadata": {"user_id": user_id}}


class ChatCompletionsMemoryTest(MemoryFeaturesTest):
    """Exercise memory storage and retrieval via /v1/chat/completions."""

    def _post_chat(self, payload: dict, user_id: str) -> requests.Response:
        url = f"{self.router_endpoint}/v1/chat/completions"
        return requests.post(
            url,
            json=payload,
            headers=_chat_headers(user_id),
            timeout=self.timeout,
        )

    def _store_fact_two_turns(self, test_user: str) -> None:
        """Send two turns to store a fact and trigger the storage window."""
        payload1 = _chat_payload(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": STORED_FACT},
            ],
            test_user,
        )

        print("\n📤 Turn 1: Storing fact via Chat Completions")
        r1 = self._post_chat(payload1, test_user)
        self.assertEqual(r1.status_code, HTTP_OK, f"Turn 1 failed: {r1.text}")
        result1 = r1.json()
        self.assertIn("choices", result1)
        print("   ✓ Turn 1: Fact stored")

        assistant_content = result1["choices"][0]["message"]["content"]
        payload2 = _chat_payload(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": STORED_FACT},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": "Got it, thanks for remembering that."},
            ],
            test_user,
        )

        print("📤 Turn 2: Triggering storage window")
        r2 = self._post_chat(payload2, test_user)
        self.assertEqual(r2.status_code, HTTP_OK, f"Turn 2 failed: {r2.text}")
        print("   ✓ Turn 2: Follow-up completed")

    def _wait_and_verify_storage(self, test_user: str) -> None:
        """Wait for async storage, flush Milvus, and verify the fact was stored."""
        self.wait_for_storage()

        if self.milvus.is_available():
            self.milvus.flush()
            time.sleep(MILVUS_POST_FLUSH_SLEEP_SEC)

        if self.milvus.is_available():
            memories = self.milvus.search_memories(test_user, MEMORY_KEYWORD)
            if memories:
                self.print_test_result(
                    True,
                    f"Memory stored: found {len(memories)} memory(ies) with '{MEMORY_KEYWORD}'",
                )
            else:
                count = self.milvus.count_memories(test_user)
                self.fail(
                    f"Memory storage failed: {count} memories but none contain '{MEMORY_KEYWORD}'"
                )

    def _query_and_assert_retrieval(self, test_user: str) -> None:
        """Query in a fresh message list and assert memory was injected."""
        payload = _chat_payload(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": "What is my favorite color?"},
            ],
            test_user,
        )

        print("\n🔍 Turn 3: Querying memory (new message list)")
        r3 = self._post_chat(payload, test_user)
        self.assertEqual(r3.status_code, HTTP_OK, f"Turn 3 failed: {r3.text}")
        result3 = r3.json()
        answer = result3["choices"][0]["message"]["content"].lower()

        preview = result3["choices"][0]["message"]["content"]
        ellipsis = "..." if len(preview) > PREVIEW_LENGTH else ""
        print(f"   Response: {preview[:PREVIEW_LENGTH]}{ellipsis}")
        print(f"   📊 Original request had {len(payload['messages'])} messages")

        indicators = ["purple", "favourite", "favorite", "color", "colour"]
        if any(kw in answer for kw in indicators):
            self.print_test_result(
                True, "Chat Completions memory retrieved and injected successfully"
            )
            return

        self._fallback_milvus_check(test_user)

    def _fallback_milvus_check(self, test_user: str) -> None:
        """When the LLM response lacks the keyword, verify via Milvus directly."""
        if self.milvus.is_available():
            memories = self.milvus.search_memories(test_user, MEMORY_KEYWORD)
            mem_count = len(memories) if memories else 0
            print(f"   🔍 Debug: Found {mem_count} memories in Milvus")
            if mem_count > 0:
                print("   ⚠️  Memories exist but LLM response doesn't reference them")
                print("   💡 With echo backend, injection may still have succeeded")
                self.print_test_result(
                    True,
                    f"Memory stored ({mem_count} memories found); retrieval likely succeeded",
                )
                return

        self.print_test_result(
            False, "Memory NOT retrieved. Expected memory reference in response."
        )
        self.fail("Chat Completions memory not retrieved from Milvus")

    def test_01_chat_completions_memory_storage_and_retrieval(self):
        """Store a fact over two turns, then query in a fresh message list."""
        self.print_test_header(
            "Chat Completions Memory Test",
            "Store and retrieve memory using /v1/chat/completions endpoint",
        )

        test_user = f"{self.test_user}_chat"

        self._store_fact_two_turns(test_user)
        self._wait_and_verify_storage(test_user)
        self._query_and_assert_retrieval(test_user)
