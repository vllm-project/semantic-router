"""User memory isolation (security) tests.

Verifies that memories stored by one user are never visible to another,
both at the Milvus storage level and via the router's retrieval path.
"""

import time

from memory_tests.base import PREVIEW_LENGTH, MemoryFeaturesTest


class UserIsolationTest(MemoryFeaturesTest):
    """Test user memory isolation (security)."""

    def setUp(self):
        """Set up test configuration with two users."""
        super().setUp()
        self.user_a = f"isolation_user_a_{int(time.time())}"
        self.user_b = f"isolation_user_b_{int(time.time())}"
        self.user_a_secret = "My secret PIN is 9876"
        self.user_b_secret = "My password is hunter2"

    def _store_secret_with_followup(self, secret: str, user_id: str, followup: str):
        """Store a secret and send a follow-up turn for a user."""
        result = self.send_memory_request(
            message=f"Remember this: {secret}",
            auto_store=True,
            user_id=user_id,
            verbose=False,
        )
        response_id = result.get("id") if result else None
        if response_id:
            self.send_memory_request(
                message=followup,
                auto_store=True,
                user_id=user_id,
                previous_response_id=response_id,
                verbose=False,
            )
        return result

    def test_01_store_user_a_memory(self):
        """Store a secret for User A."""
        self.print_test_header(
            "Store User A Secret", f"Storing: '{self.user_a_secret}'"
        )

        result = self.send_memory_request(
            message=f"Remember this: {self.user_a_secret}",
            auto_store=True,
            user_id=self.user_a,
        )

        self.assertIsNotNone(result, "Failed to store User A memory")
        self.assertEqual(result.get("status"), "completed")
        first_response_id = result.get("id")
        print(f"   ✓ Secret stored (response_id: {first_response_id[:20]}...)")

        _result2 = self.send_memory_request(
            message="Got it, I'll keep that safe.",
            auto_store=True,
            user_id=self.user_a,
            previous_response_id=first_response_id,
            verbose=False,
        )
        self.assertIsNotNone(_result2, "Failed to send follow-up")
        print("   ✓ Follow-up stored")

        self.print_test_result(True, "User A secret stored")

        self.wait_for_storage()

    def test_02_user_b_cannot_see_user_a_secret(self):
        """Security: User B should NOT see User A's secret."""
        self.print_test_header(
            "Security Check: User B queries User A's secret",
            "User B should NOT see User A's PIN",
        )

        self._store_secret_with_followup(
            self.user_a_secret, self.user_a, "That's my secret PIN."
        )
        print("   ✓ User A's secret stored")

        self.wait_for_storage()

        result = self.send_memory_request(
            message="What is my PIN?", auto_store=False, user_id=self.user_b
        )

        self.assertIsNotNone(result, "Request failed")
        output = result.get("_output_text", "").lower()

        has_leaked = "9876" in output

        if has_leaked:
            self.print_test_result(
                False, "SECURITY VIOLATION: User B saw User A's PIN!"
            )
            self.fail(
                f"SECURITY VIOLATION: User B saw User A's secret: "
                f"{output[:PREVIEW_LENGTH]}"
            )
        else:
            self.print_test_result(True, "User B correctly cannot see User A's secret")

    def test_03_user_a_can_see_own_memory(self):
        """User A should be able to see their own secret from Milvus."""
        self.print_test_header(
            "User A Queries Own Memory", "User A should see their own PIN from Milvus"
        )

        self._store_secret_with_followup(
            self.user_a_secret, self.user_a, "That's my secret PIN."
        )
        print("   ✓ User A's secret stored")

        self.wait_for_storage()

        # User A queries in NEW SESSION (no previous_response_id)
        result = self.send_memory_request(
            message="What is my PIN?",
            auto_store=False,
            user_id=self.user_a,
        )

        self.assertIsNotNone(result, "Request failed")
        output = result.get("_output_text", "").lower()

        if "9876" in output:
            self.print_test_result(
                True, "User A correctly retrieved their own PIN from Milvus"
            )
        else:
            self.print_test_result(
                False,
                f"User A's PIN NOT found. Expected '9876'. "
                f"Response: {output[:PREVIEW_LENGTH]}...",
            )
            self.fail("User A cannot retrieve their own memory from Milvus")

    def _verify_milvus_isolation(self):
        """Check Milvus storage for cross-user data leaks."""
        if not self.milvus.is_available():
            print("   ⚠️  Milvus verification skipped (not available)")
            return

        user_a_memories = self.milvus.search_memories(self.user_a, "9876")
        user_b_memories = self.milvus.search_memories(self.user_b, "hunter2")

        user_a_leak = self.milvus.search_memories(self.user_a, "hunter2")
        user_b_leak = self.milvus.search_memories(self.user_b, "9876")

        if user_a_leak:
            self.fail("SECURITY: User A's Milvus partition contains User B's password!")
        if user_b_leak:
            self.fail("SECURITY: User B's Milvus partition contains User A's PIN!")

        print(
            f"   ✓ Milvus isolation verified: "
            f"User A has {len(user_a_memories)} memories, "
            f"User B has {len(user_b_memories)}"
        )

    def test_04_bidirectional_isolation(self):
        """Test isolation works both ways - query in NEW sessions."""
        self.print_test_header(
            "Bidirectional Isolation",
            "Neither user should see the other's secrets from Milvus",
        )

        # Store secrets for both users
        self._store_secret_with_followup(
            self.user_a_secret, self.user_a, "That's my secret PIN."
        )
        self._store_secret_with_followup(
            self.user_b_secret, self.user_b, "That's my password."
        )
        print("   ✓ Both users' secrets stored")

        self.wait_for_storage(3)

        self._verify_milvus_isolation()

        # User A tries to get User B's password in NEW SESSION
        result = self.send_memory_request(
            message="What is my password?",
            auto_store=False,
            user_id=self.user_a,
        )
        self.assertIsNotNone(result, "Request failed")
        output_a = result.get("_output_text", "").lower()

        # User B tries to get User A's PIN in NEW SESSION
        result = self.send_memory_request(
            message="What is my PIN?",
            auto_store=False,
            user_id=self.user_b,
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

        self.print_test_result(True, "Bidirectional isolation verified at Milvus level")
