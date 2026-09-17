"""Unit tests for the PR2 budget ledger."""

import threading
import unittest

from .accounting import (
    BudgetExhausted,
    BudgetLedger,
    Usage,
    estimate_prompt_tokens,
)


class AccountingTests(unittest.TestCase):
    def test_usage_preserves_missing_and_zero_fields(self):
        usage = Usage.from_mapping(
            {"prompt_tokens": 0, "completion_tokens": 3, "total_tokens": 3}
        )
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertTrue(usage.complete)
        self.assertEqual(Usage.from_mapping({}).as_record()["total_tokens"], None)

    def test_unknown_usage_is_charged_at_reservation(self):
        ledger = BudgetLedger(max_calls=2, max_total_tokens=20)
        reservation = ledger.reserve("call-a", 10)
        settlement = ledger.settle(reservation, {})
        self.assertFalse(settlement.usage_known)
        self.assertEqual(settlement.usage_source, "reservation")
        self.assertEqual(settlement.charged_tokens, 10)
        self.assertEqual(ledger.snapshot().unknown_usage_calls, 1)

    def test_partial_usage_uses_prompt_plus_completion_for_admission(self):
        ledger = BudgetLedger(max_calls=1, max_total_tokens=10)
        reservation = ledger.reserve("call-a", 10)
        settlement = ledger.settle(
            reservation, {"prompt_tokens": 4, "completion_tokens": 2}
        )
        self.assertTrue(settlement.usage_known)
        self.assertEqual(settlement.charged_tokens, 6)
        # The provider total remains unknown in the normalized evidence.
        self.assertIsNone(settlement.usage.total_tokens)

    def test_reservation_prevents_concurrent_oversubscription(self):
        ledger = BudgetLedger(max_calls=8, max_total_tokens=10)
        admitted = []
        rejected = []
        lock = threading.Lock()

        def worker(index):
            try:
                reservation = ledger.reserve("call-{}".format(index), 6)
            except BudgetExhausted:
                with lock:
                    rejected.append(index)
                return
            with lock:
                admitted.append(reservation.call_id)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(admitted), 1)
        self.assertEqual(len(rejected), 7)
        self.assertEqual(ledger.snapshot().reserved_tokens, 6)

    def test_actual_provider_overrun_marks_budget_exhausted(self):
        ledger = BudgetLedger(max_calls=1, max_total_tokens=5)
        reservation = ledger.reserve("call-a", 5)
        settlement = ledger.settle(
            reservation,
            {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
        )
        self.assertTrue(settlement.exhausted)
        self.assertEqual(ledger.snapshot().tokens, 8)

    def test_prompt_estimate_is_stable_and_positive(self):
        messages = [{"role": "user", "content": "hello"}]
        self.assertEqual(
            estimate_prompt_tokens(messages), estimate_prompt_tokens(messages)
        )
        self.assertGreater(estimate_prompt_tokens(messages), 0)


if __name__ == "__main__":
    unittest.main()
