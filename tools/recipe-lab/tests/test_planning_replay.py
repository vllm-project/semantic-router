"""Check what an offline routing preflight can establish from real wire fields."""

import hashlib
import json
import unittest

from recipe_lab.planning_replay import replay_task


def pair(call, roles, start, end, model="strong"):
    return [
        {
            "event": "dispatch",
            "call": call,
            "model": "public",
            "wire_fields": ["messages", "model", "max_completion_tokens"],
            "last_role": roles[-1],
            "roles": roles,
            "messages_sha256": hashlib.sha256(str(call).encode()).hexdigest(),
            "params": {"max_completion_tokens": 32},
            "start": start,
        },
        {
            "event": "response",
            "call": call,
            "http": 200,
            "done": True,
            "end": end,
            "selected_model": model,
            "returned_model": model,
            "usage": {"prompt_tokens": 7},
        },
    ]


def replay(rows, feature="tool_results", **kwargs):
    args = dict(
        public_model="public",
        fast="fast",
        strong="strong",
        candidate={"id": "late", "feature": feature, "lower": 2, "upper": None},
        fast_context=100,
        fast_max_output=64,
        task_seconds=20,
    )
    return replay_task(
        "\n".join(map(json.dumps, rows)).encode("utf-8"), **(args | kwargs)
    )


class PlannerReplayTest(unittest.TestCase):
    def test_role_choice_changes_calls_and_model_exposure_is_wall_union(self):
        short = ["user", "assistant"]
        long = [*short, "tool", "tool"]
        rows = pair(1, short, 0, 1) + pair(2, long, 2, 7) + pair(3, long, 4, 10)
        rows += pair(4, long, 10, 11, "fast")
        tools = replay(rows)
        self.assertEqual(tools["matched_calls"], [2, 3, 4])
        self.assertEqual(tools["matched_source_fast_calls"], [4])
        self.assertEqual(tools["opportunity_calls"], [2, 3])
        self.assertEqual(tools["opportunity_seconds"], 8)
        self.assertEqual(tools["opportunity_fraction"], 0.4)
        self.assertEqual(replay(rows, "all_messages")["matched_calls"], [1, 2, 3, 4])
        self.assertEqual(replay(rows, "fixed_strong")["matched_calls"], [])
        self.assertEqual(tools["actual_candidate_route"], "unobserved")

    def test_capacity_is_an_estimate_and_explicit_over_limit_is_a_blocker(self):
        roles = ["user", "tool", "tool"]
        rows = pair(1, roles, 0, 1) + pair(2, roles, 1, 2) + pair(3, roles, 2, 3)
        rows[2]["params"]["max_completion_tokens"] = 101
        rows[5].pop("usage")
        out = replay(rows, fast_context=33)
        self.assertEqual(out["opportunity_calls"], [1, 3])
        self.assertEqual(out["fit_estimate_exceeded_calls"], [1])
        self.assertEqual(out["hard_blocked_calls"], [2])
        self.assertEqual(out["missing_fit_calls"], [3])
        self.assertEqual(out["calls"][0]["fast_token_fit"]["source_prompt_tokens"], 7)

    def test_message_count_matches_router_facts_and_content_filter(self):
        roles = ["system", "system", "developer", "developer", "user", "tool", "user"]
        rows = pair(1, roles, 1, 2)
        self.assertEqual(replay(rows, "all_messages")["calls"][0]["count"], 6)
        filtered = replay(rows, "all_messages", skip_user_messages_matching=True)
        self.assertEqual(
            filtered["unknown_calls"],
            {1: "user_message_filter_requires_message_content"},
        )
        self.assertEqual(
            replay(rows, skip_user_messages_matching=True)["calls"][0]["count"], 1
        )

    def test_incomplete_and_ambiguous_model_ids_keep_separate_receipts(self):
        roles = ["user", "tool", "tool"]
        rows = sum((pair(i, roles, i, i + 1) for i in range(1, 5)), [])
        rows[1]["http"], rows[1]["done"] = 429, False
        rows[3]["done"] = False
        rows[5]["returned_model"] = "strong-service"
        rows[7]["returned_model"] = "unexpected"
        r = replay(rows, backend_aliases={"strong": "strong-service"})
        self.assertEqual(r["rate_limited_calls"], [1])
        self.assertEqual(r["incomplete_calls"], [2])
        self.assertEqual(r["transport_failure_calls"], [1, 2])
        self.assertEqual(r["opportunity_calls"], [3])
        self.assertEqual(r["matched_unknown_source_calls"], [4])

    def test_unknown_roles_and_timing_do_not_become_fabricated_estimates(self):
        roles = ["user", "tool", "tool"]
        rows = pair(1, roles, 0, 1) + pair(2, roles, 1, 2)
        rows[0].pop("roles")
        rows[2].pop("start")
        r = replay(rows)
        self.assertIn(1, r["unknown_calls"])
        self.assertEqual(r["matched_calls"], [2])
        self.assertEqual(r["missing_opportunity_times"], [2])
        self.assertIsNone(r["opportunity_fraction"])
        self.assertIsNone(r["opportunity_seconds"])
        unsupported = replay(rows, "unknown")
        self.assertEqual(len(unsupported["unknown_calls"]), 2)
        self.assertIsNone(unsupported["opportunity_seconds"])

    def test_invalid_inputs_fail_closed(self):
        for damage in [
            "bad_public",
            "duplicate",
            "missing_pair",
            "bad_role",
            "bad_interval",
        ]:
            rows = pair(1, ["user", "tool", "tool"], 0, 1)
            with self.subTest(damage=damage):
                if damage == "bad_public":
                    rows[0]["model"] = "other"
                elif damage == "duplicate":
                    rows.extend(rows[:1])
                elif damage == "missing_pair":
                    rows.pop()
                elif damage == "bad_role":
                    rows[0]["last_role"] = "user"
                else:
                    rows[0]["start"] = 10
                with self.assertRaises(ValueError):
                    replay(rows)
