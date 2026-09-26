"""Gold alignment when projecting frozen typed rows to native Kai training."""

from __future__ import annotations

import unittest

from training.data.build_kai06b_native_v1 import convert


class KaiNativeProjectionTest(unittest.TestCase):
    def _project(self, *, kind: str, options: list[dict], label: int):
        observed = {}

        def native_converter(request, targets, **kwargs):
            observed.update(request=request, targets=targets, kwargs=kwargs)
            return [{"id": "native-0"}]

        row = {
            "id": "source-0",
            "source": "source",
            "group_id": "group-0",
            "state": {"fact": "given"},
            "instructions": "Decide from the fact.",
            "task_type": kind,
            "options": options,
            "label": label,
        }
        result, fallback = convert(row, native_converter)
        self.assertEqual(result["id"], "native-0")
        self.assertEqual(observed["kwargs"]["component_id"], "group-0")
        self.assertEqual(observed["request"]["state"], row["state"])
        return observed, fallback

    def test_reversed_noul_keys_keep_yes_gold(self) -> None:
        observed, fallback = self._project(
            kind="noul",
            options=[
                {"key": "true", "description": "Yes"},
                {"key": "false", "description": "No"},
            ],
            label=0,
        )
        self.assertEqual(observed["targets"]["decision"], {"probability": 1.0})
        self.assertEqual(observed["kwargs"]["hard_target_ids"], {"decision": "yes"})
        self.assertFalse(fallback)

    def test_score_keeps_order_and_explicit_null_fallback(self) -> None:
        observed, fallback = self._project(
            kind="score",
            options=[
                {"key": "low", "description": None},
                {"key": "middle", "description": "mid"},
                {"key": "high", "description": "top"},
            ],
            label=2,
        )
        self.assertEqual(
            observed["request"]["questions"]["decision"]["criteria"],
            ["low", "mid", "top"],
        )
        self.assertEqual(
            observed["targets"]["decision"], {"probabilities": [0.0, 0.0, 1.0]}
        )
        self.assertEqual(observed["kwargs"]["hard_target_ids"], {"decision": "2"})
        self.assertTrue(fallback)


if __name__ == "__main__":
    unittest.main()
