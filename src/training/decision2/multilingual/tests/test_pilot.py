import unittest

from multilingual.pilot import make_rows


class PilotTest(unittest.TestCase):
    def test_pairs_and_gold_free_prompts(self):
        prompts, targets, counts = make_rows()
        self.assertEqual(len(prompts), 210)
        self.assertEqual(counts["semantic_groups"], 18)
        self.assertEqual(set(counts["by_language"].values()), {30})
        self.assertEqual(counts["by_type"], {"choice": 84, "noul": 84, "score": 42})
        self.assertTrue(all("gold" not in prompt for prompt in prompts))
        by_id = {row["id"]: row for row in targets}
        prompt_by_id = {row["id"]: row for row in prompts}
        self.assertEqual(len(by_id), 210)
        for row in targets:
            if row["task_type"] == "choice":
                self.assertEqual(
                    row["semantic_by_label"][row["gold"]], row["semantic_gold"]
                )
        base = prompt_by_id["mldev-noul-shipped-zh-base"]["questions"]["decision"][
            "criteria"
        ]
        restyled = prompt_by_id["mldev-noul-shipped-zh-label_style"]["questions"][
            "decision"
        ]["criteria"]
        self.assertNotEqual(base, restyled)


if __name__ == "__main__":
    unittest.main()
