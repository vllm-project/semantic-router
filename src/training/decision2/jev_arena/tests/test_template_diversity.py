"""The authored diversity audit distinguishes scenarios from templates."""

import unittest

from jev_arena.sealed_authored import CHALLENGES, DOMAINS, TYPES, make_spec, render
from jev_arena.template_diversity import choose_sample, semantic_signature


class TemplateDiversityTest(unittest.TestCase):
    def test_domain_words_do_not_create_new_logical_policy(self):
        seed = bytes(range(32))
        signatures = set()
        for domain in DOMAINS:
            spec = make_spec(seed, "release", "choice", "near_distractor", domain, 0)
            prompt, _ = render(spec)
            signatures.add(semantic_signature(prompt, spec))
        self.assertEqual(len(signatures), 1)

    def test_sample_covers_all_families_and_domains(self):
        seed = bytes(range(32))
        targets = []
        for kind in TYPES:
            for challenge in CHALLENGES:
                for domain in DOMAINS:
                    for ordinal in range(2):
                        spec = make_spec(seed, "dev", kind, challenge, domain, ordinal)
                        _, target = render(spec)
                        targets.append(target)
        sample = [targets[index] for index in choose_sample(targets)]
        self.assertEqual(len(sample), 20)
        self.assertEqual(len({row["family"] for row in sample}), 12)
        self.assertEqual({row["domain"] for row in sample}, set(DOMAINS))


if __name__ == "__main__":
    unittest.main()
