import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import framing

HOOKS = Path(__file__).resolve().parents[3] / "hooks"


def ts(text: str, quote: str = "'") -> str:
    """A string as the hooks spell it: a quoted TypeScript literal."""
    return quote + text.replace("\\", "\\\\").replace("\n", "\\n") + quote


def strings(question: dict) -> list[str]:
    """Every instructions string, criterion, choice key and choice description of a question."""
    found = [question["instructions"], *question.get("criteria", ())]
    for choice in question.get("choices", ()):
        found += [choice["key"], choice["description"]]
    return found


class CopiedFromTheHooks(unittest.TestCase):
    """framing.py copies the hooks' questions; a change on either side alone fails here."""

    def assert_in(self, source: str, copied: list[str], quote: str = "'"):
        text = (HOOKS / source).read_text()
        for one in copied:
            with self.subTest(source=source, copied=one[:60]):
                self.assertIn(ts(one, quote), text)

    def test_questions(self):
        self.assert_in(
            "router.ts", strings(framing.COMPLEXITY) + strings(framing.STEPS)
        )
        self.assert_in(
            "escalate.ts", strings(framing.USER_FEEDBACK) + strings(framing.REASK)
        )
        self.assert_in(
            "guard.ts",
            [s for q in framing.GUARD_QUESTIONS.values() for s in strings(q)],
        )
        self.assert_in("output.ts", strings(framing.JAILBREAK))
        self.assert_in("check.ts", strings(framing.HALLUCINATION))

    def test_framings(self):
        self.assert_in("guard.ts", [framing.GUARD_FRAME], quote="`")
        self.assert_in("output.ts", [framing.SHIELD_FRAME], quote="`")

    def test_pii_question_and_tags(self):
        text = (HOOKS / "lib" / "redact.ts").read_text()
        preset = re.search(r"PII_QUESTIONS = (\{.*\})", text).group(1)
        self.assertEqual(
            json.loads(re.sub(r"(\w+):", r'"\1":', preset).replace("'", '"')),
            framing.PII_QUESTIONS,
        )
        tags = re.search(r"LABEL_TAGS[^=]*= \{(.*?)\n\}", text, re.S).group(1)
        self.assertEqual(
            dict(re.findall(r"(\w+): '(\w+)'", tags)),
            framing.PII_TAGS,
        )


class Framed(unittest.TestCase):
    def test_the_router_probe_asks_both_questions_in_one_request_as_the_hook_does(self):
        state, questions = framing.framed("router", "Refactor login")
        self.assertEqual(state, "Refactor login")
        self.assertEqual(set(questions), {"complexity", "steps"})

    def test_the_framings_fill_in_the_input(self):
        self.assertEqual(
            framing.framed("guard", "git status")[0], "Tool: Bash\nCommand: git status"
        )
        self.assertEqual(
            framing.framed("shield", {"source": "a.md", "text": "${x}"})[0],
            "Contents of a.md:\n${x}",
        )


if __name__ == "__main__":
    unittest.main()
