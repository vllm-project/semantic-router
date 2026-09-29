"""Planted-quote flag spot-check (node A, private reading only).

Draws 15 flagged and 15 unflagged hard items (seeded) and prints, for flagged
items, the regex match with context; for unflagged items, every quoted span and
the state tail, so a reader can judge whether a person-attributed conclusion
was missed. Nothing printed here is copied into the worktree.
"""

import json
import random
import re
import sys

ARGS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
PRIV = "/data/dev2/private/eval/jevbench-value/gap"
PROMPTS = "/data/dev2/private/panels/goldfree/public231.prompts.jsonl"
CUE = re.compile(
    r"(note|comment|remark|draft|recommend\w*|message|e-mail|email|chat|says|said|"
    r"screener|planner|approver|supervisor|reviewer|clerk|analyst|manager|lead)"
    r"[^\"\u201c\n]{0,80}?[:,]\s*[\"\u201c]",
    re.I,
)
QUOTE = re.compile(r"[\"\u201c][^\"\u201d]{15,300}[\"\u201d]")

prompts = {p["id"]: p for p in (json.loads(x) for x in open(PROMPTS))}
flags = json.load(open(PRIV + "/cue_flags.json"))
hard = sorted(i for i in flags if i.startswith("hard"))
rnd = random.Random(ARGS.get("seed", 20260929))
fl = rnd.sample([i for i in hard if flags[i]], 15)
un = rnd.sample([i for i in hard if not flags[i]], 15)
json.dump({"flagged": fl, "unflagged": un}, open(PRIV + "/cue_spot_sample.json", "w"))
part = ARGS.get("part", "flagged")
for i in fl if part == "flagged" else un:
    st = prompts[i]["state"]
    s = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
    print("=" * 90)
    print(i, "chars", len(s))
    if part == "flagged":
        for m in list(CUE.finditer(s))[:3]:
            print(
                "  MATCH:", s[max(0, m.start() - 60) : m.end() + 260].replace("\n", " ")
            )
    else:
        if isinstance(st, str):
            for m in list(QUOTE.finditer(s))[:6]:
                print(
                    "  QUOTE:", s[max(0, m.start() - 80) : m.end()].replace("\n", " ")
                )
        print("  TAIL:", s[-ARGS.get("tail", 500) :].replace("\n", " "))
