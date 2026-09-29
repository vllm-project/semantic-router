"""State-length and question-type profile of gold-free formal panels (aggregates only).

Stdlib only; run on node A via stdin. Arg: JSON {"panels": {name: goldfree_prompts_path}}.
Reads gold-free prompts only (no gold files).
"""

import json
import sys
from collections import Counter


def slen(state):
    return len(
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    )


def main():
    args = json.loads(sys.argv[1])
    out = {}
    for name, path in args["panels"].items():
        lens, types, dict_states = [], Counter(), 0
        for line in open(path):
            p = json.loads(line)
            lens.append(slen(p["state"]))
            dict_states += isinstance(p["state"], dict)
            for q in p["questions"].values():
                types[q.get("type")] += 1
        lens.sort()
        n = len(lens)
        out[name] = {
            "prompts": n,
            "question_types": dict(types),
            "dict_states": dict_states,
            "state_chars_median": lens[n // 2],
            "state_chars_p90": lens[int(n * 0.9)],
            "state_chars_max": lens[-1],
            "over_2000_chars": sum(x > 2000 for x in lens),
            "over_4000_chars": sum(x > 4000 for x in lens),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
