"""Write the M5 Z soup spec (amendment M5-2): every seed of T, V2 and each guard-passing M5 arm.

python3 m5_zspec.py <output-spec.json> <arm> [<arm> ...]
"""

import json
import sys

A = "/data/dev2/runs/06b/m1/arms"
recipes = ["m4-t-a7", "m4-v2"] + sys.argv[2:]
ingredients = []
for recipe in recipes:
    for s in ("s1", "s2", "s3"):
        best = json.load(open(f"{A}/{recipe}-{s}/full/BEST.json"))
        ingredients.append(
            {
                "arm": f"{recipe}-{s}",
                "run": f"/runs/m1/arms/{recipe}-{s}/full",
                "state_sha256": best["state_sha256"],
            }
        )
spec = {
    "arm": "m5-z-soup",
    "family": "qwen-causal",
    "recipe": "/src/src/training/decision2/v2/06b/records/arms/m4-t-a7-s1.json",
    "ingredients": ingredients,
    "allowed_differences": [
        "arm",
        "seed",
        "data.mixture",
        "teacher",
        "start.collapse_stop",
        "gpu_hour_cap",
        "max_reserved_gib",
    ],
    "export": True,
}
with open(sys.argv[1], "x") as stream:
    json.dump(spec, stream, indent=2)
    stream.write("\n")
print(len(ingredients), "ingredients:", ", ".join(recipes))
