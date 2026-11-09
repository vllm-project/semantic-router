"""Build vela3-spec.json from the release packages' own files in pkg_src/ (fetched from the Hub at the pinned revisions in
README.md (README-vela3.md in the research kit)). Adapter ranks and first layers come from vela3_config.json; the vela3-fast unfrozen top layers and the d3
adapter metadata were read from the adapter safetensors headers (recorded in ADAPTER_HEADERS below, see README.md (README-vela3.md in the research kit)).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "pkg_src"
REV = {
    "vela3-fast": "68a57a3779cac946370d640a388d6ada60940dab",
    "vela3-fast-vision": "e2b12d698f471a7e7d41c71cb2fbda7012b83f6f",
    "vela3-lite": "3e4f02e4134dd4c16e098561118c652081ace741",
    "vela3-mini": "f1c73f5d1b63b4decc4883d8afa78df7c1d1fdaa",
    "vela3-flash": "0e0452e01a0c1a3b818e224d3af2671b105ea9a2",
    "vela3-turbo": "39af842a9cdae46c8feb79b3b52d3ac6b78979e8",
}
SIZE = {
    "vela3-lite": "0.8B",
    "vela3-mini": "4B",
    "vela3-flash": "9B",
    "vela3-turbo": "27B",
}
# safetensors headers of the vela3-fast adapters (unfreeze_top, lora_rank / lora_alpha), read 25 Oct 2026 on the node
ADAPTER_HEADERS = {
    "span": (4, 64, 128),
    "span_sets": (4, 64, 128),
    "evidence": (4, 64, 128),
    "ner": (4, 128, 256),
    "relation": (4, 128, 256),
    "router": (8, 64, 128),
    "factcheck": (8, 64, 128),
    "feedback_df": (8, 64, 128),
    "domain": (12, 64, 128),
}
NICE = {
    "span": "Span LoRA",
    "ner": "NER LoRA",
    "relation": "Relation LoRA",
    "image_halu": "Image-halu LoRA",
    "decision": "Decision LoRA (router ids)",
    "decision_df": "Domain / feedback LoRA",
    "decision_hz": "Hazard LoRA",
    "decision_fc": "Fact-check LoRA",
    "decision_optin": "Opt-in decision LoRA",
}


def j(p):
    return json.loads(p.read_text())


def main():
    spec = {
        "note": "static config / header inspection; no weights loaded",
        "revisions": REV,
    }
    c = j(P / "vela3-fast/config.json")
    wins = c["attention_windows"]
    spec["vela3-fast"] = {
        "hidden": c["hidden_size"],
        "layers": c["num_hidden_layers"],
        "ffn": c["intermediate_size"],
        "heads": c["num_attention_heads"],
        "kv_heads": c["num_key_value_heads"],
        "head_dim": c["head_dim"],
        "vocab": c["vocab_size"],
        "vocab_base": c["decision"]["n_vocab_base"],
        "markers": len(c["decision"]["marker_ids"]),
        "proj_dim": c["decision"]["proj_dim"],
        "max_len": c["decision"]["max_len"],
        "window": max(w for w in wins if w),
        "global_layers": [i + 1 for i, w in enumerate(wins) if w is None],
        "rope_local": int(min(c["rope_theta"])),
        "rope_global": int(max(c["rope_theta"])),
        "adapters": ADAPTER_HEADERS,
    }
    v = j(P / "vela3-fast-vision/config.json")
    tc, vc = v["text_config"], v["vision_config"]
    vv = j(P / "vela3-fast-vision/vela3v_config.json")
    spec["vela3-fast-vision"] = {
        "text": {
            "hidden": tc["hidden_size"],
            "layers": tc["num_hidden_layers"],
            "ffn": tc["intermediate_size"],
            "heads": tc["num_attention_heads"],
            "head_dim": tc["head_dim"],
            "window": tc["sliding_window"],
            "max_pos": tc["max_position_embeddings"],
        },
        "vision": {
            "layers": vc["num_hidden_layers"],
            "hidden": vc["hidden_size"],
            "heads": vc["num_attention_heads"],
            "patch": vc["patch_size"],
            "tile": vc["image_size"],
            "edge": vv["tasks"]["decision"]["edge"],
            "tokens_1024": 320,
            "pixel_shuffle": v["pixel_shuffle_factor"],
        },
        "decision_state_tokens": vv["tasks"]["decision"]["max_state_tokens"],
        "tasks": {
            k: {"backbone": a.get("backbone"), "r": a.get("lora_rank")}
            for k, a in vv["tasks"].items()
        },
    }
    for n in ("vela3-lite", "vela3-mini", "vela3-flash", "vela3-turbo"):
        c = j(P / n / "config.json")
        t = c["text_config"]
        vc = c["vision_config"]
        vj = j(P / n / "vela3_config.json")
        order = [
            "decision",
            "decision_df",
            "decision_fc",
            "decision_hz",
            "decision_optin",
            "span",
            "ner",
            "relation",
            "image_halu",
        ]
        ads = [
            {
                "name": NICE[k],
                "r": vj["adapters"][k]["r"],
                "first": vj["adapters"][k]["first_layer"],
            }
            for k in order
            if k in vj["adapters"]
        ]
        spec[n] = {
            "size": SIZE[n],
            "base": vj["base"]["repo_id"].split("/")[1],
            "base_rev": vj["base"]["revision"],
            "hidden": t["hidden_size"],
            "layers": t["num_hidden_layers"],
            "ffn": t["intermediate_size"],
            "q_heads": t["num_attention_heads"],
            "kv_heads": t["num_key_value_heads"],
            "head_dim": t["head_dim"],
            "lin_k": t["linear_num_key_heads"],
            "lin_v": t["linear_num_value_heads"],
            "lin_dim": t["linear_key_head_dim"],
            "vocab": t["vocab_size"],
            "vision_depth": vc["depth"],
            "vision_hidden": vc["hidden_size"],
            "adapters": ads,
            "definitions": vj["definitions"]["writer"],
        }
    (ROOT / "vela3-spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "adapters"}
                for k, v in spec.items()
                if k.startswith("vela3")
            },
            indent=0,
        )[:3000]
    )


if __name__ == "__main__":
    main()
