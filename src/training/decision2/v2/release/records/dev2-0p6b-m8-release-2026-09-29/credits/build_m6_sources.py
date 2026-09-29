"""Build m6-sources.json: per-upstream credits for the M6 training union (local, no network).

Inputs: m6-union-counts.json (from count_m6_sources.py on node A), the licence registries under
v2/data, the M6 records and the current DEV2.0-0.6B release spec. Fails on any inconsistency.
"""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[3]
REPO = V2.parents[3]
COUNTS = HERE / "m6-union-counts.json"
REGISTRIES = [
    V2 / "data/a7/license-registry-a7-v2.json",
    V2 / "data/records/license-registry-v2.json",
    V2 / "data/records/license-registry-m3b.json",
    V2 / "data/records/license-registry-v1.json",
    V2 / "data/a7/license-registry-a7-v1.json",
]
M6 = V2 / "06b/records"
VERIFY = M6 / "m6/m6-verify.json"
MIXTURES = [
    M6 / f"mixtures/m6-{f}-s{k}.{x}json"
    for f in ("cx", "mx")
    for k in (1, 2, 3)
    for x in ("", "report.")
]
RECORDS = [
    M6 / "m6-prereg-2026-09-29.md",
    M6 / "m6-results-2026-09-29.md",
    M6 / "m8-results-2026-09-29.md",
]
SPEC = V2 / "release/specs/dev2-0p6b-release.json"
LICENCE_PY = V2 / "release/licence.py"
LUX = "[Decision 1.0 Lux-9B](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B)"
WIKI = "wikipedia-text-cc-by-sa"
SA = "share-alike"

# upstream id, card name, citation, card licence, licence id, source keys (#part = split of a key), flags
UPSTREAM = [
    (
        "goemotions",
        "GoEmotions",
        "Demszky et al., ACL 2020",
        "CC BY 4.0",
        "cc-by-4.0",
        ["google_goemotions_official_train"],
        [],
    ),
    (
        "snli",
        "SNLI",
        "Bowman et al., EMNLP 2015",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        ["dec10:snli_train", "legacy:snli"],
        [SA],
    ),
    (
        "multinli-nonfiction",
        "MultiNLI non-fiction genres",
        "Williams, Nangia and Bowman, NAACL 2018",
        "OANC terms",
        "LicenseRef-OANC",
        ["dec10:multinli_nonfiction_train"],
        ["non-spdx-terms"],
    ),
    (
        "clinc150",
        "CLINC150",
        "Larson et al., EMNLP-IJCNLP 2019",
        "CC BY 3.0",
        "cc-by-3.0",
        ["dec10:clinc150_train", "legacy:stage3_replay#clinc150"],
        [],
    ),
    (
        "banking77",
        "BANKING77",
        "Casanueva et al., 2020",
        "CC BY 4.0",
        "cc-by-4.0",
        ["dec10:banking77_train", "legacy:stage3_replay#banking77"],
        [],
    ),
    (
        "klue",
        "KLUE STS, MRC and YNAT",
        "Park et al., NeurIPS 2021 Datasets and Benchmarks",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        [
            "klue_sts_train",
            "a7:klue_sts_v1.1_train",
            "klue_mrc_train",
            "klue_ynat_train",
        ],
        [SA],
    ),
    (
        "jglue",
        "JGLUE JSTS, JSQuAD, JCommonsenseQA and JNLI",
        "Kurihara et al., LREC 2022",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        [
            "jglue_jsts_v1.3_train",
            "a7:jglue_jsts_v1.3_train",
            "jsquad_v1.3_train",
            "jglue_jcommonsenseqa_v1.3_train",
            "jglue_jnli_v1.3_train",
        ],
        [SA],
    ),
    (
        "argq30k",
        "IBM ArgQ-30k",
        "Gretz et al., AAAI 2020",
        "CC BY-SA 3.0",
        "cc-by-sa-3.0",
        ["argq30k_train"],
        [SA],
    ),
    (
        "saf",
        "SAF",
        "Filighera et al., ACL 2022",
        "CC BY 4.0",
        "cc-by-4.0",
        ["saf_en_train", "saf_de_train"],
        [],
    ),
    (
        "oasst1",
        "OpenAssistant OASST1",
        "Köpf et al., NeurIPS 2023 Datasets and Benchmarks",
        "Apache-2.0",
        "apache-2.0",
        ["a7:oasst1_train"],
        [],
    ),
    (
        "sentimix-hinglish",
        "SentiMix Hinglish",
        "Patwa et al., SemEval 2020",
        "CC BY 4.0",
        "cc-by-4.0",
        ["a7:sentimix_hinglish_train"],
        ["social-media-text"],
    ),
    (
        "afrisenti-sw",
        "AfriSenti Swahili",
        "Muhammad et al., EMNLP 2023",
        "CC BY 4.0",
        "cc-by-4.0",
        ["a7:afrisenti_sw_train"],
        ["social-media-text"],
    ),
    (
        "tydiqa",
        "TyDi QA",
        "Clark et al., TACL 2020",
        "Apache-2.0, Wikipedia text CC BY-SA",
        "apache-2.0",
        ["tydiqa_primary_train"],
        [WIKI],
    ),
    (
        "winogrande",
        "WinoGrande",
        "Sakaguchi et al., AAAI 2020",
        "Apache-2.0",
        "apache-2.0",
        ["winogrande_xl_train"],
        [],
    ),
    ("gsm8k", "GSM8K", "Cobbe et al., 2021", "MIT", "mit", ["gsm8k_train"], []),
    (
        "mlqe-pe",
        "MLQE-PE",
        "Fomicheva et al., LREC 2022",
        "CC0 1.0, Wikipedia text CC BY-SA",
        "cc0-1.0",
        ["mlqepe_train"],
        [WIKI],
    ),
    (
        "2wikimultihopqa",
        "2WikiMultihopQA",
        "Ho et al., COLING 2020",
        "Apache-2.0, Wikipedia text CC BY-SA",
        "apache-2.0",
        ["twowiki_train"],
        [WIKI],
    ),
    (
        "hotpotqa",
        "HotpotQA",
        "Yang et al., EMNLP 2018",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        ["hotpotqa_distractor_train"],
        [SA],
    ),
    (
        "commonsenseqa",
        "CommonsenseQA",
        "Talmor et al., NAACL 2019",
        "MIT",
        "mit",
        ["csqa_train"],
        [],
    ),
    (
        "squad2",
        "SQuAD 2.0",
        "Rajpurkar, Jia and Liang, ACL 2018",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        ["squad2_train"],
        [SA],
    ),
    (
        "multiwoz22",
        "MultiWOZ 2.2",
        "Zang et al., 2020",
        "MIT",
        "mit",
        ["multiwoz22_train"],
        [],
    ),
    (
        "scitail",
        "SciTail",
        "Khot et al., AAAI 2018",
        "Apache-2.0",
        "apache-2.0",
        ["scitail_train"],
        [],
    ),
    (
        "musique",
        "MuSiQue",
        "Trivedi et al., TACL 2022",
        "CC BY 4.0, Wikipedia text CC BY-SA",
        "cc-by-4.0",
        ["musique_full_v1.0_train"],
        [WIKI],
    ),
    (
        "quac",
        "QuAC",
        "Choi et al., EMNLP 2018",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        ["quac_train_v0.2"],
        [SA, "licence-metadata-discrepancy"],
    ),
    (
        "drcd",
        "DRCD",
        "Shao et al., 2018",
        "CC BY-SA 3.0",
        "cc-by-sa-3.0",
        ["drcd_train"],
        [SA],
    ),
    (
        "sqac",
        "SQAC",
        "Gutiérrez-Fandiño et al., 2022",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        ["sqac_train"],
        [SA],
    ),
    (
        "cmrc2018",
        "CMRC 2018",
        "Cui et al., EMNLP 2019",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        ["cmrc2018_train"],
        [SA],
    ),
    ("abcd", "ABCD", "Chen et al., NAACL 2021", "MIT", "mit", ["abcd_v1.1_train"], []),
    (
        "germanquad",
        "GermanQuAD",
        "Möller et al., 2021",
        "CC BY 4.0",
        "cc-by-4.0",
        ["germanquad_train"],
        [],
    ),
    ("piaf", "PIAF", "Keraron et al., LREC 2020", "MIT", "mit", ["piaf_train"], []),
    (
        "dbpedia14",
        "DBpedia-14",
        "Zhang et al., NeurIPS 2015",
        "CC BY-SA 3.0",
        "cc-by-sa-3.0",
        ["dbpedia14_train"],
        [SA],
    ),
    (
        "mtop",
        "MTOP",
        "Li et al., EACL 2021",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        ["mtop_train"],
        [SA],
    ),
    (
        "taskmaster2",
        "Taskmaster-2",
        "Byrne et al., Google 2020",
        "CC BY 4.0",
        "cc-by-4.0",
        ["taskmaster2_train"],
        [],
    ),
    (
        "ropes",
        "ROPES",
        "Lin et al., MRQA 2019",
        "CC BY 4.0",
        "cc-by-4.0",
        ["ropes_train"],
        [],
    ),
    (
        "sgd",
        "Schema-Guided Dialogue",
        "Rastogi et al., AAAI 2020",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        ["sgd_dstc8_train"],
        [SA],
    ),
    (
        "quartz",
        "QuaRTz",
        "Tafjord et al., EMNLP 2019",
        "CC BY 4.0",
        "cc-by-4.0",
        ["quartz_train"],
        [],
    ),
    (
        "onestopenglish",
        "OneStopEnglish",
        "Vajjala and Lučić, BEA 2018",
        "CC BY-SA 4.0",
        "cc-by-sa-4.0",
        ["onestop_english"],
        [SA],
    ),
]
GENERATED = [
    "dec10:generated_stage1_3",
    "dec10:generated_stage4_v2",
    "legacy:stage4-general-composition-v2",
    "legacy:stage3_replay#project-generated",
    "decision2_verifiable_v2_a2",
    "decision2_verifiable_v2_a6",
    "decision2_verifiable_v2_a4v2h",
    "decision2_targeted_programmatic_v1",
    "decision2_programmatic_original_v1",
]
MUST_BE_ABSENT = [
    "a7:massive_1.1_train",
    "css_flute_official_train",
    "dec10:cosmos_qa_train",
    "dec10:squad2_train",
    "legacy:cosmos_qa",
    "legacy:squad2_answerability",
]
CHECK_KEYS = [
    "id_not_in_either_recipe",
    "recipe_pool_or_source_conflict",
    "same_id_differs_across_files",
    "source_differs_from_recipe",
    "teacher_ids_outside_train_union",
    "teacher_input_sha256_mismatch",
    "teacher_shared_id_differs",
    "train_union_rows_without_teacher",
]
CURRENT_CARD = {
    "GoEmotions": "goemotions",
    "SNLI": "snli",
    "MultiNLI non-fiction genres": "multinli-nonfiction",
    "CLINC150": "clinc150",
    "BANKING77": "banking77",
    "KLUE STS": "klue",
    "JGLUE JSTS": "jglue",
    "IBM ArgQ-30k": "argq30k",
    "SAF": "saf",
}
NC = re.compile(r"\bNC\b|non-?commercial", re.I)
RESEARCH = re.compile(
    r"research[- ]only|research use only|only for (nlp )?research", re.I
)
RESTRICTED = re.compile(r"not permitted|no redistribution|(?<!un)restricted", re.I)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(REPO))


def need(cond, msg):
    if not cond:
        raise SystemExit(f"check failed: {msg}")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def registry_entry(registries, key):
    base = key.split("#")[0]
    for path, reg in registries:
        if base in reg:
            return {"file": rel(path), "key": base, **reg[base]}
    raise SystemExit(f"no registry entry for {base}")


def licence_matches(licence_id, entry, part):
    text = entry["license"]
    if licence_id == "LicenseRef-OANC":
        return "oanc" in text.lower()
    if part:
        return norm(licence_id) in norm(text)
    return norm(text).startswith(norm(licence_id))


def policy_flags(entries):
    text = " ".join(f"{e['license']} {e.get('redistribution', '')}" for e in entries)
    return {
        "non_commercial": bool(NC.search(text)),
        "research_only": bool(RESEARCH.search(text)),
        "unknown_licence": any(
            not e.get("license") or "unknown" in e["license"].lower() for e in entries
        ),
        "restricted_redistribution": bool(RESTRICTED.search(text)),
    }


def main():
    counts = json.loads(COUNTS.read_text(encoding="utf-8"))
    verify = json.loads(VERIFY.read_text(encoding="utf-8"))
    registries = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in REGISTRIES]
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parts = counts["parts"]
    checks = {k: counts["checks"].get(k, 0) for k in CHECK_KEYS}
    checks["teacher_ids_in_both_files"] = counts["checks"].get(
        "teacher_ids_in_both_files", 0
    )

    union = counts["union"]
    need(
        union["distinct_ids"] == verify["native_conversion"]["rows"],
        "union rows vs m6-verify native_conversion",
    )
    need(
        union["distinct_ids"] == union["distinct_input_sha256"],
        "id and input_sha256 are 1:1",
    )
    need(
        union["task_types"] == verify["native_conversion"]["task_types"],
        "task types vs m6-verify",
    )
    for fam, n in union["family_union_ids"].items():
        need(
            n == verify["families"][fam]["union_rows"], f"{fam} union rows vs m6-verify"
        )
        for pool, v in verify["families"][fam]["union_by_pool"].items():
            need(
                counts["pools"][pool][fam] == v["rows"],
                f"{fam} {pool} union rows vs m6-verify",
            )
    for f in counts["files"]:
        name = f["file"].split(".")[0]
        report = json.loads(
            (M6 / f"mixtures/{name}.report.json").read_text(encoding="utf-8")
        )
        need(
            f["sha256"]
            == verify["mixtures"][name]["sha256"]
            == report["rows_file_sha256"],
            f"{name} sha256",
        )
        need(
            f["rows"]
            == f["distinct_ids"]
            == report["train_rows"]
            == verify["per_seed"][name]["rows"],
            f"{name} rows",
        )
    for fam, t in counts["teachers"].items():
        need(t["sha256"] == verify["teachers"][fam]["sha256"], f"{fam} teacher sha256")
        need(
            t["entries"] == t["distinct_ids"] == verify["teachers"][fam]["entries"],
            f"{fam} teacher entries",
        )
        need(t["ids_equal_family_union"], f"{fam} teacher ids equal the family union")
    need(
        counts["other_inputs"]["m6-verify.json"] == sha256(VERIFY),
        "node A m6-verify.json equals the record",
    )
    need(
        all(v == 0 for k, v in checks.items() if k != "teacher_ids_in_both_files"),
        f"row checks {checks}",
    )
    need(
        checks["teacher_ids_in_both_files"] == union["ids_in_both_families"],
        "shared teacher ids",
    )
    need(
        sum(p["rows_union"] for p in parts.values()) == union["distinct_ids"],
        "parts sum to the union",
    )
    need(
        sum(p["rows_per_file_sum"] for p in parts.values())
        == union["rows_per_file_sum"],
        "per-file sums",
    )
    for key in MUST_BE_ABSENT:
        need(key not in parts, f"{key} must have no rows")
    mapped = [k for u in UPSTREAM for k in u[5]] + GENERATED
    need(
        sorted(mapped) == sorted(parts),
        f"unmapped or unknown keys: {sorted(set(parts) ^ set(mapped))}",
    )

    def aggregate(keys):
        out = {
            "rows_union_distinct": 0,
            "rows_per_file_sum": 0,
            "rows_by_family_union": {"m6-cx": 0, "m6-mx": 0},
            "arms": {},
            "task_types": {},
            "languages": {},
        }
        for k in keys:
            p = parts[k]
            out["rows_union_distinct"] += p["rows_union"]
            out["rows_per_file_sum"] += p["rows_per_file_sum"]
            for fam, n in p["rows_family_union"].items():
                out["rows_by_family_union"][fam] += n
            for field, src in (
                ("arms", "pools"),
                ("task_types", "task_types"),
                ("languages", "languages"),
            ):
                for name, n in p[src].items():
                    out[field][name] = out[field].get(name, 0) + n
        for field in ("arms", "task_types", "languages"):
            out[field] = dict(
                sorted(out[field].items(), key=lambda kv: (-kv[1], kv[0]))
            )
        out["source_keys"] = {k: parts[k]["rows_union"] for k in keys}
        return out

    sources = []
    for uid, name, citation, card_licence, licence_id, keys, flags in UPSTREAM:
        entries = [registry_entry(registries, k) for k in keys]
        for k, e in zip(keys, entries):
            need(
                licence_matches(licence_id, e, "#" in k),
                f"{k}: registry licence {e['license']!r} vs {licence_id}",
            )
        pol = policy_flags(entries)
        need(not any(pol.values()), f"{uid}: policy flag {pol}")
        sources.append(
            {
                "upstream": uid,
                "name": name,
                "kind": "third-party",
                "citation": citation,
                "licence": licence_id,
                "licence_card": card_licence,
                "flags": {**pol, "informational": flags},
                "card_item": f"{name} ({citation}; {card_licence})",
                **aggregate(keys),
                "registry": entries,
            }
        )
    sources.sort(key=lambda s: (-s["rows_union_distinct"], s["name"]))
    generated_entries = [registry_entry(registries, k) for k in GENERATED]
    for k, e in zip(GENERATED, generated_entries):
        need(
            "project-generated" in e["license"],
            f"{k} is not registered as project-generated",
        )
    generated = {
        "upstream": "project-generated",
        "name": "project-generated decision tasks",
        "kind": "project-generated",
        "citation": None,
        "licence": "project-generated (no third-party text)",
        "licence_card": None,
        "flags": {
            **policy_flags([{"license": "project-generated", "redistribution": ""}]),
            "informational": [],
        },
        "card_item": "project-generated decision tasks",
        **aggregate(GENERATED),
        "registry": generated_entries,
    }
    third_rows = sum(s["rows_union_distinct"] for s in sources)
    need(
        third_rows + generated["rows_union_distinct"] == union["distinct_ids"],
        "third-party + generated = union",
    )

    spec_mod = importlib.util.spec_from_file_location("release_licence", LICENCE_PY)
    licence = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(licence)
    components = spec["licence"]["components"]
    package = licence.package_licence(components)

    languages = union["languages"]
    zh = languages.get("zh", 0) + languages.get("zh-hant", 0)
    named = {"en", "zh", "zh-hant", "hi-en", "es", "ja", "ko"}
    others = len([k for k in languages if k not in named])
    training_attr = (
        "Training data (not redistributed here; each keeps its own licence): "
        + ", ".join(s["card_item"] for s in sources)
        + ", and project-generated decision tasks."
    )
    lux_attr = (
        f"{LUX} (Apache-2.0), our own model: its probabilities on all {union['distinct_ids']:,} training prompts "
        "were soft training targets. None of its weights are included."
    )
    tt = union["task_types"]
    paragraph = (
        "**Recipe and data:** the weights are the uniform average of six full fine-tunes of Qwen3-0.6B-Base, "
        "two data recipes (with and without the data v2 arms) times three seeds, each trained for one epoch on "
        "its own 30M-token mixture (batches of 16 rows; backbone learning rate 1e-5 with 10% warm-up and cosine "
        "decay, head 2e-4) with cross-entropy plus 0.5 × Brier and a KL term (weight 1.0) toward soft targets "
        f"from our own {LUX} on every row; no other teacher, and no outputs of Jev or of any third-party decision "
        f"model. Together the six mixtures hold {union['distinct_ids']:,} distinct decision rows: Choice "
        f"{tt['choice']:,}, Noul {tt['noul']:,} and Score {tt['score']:,}; English {languages['en']:,}, Chinese "
        f"{zh:,}, Hinglish {languages['hi-en']:,}, Spanish {languages['es']:,}, Japanese {languages['ja']:,}, "
        f"Korean {languages['ko']:,} and {others} other languages. Of these, {generated['rows_union_distinct']:,} "
        "are program-generated tasks labeled by program oracles (Decision 1.0 stage 1–4 and Decision 2.0 "
        f"programmatic and verifiable generators; no third-party text) and {third_rows:,} come from "
        f"{len(sources)} public datasets that keep their own licences ([attributions](ATTRIBUTIONS.md)). "
        "The recipes contain no MASSIVE, PAWS-X or XNLI "
        "rows (the sources of mlx-diag), and every group with an overlap hit on an evaluation panel was removed "
        "before training."
    )
    current = [
        a for a in spec["licence"]["attributions"] if a.startswith("Training data")
    ]
    need(len(current) == 1, "one training-data attribution in the current spec")
    current_ids = set(CURRENT_CARD.values())
    for label in CURRENT_CARD:
        need(label in current[0], f"current card lists {label}")
    by_id = {s["upstream"]: s for s in sources}
    added = [
        {
            "upstream": s["upstream"],
            "name": s["name"],
            "rows": s["rows_union_distinct"],
            "licence": s["licence"],
        }
        for s in sources
        if s["upstream"] not in current_ids
    ]
    widened = [
        {
            "upstream": "klue",
            "was": "KLUE STS",
            "now": by_id["klue"]["name"],
            "rows": by_id["klue"]["source_keys"],
        },
        {
            "upstream": "jglue",
            "was": "JGLUE JSTS",
            "now": by_id["jglue"]["name"],
            "rows": by_id["jglue"]["source_keys"],
        },
    ]

    local_inputs = [
        COUNTS,
        HERE / "count_m6_sources.py",
        Path(__file__).resolve(),
        VERIFY,
        SPEC,
        LICENCE_PY,
        *REGISTRIES,
        *MIXTURES,
        *RECORDS,
    ]
    out = {
        "schema": "dev2-release-credits/1",
        "date": "2026-09-29",
        "release": "DEV2.0-0.6B successor m8-s5-b05 (weights byte-identical to m6-mxcx-soup = six-seed soup of m6-cx and m6-mx)",
        "scope": "union of the six M6 seed train files m6-{cx,mx}-s{1,2,3}.train.jsonl (node A, /data/dev2/runs/06b/m6/data)",
        "row_basis": "distinct rows, keyed by id (id and input_sha256 are 1:1 in the union); rows_per_file_sum counts a row once per seed file that holds it",
        "union": {
            "distinct_rows": union["distinct_ids"],
            "distinct_inputs": union["distinct_input_sha256"],
            "distinct_groups": union["distinct_group_ids"],
            "rows_per_file_sum": union["rows_per_file_sum"],
            "family_union_rows": union["family_union_ids"],
            "rows_in_both_families": union["ids_in_both_families"],
            "task_types": union["task_types"],
            "languages": dict(
                sorted(languages.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
        },
        "summary": {
            "upstream_datasets": len(sources),
            "third_party_rows": third_rows,
            "project_generated_rows": generated["rows_union_distinct"],
            "non_commercial": [
                s["name"] for s in sources if s["flags"]["non_commercial"]
            ],
            "research_only": [
                s["name"] for s in sources if s["flags"]["research_only"]
            ],
            "unknown_licence": [
                s["name"] for s in sources if s["flags"]["unknown_licence"]
            ],
            "restricted_redistribution": [
                s["name"] for s in sources if s["flags"]["restricted_redistribution"]
            ],
            "informational_flags": {
                flag: {
                    s["name"]: s["rows_union_distinct"]
                    for s in sources
                    if flag in s["flags"]["informational"]
                }
                for flag in (
                    "non-spdx-terms",
                    "licence-metadata-discrepancy",
                    "social-media-text",
                    WIKI,
                    SA,
                )
            },
            "absent_source_keys": MUST_BE_ABSENT,
        },
        "teacher": {
            "model": "llm-semantic-router/Decision-1.0-Lux-9B (own; Apache-2.0)",
            "files": counts["teachers"],
            "entries_total": sum(t["entries"] for t in counts["teachers"].values()),
            "distinct_rows_with_target": union["distinct_ids"]
            - checks["train_union_rows_without_teacher"],
            "shared_ids_identical_input_and_probs": checks["teacher_ids_in_both_files"]
            - checks["teacher_shared_id_differs"],
        },
        "package_licence": {
            "rule": "v2/release/licence.py package_licence over the direct weight-lineage components; training data are attributions, not lineage components",
            "components": components,
            "result": {
                k: package[k] for k in ("spdx", "license_name", "apache_compatible")
            },
            "components_change_needed": False,
        },
        "sources": sources + [generated],
        "pools": counts["pools"],
        "source_key_parts": parts,
        "origin_labels": counts["origin_labels"],
        "checks": checks,
        "card": {
            "training_data_attribution": training_attr,
            "lux_attribution": lux_attr,
            "training_paragraph": paragraph,
        },
        "vs_current_card": {
            "current_training_data_attribution": current[0],
            "removed": sorted(set(current_ids) - set(by_id)),
            "added": added,
            "widened": widened,
            "lux_prompts": {"current": 6430, "new": union["distinct_ids"]},
        },
        "inputs": {
            "node_a": {
                "train_files": counts["files"],
                "recipe_ids": counts["recipes"],
                "teachers": {
                    k: {"file": v["file"], "sha256": v["sha256"]}
                    for k, v in counts["teachers"].items()
                },
                "other": counts["other_inputs"],
            },
            "local": {rel(p): sha256(p) for p in local_inputs},
        },
        "commands": [
            "cd src/training/decision2/v2/release/records/dev2-0p6b-m8-release-2026-09-29/credits",
            "B=$(base64 -w0 count_m6_sources.py) && ssh -o ConnectTimeout=20 -o BatchMode=yes root@<node A> "
            '"python3 -c \\"\\$(echo $B | base64 -d)\\"" > m6-union-counts.json',
            "python3 build_m6_sources.py",
        ],
    }
    (HERE / "m6-sources.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": out["summary"],
                "package": out["package_licence"]["result"],
                "teacher": {
                    k: out["teacher"][k]
                    for k in ("entries_total", "distinct_rows_with_target")
                },
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
