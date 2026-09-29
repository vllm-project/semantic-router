"""Data-lock checks for decoder M6 (prereg dec-m6-prereg-2026-09-29.md), host python from the mirror.

part1 (node B): m6-xl-full-59m (budget, excluded groups by id / row id / input hash, same inputs and A0s as
  m4-xl-full-29m, nested over it, C1 registry names, component token breakdown), the own-Lux teachers
  lux-all-29m / lux-all-59m (coverage, unchanged N4XF targets, 59m == 29m on shared rows), m6-e8f-r2clean
  (removals by reason, rows otherwise byte-identical to E8F's, A7 v3 files against their registry) and
  both exposure receipts (0 groups).
sol (node B): own-Sol labels on m6-xl-full-59m and the two trainer files sol-59m / sol-29m.
eos (node A): own-Eos labels on m6-e8f-r2clean (node-A copy) and the trainer file eos-e8f-r2clean.
nodea (node A): the node-A copy of m6-e8f-r2clean and E6K's SELECT / CAL files against E8F's.
Writes /data/dev2/runs/dec/m6/lock-<phase>.json; exit 1 on any failure.

usage: PYTHONPATH=<mirror>/src/training/decision2 python3 m6-lockcheck.py part1|sol|eos|nodea
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE / "v2" / "9b"))
from lux9b.m3_data import c1_keys, denied_hits  # noqa: E402

M = Path("/data/dev2/runs/dec/m6")
DEC = Path("/data/dev2/runs/dec")
HF = Path(
    "/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots"
)
N4 = DEC / "m4/data/m4-xl-full-29m/train.jsonl"
N4_SHA = "c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60"
LUXT = DEC / "m4/teacher/m4-xl-full-29m/lux-teacher.jsonl"
LUXT_SHA = "e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c"
E8F = DEC / "m2/data/m2-full-a7-v1/train.jsonl"
E8F_SHA = "d1dc33fcb7a49fa9df9519545b1b0debeac38f86cbd760b481eb37334c9bd6f6"
PAYLOAD = Path("/data/dev2/runs/eval/m5/overlap-effects/final/excluded-groups.json")
PAYLOAD_SHA = "2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914"
EXCL_SPEC = CODE / "v2/dec/specs/m4-r2-excluded-groups.json"
C1_REGISTRY = CODE / "v2/eval/records/sealed-c1-source-registry-2026-09-28.json"
A7V3 = "3a4efc7768e3458e392f44e83391313fc25e49df"
TARGET_59M = 58_814_652  # 2 x N4XF's 29,407,326 native tokens
SOL = (
    "llm-semantic-router/Decision-1.0-Sol-2B",
    "ce0c018a28de16d6639b1cd203b761bf643b89e6",
)
EOS = (
    "llm-semantic-router/Decision-1.0-Eos-0.8B",
    "363c4a5e56afc115b1c78c837633956d0bbb63ab",
)
phase = sys.argv[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [(line, json.loads(line)) for line in f]


def sliced(rows, man):
    """Components are contiguous in spec (build) order."""
    order = man.get("component_order") or [c["name"] for c in man["spec"]["components"]]
    out, i = {}, 0
    for name in order:
        n = man["components"][name]["rows"]
        out[name] = rows[i : i + n]
        i += n
    assert i == len(rows), (i, len(rows))
    return out


payload = jload(PAYLOAD)["groups"]
ex_ids = {i for g in payload.values() for i in g["row_ids"]}
ex_hashes = {h for g in payload.values() for h in g["input_sha256"]}


def exposed(r):
    return (
        r["group_id"] in payload or r["id"] in ex_ids or r["input_sha256"] in ex_hashes
    )


def teacher_check(train, path, first=None):
    """Every TRAIN row carries a valid target with its input hash; manifest hash == file hash."""
    targets = {}
    for line, r in lines(path):
        targets[r["id"]] = (line, r)
    keys = {r["id"]: [o["key"] for o in r["options"]] for _, r in train}
    missing = Counter(r.get("_c", "?") for _, r in train if r["id"] not in targets)
    bad = sum(
        1
        for _, r in train
        if r["id"] in targets
        and (
            targets[r["id"]][1]["input_sha256"] != r["input_sha256"]
            or set(targets[r["id"]][1]["teacher_probs"]) != set(keys[r["id"]])
            or abs(sum(targets[r["id"]][1]["teacher_probs"].values()) - 1) > 1e-6
        )
    )
    man = jload(str(path) + ".manifest.json")
    out = {
        "file": str(path),
        "sha256": sha(path),
        "manifest_output_sha256": man.get("output_sha256"),
        "rows": len(train),
        "records": len(targets),
        "covered": len(train) - sum(missing.values()),
        "missing_by_component": dict(missing),
        "bad_records": bad,
        "extra_records": len(set(targets) - set(keys)),
    }
    if "sources" in man:
        out["used_by_source"] = [
            (s["file"].split("/snapshots/")[-1], s.get("used", 0))
            for s in man["sources"]
        ]
        out["overlap"] = man["overlap"]
    if "train_label_agreement" in man:
        out["gold_agreement"] = {
            k: (
                {t: round(v["accuracy"], 4) for t, v in a.items()}
                if "accuracy" not in a
                else round(a["accuracy"], 4)
            )
            for k, a in man["train_label_agreement"].items()
        }
    out["ok"] = (
        out["covered"] == len(train)
        and bad == 0
        and out["extra_records"] == 0
        and out["manifest_output_sha256"] == out["sha256"]
    )
    return out, targets


def tag(rows, man):
    for name, items in sliced(rows, man).items():
        for _, r in items:
            r["_c"] = name
    return rows


def label_check(labels, train_sha, repo_rev):
    man = jload(str(labels) + ".manifest.json")
    return {
        "labels": str(labels),
        "labels_sha256": sha(labels),
        "rows": man["rows"],
        "teacher": [man["teacher_repo"], man["teacher_revision"], man["teacher_kind"]],
        "teacher_source_fingerprint": man["teacher_source_fingerprint"],
        "temperatures": man["teacher_temperatures"],
        "max_length": man["max_length"],
        "shard": man["shard"],
        "seconds": man["seconds"],
        "device": man["device_name"],
        "train_sha256_ok": man["train_sha256"] == train_sha,
        "identity_ok": (man["teacher_repo"], man["teacher_revision"]) == repo_rev
        and man["teacher_kind"] == "decision1",
        "gold_agreement": {
            k: round(v["accuracy"], 4) for k, v in man["train_label_agreement"].items()
        },
        "mean_max_probability": {
            k: round(v, 4) for k, v in man["mean_max_probability"].items()
        },
    }


result = {"phase": phase}
fails = []
if phase == "part1":
    if sha(PAYLOAD) != PAYLOAD_SHA:
        fails.append("payload hash")
    spec_groups = set(jload(EXCL_SPEC)["group_ids"])
    result["payload"] = {
        "sha256": PAYLOAD_SHA,
        "groups": len(payload),
        "equals_builder_exclusion_list": spec_groups == set(payload),
    }
    if spec_groups != set(payload):
        fails.append("builder exclusion list != payload groups")
    # m6-xl-full-59m
    p59 = M / "data/m6-xl-full-59m/train.jsonl"
    m59 = jload(str(p59) + ".manifest.json")
    m29 = jload(str(N4) + ".manifest.json")
    r59 = tag(lines(p59), m59)
    r29 = tag(lines(N4), m29)
    if sha(N4) != N4_SHA:
        fails.append("N4XF train hash")
    c59, c29 = sliced(r59, m59), sliced(r29, m29)
    lines59 = {r["id"]: line for line, r in r59}
    nested = {
        name: {
            "rows_29m": len(items),
            "rows_59m": len(c59.get(name, [])),
            "29m_rows_absent_from_59m": sum(r["id"] not in lines59 for _, r in items),
            "29m_rows_differing": sum(
                r["id"] in lines59 and lines59[r["id"]] != line for line, r in items
            ),
        }
        for name, items in c29.items()
    }
    absent = sum(v["29m_rows_absent_from_59m"] for v in nested.values())
    differ = sum(v["29m_rows_differing"] for v in nested.values())
    keys = c1_keys(jload(C1_REGISTRY))
    src59 = Counter(r["source"] for _, r in r59)
    fam59 = Counter(r["family"] for _, r in r59)
    src29 = {r["source"] for _, r in r29}
    c1 = {
        name: hits
        for name in set(src59) | set(fam59)
        if (hits := denied_hits(name, keys, []))
    }
    comp = {
        k: {
            "rows": v["rows"],
            "tokens": v["tokens"],
            "pool_tokens": v["pool_tokens"],
            "budget_tokens": v["budget_tokens"],
            "excluded_group": v.get("excluded_group", 0),
            "excluded_family": v.get("excluded_family", 0),
            "duplicate_input": v.get("duplicate_input", 0),
        }
        for k, v in m59["components"].items()
    }

    def block(pred):
        return sum(v["tokens"] for k, v in comp.items() if pred(k))

    result["m6-xl-full-59m"] = {
        "train_sha256": sha(p59),
        "manifest_output_sha256": m59["output_sha256"],
        "spec_sha256": m59["spec_sha256"],
        "rows": m59["rows"],
        "rows_by_type": m59["rows_by_type"],
        "tokens": m59["tokens"],
        "tokens_vs_2x_n4xf": round(m59["tokens"] / TARGET_59M - 1, 5),
        "tokens_by_block": {
            "A0s": block(lambda k: k == "A0s"),
            "A7": block(lambda k: k.startswith("A7")),
            "v1": block(lambda k: k.startswith("V1:")),
            "H7": block(lambda k: k == "H7"),
            "H8": block(lambda k: k == "H8"),
            "v2_pools": block(
                lambda k: k in ("E11", "G2", "G4h", "G6", "H1", "H3", "H5", "H6")
            ),
        },
        "a7_tokens_by_subarm": {
            k: v["tokens"] for k, v in comp.items() if k.startswith("A7")
        },
        "components": comp,
        "langs_top": Counter(r["language"] for _, r in r59).most_common(10),
        "exposed_rows": sum(exposed(r) for _, r in r59),
        "same_component_names_as_29m": list(m59["components"])
        == list(m29["components"]),
        "same_inputs_as_29m": m59["inputs_sha256"] == m29["inputs_sha256"],
        "same_tokenizer_as_29m": m59["tokenizer_files_sha256"]
        == m29["tokenizer_files_sha256"],
        "a0s_identical_to_29m": [x[0] for x in c59["A0s"]]
        == [x[0] for x in c29["A0s"]],
        "nested_over_29m": absent == 0 and differ == 0,
        "29m_rows_absent_from_59m": absent,
        "29m_rows_differing": differ,
        "nested_by_component": nested,
        "sources_not_in_29m": sorted(set(src59) - src29),
        "c1_registry_hits": c1,
        "c1_registry_sha256": sha(C1_REGISTRY),
        "c1_keys": len(keys),
        "a7x_components": [k for k in comp if k.upper().startswith("A7X")],
    }
    x = result["m6-xl-full-59m"]
    if x["manifest_output_sha256"] != x["train_sha256"]:
        fails.append("59m manifest hash")
    if x["exposed_rows"]:
        fails.append("59m exposed rows")
    if not (
        x["same_component_names_as_29m"]
        and x["same_inputs_as_29m"]
        and x["a0s_identical_to_29m"]
    ):
        fails.append("59m builder identity vs 29m")
    if c1 or x["a7x_components"]:
        fails.append("59m C1 / A7x")
    if abs(x["tokens_vs_2x_n4xf"]) > 0.02:
        fails.append("59m token budget")
    # own-Lux teachers
    for name, train in (("lux-all-29m", r29), ("lux-all-59m", r59)):
        path = M / f"teacher/{name}/teacher.jsonl"
        if not path.is_file():
            result[name] = {"ok": False, "missing_file": str(path)}
            fails.append(name)
            continue
        result[name], targets = teacher_check(train, path)
        if not result[name]["ok"]:
            fails.append(name)
        result[name]["_targets"] = {i: line for i, (line, _) in targets.items()}
    if sha(LUXT) != LUXT_SHA:
        fails.append("N4XF lux teacher hash")
    luxt = {r["id"]: line for line, r in lines(LUXT)}
    if "_targets" in result.get("lux-all-29m", {}):
        t29 = result["lux-all-29m"].pop("_targets")
        changed = sum(t29.get(i) != line for i, line in luxt.items())
        result["lux-all-29m"]["n4xf_targets_changed"] = changed
        result["lux-all-29m"]["added_rows_by_component"] = dict(
            Counter(r["_c"] for _, r in r29 if r["id"] not in luxt)
        )
        if changed:
            fails.append("lux-all-29m changed N4XF targets")
        if "_targets" in result.get("lux-all-59m", {}):
            t59 = result["lux-all-59m"].pop("_targets")
            diff = sum(t59.get(i) != line for i, line in t29.items())
            result["lux-all-59m"]["differs_from_lux-all-29m_on_shared_rows"] = diff
            if diff:
                fails.append("lux-all-59m != lux-all-29m on 29m rows")
    for name in ("lux-all-29m", "lux-all-59m"):
        result.get(name, {}).pop("_targets", None)
    # m6-e8f-r2clean
    pe = M / "data/m6-e8f-r2clean/train.jsonl"
    me = jload(str(pe) + ".manifest.json")
    if sha(E8F) != E8F_SHA:
        fails.append("E8F train hash")
    kept = [line for line, _ in lines(pe)]
    removed = {i for ids in me["removed_ids"].values() for i in ids}
    src = [(line, r) for line, r in lines(E8F)]
    expect = [line for line, r in src if r["id"] not in removed]
    quarantine = set(jload(DEC / "m3/e8f-a7v3-removed-in-mixture.ids.json"))
    kept_rows = [json.loads(line) for line in kept]
    reg = jload(HF / A7V3 / "v2/a7/registry.json")["files"]
    a7files = {
        f.split("/snapshots/")[-1]: s for f, s in me["a7v3_files_sha256"].items()
    }
    a7_ok = all(
        reg.get(k.split("/", 2)[2].removeprefix("v2/"), {}).get("sha256") == s
        for k, s in a7files.items()
    )
    result["m6-e8f-r2clean"] = {
        "train_sha256": sha(pe),
        "manifest_output_sha256": me["output_sha256"],
        "rows": me["rows"],
        "rows_by_type": me["rows_by_type"],
        "tokens": me["tokens"],
        "source_rows": me["source_rows"],
        "source_tokens": me["source_tokens"],
        "removed_by_reason": me["removed_by_reason"],
        "membership_by_reason": me["membership_by_reason"],
        "removed_groups_by_reason": me["removed_groups_by_reason"],
        "removed_by_component": me["removed_by_component"],
        "removed_by_source": me["removed_by_source"],
        "removed_tokens": me["removed_tokens"],
        "components": me["components"],
        "langs_top": list(me["rows_by_language"].items())[:10],
        "kept_rows_byte_identical_in_order": kept == expect,
        "exposed_rows": sum(exposed(r) for r in kept_rows),
        "quarantine_ids_left": sum(r["id"] in quarantine for r in kept_rows),
        "quarantine_sha256": sha(DEC / "m3/e8f-a7v3-removed-in-mixture.ids.json"),
        "a7v3_files_match_registry": a7_ok,
        "a7v3_registry_sha256": sha(HF / A7V3 / "v2/a7/registry.json"),
        "tokenizer_note": "tokens = E8F manifest tokens minus encode lengths of removed rows",
    }
    e = result["m6-e8f-r2clean"]
    if not (
        e["kept_rows_byte_identical_in_order"]
        and e["exposed_rows"] == 0
        and e["quarantine_ids_left"] == 0
        and a7_ok
        and e["manifest_output_sha256"] == e["train_sha256"]
    ):
        fails.append("m6-e8f-r2clean")
    # exposure receipts
    for mix, train_sha in (
        ("m6-xl-full-59m", x["train_sha256"]),
        ("m6-e8f-r2clean", e["train_sha256"]),
    ):
        rp = M / f"exposure/{mix}/exposure-{mix}.json"
        rec = jload(rp)
        result[f"exposure-{mix}"] = {
            "receipt": str(rp),
            "receipt_sha256": sha(rp),
            "groups": len(rec["groups"]),
            "methods_agree": rec["methods_agree"],
            "payload_sha256": rec["payload_sha256"],
            "files": rec["files"],
        }
        if (
            rec["groups"]
            or rec["payload_sha256"] != PAYLOAD_SHA
            or rec["files"][0]["sha256"] != train_sha
        ):
            fails.append(f"exposure {mix}")
elif phase == "sol":
    p59 = M / "data/m6-xl-full-59m/train.jsonl"
    s59 = sha(p59)
    r59, r29 = lines(p59), lines(N4)
    result["labels"] = label_check(M / "teacher/sol-labels/labels.jsonl", s59, SOL)
    lab = result["labels"]
    if not (lab["train_sha256_ok"] and lab["identity_ok"] and lab["rows"] == len(r59)):
        fails.append("sol labels")
    result["sol-59m"], t59 = teacher_check(r59, M / "teacher/sol-59m/teacher.jsonl")
    result["sol-29m"], t29 = teacher_check(r29, M / "teacher/sol-29m/teacher.jsonl")
    diff = sum(
        t59[i][1]["teacher_probs"] != t[1]["teacher_probs"]
        for i, t in t29.items()
        if i in t59
    )
    result["sol-29m"]["differs_from_sol-59m_on_shared_rows"] = diff
    for k in ("sol-59m", "sol-29m"):
        if not result[k]["ok"]:
            fails.append(k)
    if diff:
        fails.append("sol-29m != sol-59m")
elif phase == "eos":
    pe = M / "data/m6-e8f-r2clean/train.jsonl"
    se = sha(pe)
    re_ = lines(pe)
    result["labels"] = label_check(M / "teacher/eos-labels/labels.jsonl", se, EOS)
    lab = result["labels"]
    if not (lab["train_sha256_ok"] and lab["identity_ok"] and lab["rows"] == len(re_)):
        fails.append("eos labels")
    result["eos-e8f-r2clean"], _ = teacher_check(
        re_, M / "teacher/eos-e8f-r2clean/teacher.jsonl"
    )
    if not result["eos-e8f-r2clean"]["ok"]:
        fails.append("eos-e8f-r2clean")
elif phase == "nodea":
    pe = M / "data/m6-e8f-r2clean/train.jsonl"
    want = (M / "data/m6-e8f-r2clean/READY").read_text().split()[0]
    d = Path("/data/decision20-20260926/data/hf-private-decision20-clean-v2")
    result["m6-e8f-r2clean"] = {"sha256": sha(pe), "expected": want}
    result["data_dir"] = {
        "dir": str(d),
        "select.jsonl": sha(d / "select.jsonl"),
        "cal.jsonl": sha(d / "cal.jsonl"),
    }
    # E8F's seeds mounted node B data-cleanv2 (launch receipts m2/arms/full/m2-E8F-s*.launch.json)
    if result["m6-e8f-r2clean"]["sha256"] != want:
        fails.append("node-A copy hash")
    if (
        result["data_dir"]["select.jsonl"]
        != "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6"
        or result["data_dir"]["cal.jsonl"]
        != "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a"
    ):
        fails.append("node-A SELECT / CAL != E8F's")
else:
    sys.exit(f"unknown phase {phase}")
result["fails"] = fails
result["status"] = "PASS" if not fails else "FAIL"
out = M / f"lock-{phase}.json"
out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
print(
    json.dumps(
        {
            "status": result["status"],
            "fails": fails,
            "file": str(out),
            "sha256": sha(out),
        }
    )
)
sys.exit(0 if not fails else 1)
