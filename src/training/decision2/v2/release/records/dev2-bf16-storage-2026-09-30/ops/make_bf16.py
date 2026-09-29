"""Specs and final decisions of the BF16 storage revisions of DEV2.0-0.8B / 2B / 4B.

Coordinator note 2026-09-30 06:10 (27B release path, step 1): convert the released FP32 packages to BF16 with
v2.release.bf16_copy, adopting only with exact parity on every scored prompt and on mlx-diag, publish each as a
card-only-equivalent new revision with a new decision, then purge the FP32 blobs (rewrite_history=False).

Each spec is the size's current card-c1 spec with: checkpoint = the node-A BF16 copy, bf16_copy = its receipt
(pinned by SHA-256), expected_identity = the copy's model hash, gate_receipt = the new final decision, and one
sentence in runtime_equivalence on how the weights are stored. Scored run, card inputs, runtime and vendor pins
(the mirror that built the replaced revision), calibration and licence stay as they are.

Each decision copies the superseded final decision's judgement (scored report, paired comparison, calibration,
licence, disclosures, C1 line) and changes only the identity, which the storage cast changes. Run from
src/training/decision2 after the receipts are copied under the records directory:

  python3 v2/release/records/dev2-bf16-storage-2026-09-30/ops/make_bf16.py [--check]
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

RECORDS = Path("v2/release/records")
SPECS = Path("v2/release/specs")
OUT = RECORDS / "dev2-bf16-storage-2026-09-30"
CARD_PASS = RECORDS / "dev2-c1-card-pass-2026-09-29"
DECISIONS = "/data/dev2/runs/release/decisions"
INPUTS = "/data/dev2/runs/release/inputs"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the BF16 storage conversion of the released FP32 "
    "packages at exact parity, assigned to release engineering in the cross-track note of 2026-09-30 06:10 "
    "UTC+8 (27B release path, step 1), under the user's full-autonomy mandate and the user directive of "
    "2026-09-29 16:05 UTC+8 (prefer BF16 packages when v2.release.bf16_copy gives exact parity)"
)
PREPARED_BY = "Decision 2.0 release engineering, ~27B release worker (worktree vllm-sr-dev2-release-27b)"
ORDER = "0.6B, 0.8B, 2B, 4B, 9B, 27B"
STORAGE = (
    " The scored checkpoint ({source}) stored every tensor in FP32; this package (v2.release.bf16_copy, receipt "
    "{receipt}) stores its {n} Linear projection matrices in BF16 exactly as BF16 autocast rounds them and every "
    "other tensor bit for bit in FP32."
)
TIERS = (
    {"tier": "0.8B", "key": "0p8b"},
    {"tier": "2B", "key": "2b"},
    {"tier": "4B", "key": "4b"},
)
PANELS_0P8B = (
    "(typed-final 1,600, css15 6,547, public231 231) by release.sh --parity, with the scored run's persisted "
    "Triton autotune cache."
)
PANELS_0P8B_NEW = (
    "(typed-final 1,600, css15 6,547, public231 231) and of the mlx-diag diagnostic (2,275) by release.sh "
    "--parity, with the scored run's persisted Triton autotune cache."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_for(t: dict) -> tuple[dict, dict]:
    old = json.loads(
        (SPECS / f"dev2-{t['key']}-card-c1.json").read_text(encoding="utf-8")
    )
    receipt_path = OUT / t["key"] / "bf16-copy.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert (
        receipt["source_model_sha256"] == old["expected_identity"]["model_sha256"]
    ), t["tier"]
    node = f"{INPUTS}/dev2-{t['key']}-bf16"
    spec = copy.deepcopy(old)
    spec["checkpoint"] = f"{node}/checkpoint"
    spec["bf16_copy"] = {
        "receipt": f"{node}/bf16-copy.json",
        "sha256": sha(receipt_path),
    }
    spec["expected_identity"] = {"model_sha256": receipt["model_sha256"]}
    spec["gate_receipt"] = f"{DECISIONS}/DEV2.0-{t['tier']}.decision.bf16.json"
    text = spec["runtime_equivalence"]
    if t["tier"] == "0.8B":
        assert text.endswith(PANELS_0P8B), t["tier"]
        text = text[: -len(PANELS_0P8B)] + PANELS_0P8B_NEW
    marker = " Checked on one GPU of the scoring node"
    assert text.count(marker) == 1, t["tier"]
    storage = STORAGE.format(
        source=receipt["source_model_sha256"][:8],
        receipt=sha(receipt_path)[:8],
        n=receipt["tensors_by_storage_dtype"]["BF16"],
    )
    spec["runtime_equivalence"] = text.replace(marker, storage + marker)
    spec["_release"] = {
        "bf16_storage": (
            "Storage-only revision (coordinator note 2026-09-30 06:10 UTC+8, 27B release path step 1): the FP32 "
            f"backbone of the replaced revision is stored as its v2.release.bf16_copy (receipt {sha(receipt_path)[:8]}, "
            f"identity {receipt['source_model_sha256'][:8]} -> {receipt['model_sha256'][:8]}; "
            f"{receipt['tensors_by_storage_dtype']['BF16']} Linear projection matrices in BF16 as BF16 autocast rounds "
            "them, every other tensor FP32 bit for bit). Adopted only with 0 answer changes on every scored prompt "
            "and on mlx-diag. The card, scored run, runtime and vendored sources (pinned to the mirror that built the "
            "replaced revision) are unchanged; the runtime-equivalence text gains one storage sentence."
        ),
        "previous": old.get("_release"),
    }
    return spec, receipt


def decision_for(t: dict, spec: dict, spec_sha: str, receipt: dict) -> dict:
    old_path = CARD_PASS / f"DEV2.0-{t['tier']}.decision.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    gate = json.loads(
        (CARD_PASS / t["key"] / "receipts" / "gate.json").read_text(encoding="utf-8")
    )
    old_sha = sha(old_path)
    assert gate["decision_sha256"] == old_sha and old["status"] == "final", t["tier"]
    assert old["identity"]["model_sha256"] == receipt["source_model_sha256"], t["tier"]
    repo, revision = old["repo_id"], gate["revision"]
    new = dict(old)
    for key in (
        "approved_package",
        "card_revision",
        "previous_rationale",
        "revision_kind",
        "package_verification",
    ):
        new.pop(key, None)
    receipt_sha = spec["bf16_copy"]["sha256"]
    new.update(
        {
            "identity": {"model_sha256": receipt["model_sha256"]},
            "prepared_by": PREPARED_BY,
            "decided_by": DECIDED_BY,
            "decided_utc": "2026-09-29T22:10:00Z",
            "action": (
                f"Storage-only revision of the private repository {repo}: the FP32 backbone files of the released "
                f"revision {revision} are replaced by their v2.release.bf16_copy (receipt {receipt_sha[:8]}; "
                f"{receipt['tensors_by_storage_dtype']['BF16']} Linear projection matrices stored in BF16 exactly as "
                "the runtime's BF16 autocast rounds them, every other tensor FP32 bit for bit). The decision head, "
                "tokenizer, configs, runtime, vendored sources and card text stay as they are; MODEL_MANIFEST.json "
                "records the new identity and one storage sentence. After the new revision verifies, the superseded "
                "FP32 backbone LFS objects are purged with rewrite_history=False; the node-A copies stay the durable "
                f"store. The repository stays in the private collection '🎲 Decision 2.0', ordered {ORDER}; "
                "everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                "scored report, paired comparison, calibration (T = 1) and licence decision. Only the checkpoint "
                f"identity changes ({receipt['source_model_sha256'][:8]} -> {receipt['model_sha256'][:8]}), because the "
                "stored Linear weights are already rounded as the runtime rounds them before every matmul. Adopted "
                "only with exact parity (0 answer changes, 0 missing) against the T = 1 bindings on every scored "
                "prompt (typed-final 1,600, css15 6,547, public231 231) and on mlx-diag (2,275): release.sh parity-pre "
                "blocks the upload otherwise, and parity-post repeats it on the real download. The coordinator's note "
                "of 2026-09-30 06:10 UTC+8 frees private Hugging Face storage this way for the ~27B successor."
            ),
            "previous_rationale": old["rationale"],
            "storage_revision": {
                "kind": "bf16-storage",
                "spec": f"v2/release/specs/dev2-{t['key']}-bf16.json",
                "spec_sha256": spec_sha,
                "bf16_copy_receipt_sha256": receipt_sha,
                "source_model_sha256": receipt["source_model_sha256"],
                "tensors_by_storage_dtype": receipt["tensors_by_storage_dtype"],
                "tensor_bytes": receipt["tensor_bytes"],
            },
            "c1": (
                old["c1"]
                + " The BF16 storage copy computes the same BF16 matmuls as the scored FP32 checkpoint, so the C1 "
                "baseline of these weights stays valid for this revision."
            ),
            "supersedes": {
                "final_sha256": old_sha,
                "released_as": f"{repo}@{revision}",
                "released_manifest_sha256": gate["manifest_sha256"],
                "released_gate_sha256": sha(
                    CARD_PASS / t["key"] / "receipts" / "gate.json"
                ),
                "card_revision": old.get("card_revision"),
                "earlier": old.get("supersedes"),
            },
        }
    )
    return new


def main() -> int:
    check = "--check" in sys.argv[1:]
    problems = []
    for t in TIERS:
        spec, receipt = spec_for(t)
        spec_path = SPECS / f"dev2-{t['key']}-bf16.json"
        spec_text = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
        spec_sha = hashlib.sha256(spec_text.encode()).hexdigest()
        decision = decision_for(t, spec, spec_sha, receipt)
        target = OUT / f"DEV2.0-{t['tier']}.decision.json"
        decision_text = json.dumps(decision, ensure_ascii=False, indent=2) + "\n"
        for path, text in ((spec_path, spec_text), (target, decision_text)):
            if check:
                if not path.is_file() or path.read_text(encoding="utf-8") != text:
                    problems.append(f"{path}: differs from the derivation")
            else:
                path.write_text(text, encoding="utf-8")
            print(path, hashlib.sha256(text.encode()).hexdigest())
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
