"""Normalize pinned public Chinese/Russian typed-decision data for DEV only.

The upstream questions are already public, including their labels. This tool
keeps source text and targets outside Git, excludes rows without a verifiable
MASSIVE source identity, and cannot create a sealed release panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from inference.run import digest, file_digest

VERSION = "decision2-public-multilingual-typed-dev/2"
ZH_COMMIT = "4d6f0a9875d5558efec8b6b48323de65e20724b6"
RU_COMMIT = "10713af21eb5772be20ee8a6ab8263d49c1171ac"
MASSIVE_ARCHIVE_SHA256 = (
    "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
)
MASSIVE_LOCALE_SHA256 = {
    "zh-CN": "992bf0bef3d678f08c27e514739bc851163e8f40f530bfb4d5970a2c24408ace",
    "ru-RU": "af7367861ea21c1d69a084b290145815f7eaf82dcdc43b7ab572b993f9b2a69d",
}
MASSIVE_TRAIN_SHA256 = {
    "v1": "4c57dac9d5dd39cf3e0920cda7c1e975f5e2dacf46379e7232797efb3ae2d9ba",
    "v2": "55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80",
}
SOURCE_FILES = {
    "zh": {
        "README.md": "834bdb00d4a9551c82aa87c5d2580c2c2982b19d5c23eaa26eb49464fa8bf214",
        "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "LICENSE-DATA": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
        "src/convert_massive.py": "aeabae3cddee5957715b67f5187149ae7e9dc9ae53007b121d1fcbbee0302bd3",
        "data/review_log.md": "4a1939785b05d693f06d7490c654784bc2d464178febf145ad0df6ebad5c29eb",
        "data/massive_items.jsonl": "d8b5fcb0e5d4ea359d48a940383cf76795fdc3b4e80d614d3dd9a2fe18408424",
        "data/synthetic_items.jsonl": "86384af1471c74aa568669817228dc5f16b3b61cc180e8c4e7e92fae75a689f7",
    },
    "ru": {
        "README.md": "1429ec5a0c5edb12a42ba349e43624f9614dc0450c78e7f2c7f3b2b12a1a6341",
        "score.py": "6a7929bcfe3bff33b5633368c0ed523402942bbe271084d8267027a2aa100473",
        "data/track_a_unseen.jsonl": "826ea9fe62fdb2e88d308d9c232cde36b237db99b1ee20051d6db75dd8b992ac",
        "data/track_b_applied.jsonl": "a9638dd72945ea1aeb23cff5ca5fd8af30c7586430563c7f066a9de1232f1482",
    },
}
ZH_MASSIVE_ID = re.compile(r"(?:^|, )massive_id=(\d+)$")
FORBIDDEN_PROMPT_KEYS = {"gold", "answer", "label", "target", "audit_metadata"}
MODEL_KEYS = ("backend", "model_id", "model_revision", "adapter_version")


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(item)
    if not rows and not allow_empty:
        raise ValueError(f"{path}: empty JSONL")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            # Native Choice reads criteria in insertion order. Sorting nested
            # keys here changes the model input and its recorded fingerprint.
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_serialized_inputs(
    prompts: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> None:
    if len(prompts) != len(targets):
        raise ValueError("Prompt/target row count mismatch")
    for prompt, target in zip(prompts, targets, strict=True):
        if prompt["id"] != target["id"]:
            raise ValueError("Prompt/target ID mismatch")
        payload = {"state": prompt["state"], "questions": prompt["questions"]}
        if digest(payload) != target["source_input_sha256"]:
            raise ValueError(f"{prompt['id']}: serialized input fingerprint mismatch")
        question = prompt["questions"]["decision"]
        if (
            question["type"] == "choice"
            and list(question["criteria"]) != target["options"]
        ):
            raise ValueError(f"{prompt['id']}: serialized Choice order changed")


def verify_source(root: Path, name: str) -> dict[str, Any]:
    expected_commit = {"zh": ZH_COMMIT, "ru": RU_COMMIT}[name]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if revision != expected_commit:
        raise ValueError(f"{name}: source commit changed")
    hashes = {}
    for relative, expected in SOURCE_FILES[name].items():
        path = root / relative
        actual = file_digest(path)
        if actual != expected:
            raise ValueError(f"{name}: {relative} differs from pinned source")
        hashes[relative] = actual
    return {
        "url": {
            "zh": "https://github.com/CodyQin/zh-decision-bench",
            "ru": "https://github.com/smolnikov-k/rudecide",
        }[name],
        "commit": revision,
        "files_sha256": hashes,
    }


def _question_target(
    question: dict[str, Any], answer: Any
) -> tuple[str, Any, list[str]]:
    if (
        not isinstance(question, dict)
        or not isinstance(question.get("instructions"), str)
        or not question["instructions"].strip()
    ):
        raise ValueError("Question instructions missing")
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if (
            not isinstance(criteria, dict)
            or len(criteria) < 2
            or any(
                not isinstance(key, str)
                or not key
                or (
                    value is not None
                    and (not isinstance(value, str) or not value.strip())
                )
                for key, value in criteria.items()
            )
            or not isinstance(answer, str)
            or answer not in criteria
        ):
            raise ValueError("Choice labels/answer invalid")
        return kind, answer, list(criteria)
    if kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) != {"true", "false"}
        ):
            raise ValueError("Noul criteria invalid")
        if type(answer) is bool:
            return kind, answer, ["false", "true"]
        if isinstance(answer, str) and answer in {"true", "false"}:
            return kind, answer == "true", ["false", "true"]
        raise ValueError("Noul answer invalid")
    if kind == "score":
        if (
            not isinstance(criteria, list)
            or not 2 <= len(criteria) <= 10
            or any(
                not isinstance(value, str) or not value.strip() for value in criteria
            )
            or len(set(criteria)) != len(criteria)
        ):
            raise ValueError("Score levels invalid")
        if type(answer) is int and 0 <= answer < len(criteria):
            index = answer
        elif isinstance(answer, str) and answer in criteria:
            index = criteria.index(answer)
        elif isinstance(answer, str) and answer in [
            str(i) for i in range(len(criteria))
        ]:
            index = int(answer)
        else:
            raise ValueError("Score answer invalid")
        return kind, index, [str(i) for i in range(len(criteria))]
    raise ValueError("Unknown typed-decision question")


def _make_row(
    item_id: str,
    group_id: str,
    task: str,
    language: str,
    state: str,
    question: dict[str, Any],
    answer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(state, str) or not state.strip():
        raise ValueError("Empty state")
    kind, gold, options = _question_target(question, answer)
    prompt = {"id": item_id, "state": state, "questions": {"decision": question}}
    if FORBIDDEN_PROMPT_KEYS & set(prompt) or FORBIDDEN_PROMPT_KEYS & set(question):
        raise ValueError("Answer leaked into prompt")
    target = {
        "id": item_id,
        "group_id": group_id,
        "task": task,
        "language": language,
        "task_type": kind,
        "gold": gold,
        "options": options,
        "source_input_sha256": digest(
            {"state": state, "questions": prompt["questions"]}
        ),
    }
    return prompt, target


def protected_massive_ids(candidate: Path) -> tuple[set[str], dict[str, Any]]:
    rows = _read_jsonl(candidate)
    ids: set[str] = set()
    for row in rows:
        meta = row.get("audit_metadata")
        source_id = meta.get("source_id") if isinstance(meta, dict) else None
        group_id = row.get("group_id")
        if (
            not isinstance(source_id, str)
            or not source_id.isdecimal()
            or group_id != f"massive-1.1:{source_id}"
        ):
            raise ValueError(
                "Protected MASSIVE candidate lacks matching source ID/group"
            )
        ids.add(source_id)
    return ids, {
        "sha256": file_digest(candidate),
        "rows": len(rows),
        "source_groups": len(ids),
    }


def audit_massive_ids(
    zh_ids: set[str], ru_ids: set[str], candidates: dict[str, Path]
) -> dict[str, Any]:
    if set(candidates) != {"v1", "v2"} or len(zh_ids) != 179 or len(ru_ids) != 290:
        raise ValueError(
            "Both known MASSIVE TRAIN candidates and complete source IDs required"
        )
    report = {}
    for name, path in candidates.items():
        train_ids, receipt = protected_massive_ids(path)
        overlap = train_ids & (zh_ids | ru_ids)
        if overlap:
            raise ValueError(
                f"Public MASSIVE overlaps protected {name} TRAIN: {len(overlap)} source groups"
            )
        report[name] = {
            **receipt,
            "zh_overlap_source_groups": 0,
            "ru_overlap_source_groups": 0,
        }
    return report


def official_massive(archive: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if file_digest(archive) != MASSIVE_ARCHIVE_SHA256:
        raise ValueError("Official MASSIVE archive changed")
    by_id = {}
    by_test_utterance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_sha = {}
    with tarfile.open(archive, "r:gz") as source:
        for locale in MASSIVE_LOCALE_SHA256:
            blob = source.extractfile(f"1.1/data/{locale}.jsonl").read()
            actual = hashlib.sha256(blob).hexdigest()
            if actual != MASSIVE_LOCALE_SHA256[locale]:
                raise ValueError(f"Official MASSIVE {locale} source changed")
            source_sha[locale] = actual
            rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
            if len(rows) != 16521 or len({str(row["id"]) for row in rows}) != 16521:
                raise ValueError(f"Official MASSIVE {locale} identity changed")
            if locale == "zh-CN":
                by_id = {str(row["id"]): row for row in rows}
            else:
                for row in rows:
                    if row["partition"] == "test":
                        by_test_utterance[row["utt"].strip()].append(row)
    return {"zh_by_id": by_id, "ru_test_by_utterance": by_test_utterance}, {
        "archive_sha256": MASSIVE_ARCHIVE_SHA256,
        "locale_file_sha256": source_sha,
        "lineage_use": "ZH original ID+utterance+DEV; RU exact unique TEST utterance; ambiguous RU mappings held",
    }


def _zh_rows(
    root: Path, official_by_id: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    massive = _read_jsonl(root / "data/massive_items.jsonl")
    synthetic = _read_jsonl(root / "data/synthetic_items.jsonl")
    if len(massive) != 179 or len(synthetic) != 40:
        raise ValueError("ZH source roster changed")
    prompts, targets, source_ids = [], [], set()
    item_ids = set()
    for source, rows in (("massive", massive), ("synthetic", synthetic)):
        for item in rows:
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id or item_id in item_ids:
                raise ValueError("ZH item ID missing or duplicate")
            item_ids.add(item_id)
            if source == "massive":
                match = ZH_MASSIVE_ID.search(item.get("notes", ""))
                if not match or item.get("domain") != "voice_assistant_routing":
                    raise ValueError("ZH MASSIVE lineage missing")
                if match.group(1) in source_ids:
                    raise ValueError("ZH MASSIVE source ID repeated")
                official = official_by_id.get(match.group(1))
                if (
                    official is None
                    or official["partition"] != "dev"
                    or official["utt"].strip() != item["state"].strip()
                ):
                    raise ValueError(
                        "ZH MASSIVE source ID/utterance/DEV lineage mismatch"
                    )
                source_ids.add(match.group(1))
            elif item.get("domain") not in {"ecommerce_cs", "content_moderation"}:
                raise ValueError("ZH synthetic domain changed")
            questions, gold = item.get("questions"), item.get("gold")
            if (
                not isinstance(questions, dict)
                or not isinstance(gold, dict)
                or set(questions) != set(gold)
            ):
                raise ValueError("ZH question/gold inventory mismatch")
            for question_name, question in questions.items():
                row_id = f"zh-decision-bench/{item_id}/{question_name}"
                prompt, target = _make_row(
                    row_id,
                    f"zh-decision-bench/{item_id}",
                    f"zh/{item['domain']}/{question_name}",
                    "zh-CN",
                    item["state"],
                    question,
                    gold[question_name],
                )
                prompts.append(prompt)
                targets.append(target)
    if len(prompts) != 284 or len(source_ids) != 179:
        raise ValueError("ZH question/source counts changed")
    return prompts, targets, source_ids


def _ru_rows(
    root: Path, official_test: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], dict[str, int]]:
    prompts, targets = [], []
    held = Counter()
    seen = set()
    source_ids = set()
    for track, expected in (("track_a_unseen", 2235), ("track_b_applied", 1543)):
        rows = _read_jsonl(root / f"data/{track}.jsonl")
        if len(rows) != expected:
            raise ValueError("RU track roster changed")
        for item in rows:
            if item.get("track") != track or not isinstance(item.get("task"), str):
                raise ValueError("RU track/task changed")
            source_id = item.get("id")
            if not isinstance(source_id, str) or source_id in seen:
                raise ValueError("RU row ID missing or duplicate")
            seen.add(source_id)
            state = item.get("state")
            if not isinstance(state, str):
                raise ValueError("RU state missing")
            group_id = None
            # The published Russian rows omit original MASSIVE IDs. Only an
            # exact, unique match in the pinned official TEST file recovers one.
            if item["task"] == "massive_ru":
                matches = official_test.get(state.strip(), [])
                if len(matches) != 1:
                    held["massive_ru_ambiguous_source_id"] += 1
                    continue
                source_id = str(matches[0]["id"])
                source_ids.add(source_id)
                group_id = f"massive-1.1:{source_id}"
            if group_id is None:
                state_sha = hashlib.sha256(state.encode("utf-8")).hexdigest()[:20]
                group_id = f"rudecide/{track}/{item['task']}/{state_sha}"
            prompt, target = _make_row(
                f"rudecide/{item['id']}",
                group_id,
                f"ru/{track}/{item['task']}",
                "ru-RU",
                state,
                item["question"],
                item["answer"],
            )
            prompts.append(prompt)
            targets.append(target)
    if (
        len(seen) != 3778
        or held["massive_ru_ambiguous_source_id"] != 10
        or len(source_ids) != 290
        or len(prompts) != 3768
    ):
        raise ValueError("RU eligible/held roster changed")
    return prompts, targets, source_ids, dict(held)


def build(
    zh_root: Path,
    ru_root: Path,
    massive_archive: Path,
    output: Path,
    massive_v1: Path,
    massive_v2: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    sources = {"zh": verify_source(zh_root, "zh"), "ru": verify_source(ru_root, "ru")}
    official, official_receipt = official_massive(massive_archive)
    zh_prompts, zh_targets, zh_ids = _zh_rows(zh_root, official["zh_by_id"])
    ru_prompts, ru_targets, ru_ids, held = _ru_rows(
        ru_root, official["ru_test_by_utterance"]
    )
    overlap = audit_massive_ids(zh_ids, ru_ids, {"v1": massive_v1, "v2": massive_v2})
    if any(
        overlap[name]["sha256"] != expected
        for name, expected in MASSIVE_TRAIN_SHA256.items()
    ):
        raise ValueError("Protected MASSIVE TRAIN candidate revision changed")
    prompts, targets = zh_prompts + ru_prompts, zh_targets + ru_targets
    if len(prompts) != len(targets) or len({row["id"] for row in prompts}) != len(
        prompts
    ):
        raise ValueError("Normalized prompt/target identity mismatch")
    output.mkdir(mode=0o700, parents=True)
    prompt_path, target_path = (
        output / "prompts.jsonl",
        output / "targets.private.jsonl",
    )
    _write_jsonl(prompt_path, prompts)
    _write_jsonl(target_path, targets)
    target_path.chmod(0o600)
    _validate_serialized_inputs(_read_jsonl(prompt_path), _read_jsonl(target_path))
    counts = Counter((row["language"], row["task_type"]) for row in targets)
    task_counts = Counter(row["task"] for row in targets)
    manifest = {
        "schema_version": VERSION,
        "adapter_source_sha256": file_digest(Path(__file__)),
        "scope": "exposed, public, supplementary DEV diagnostic only",
        "release_panel_eligible": False,
        "training_approved": False,
        "source": sources,
        "massive_official_source": official_receipt,
        "upstream_rights": {
            "zh": "CC BY 4.0 data; Apache-2.0 code; attribute author and Amazon MASSIVE",
            "ru": "mixed per-task upstream licenses; evaluation-only; no raw-data redistribution from this adapter",
        },
        "massive_train_source_id_audit": overlap,
        "held_upstream_rows": held,
        "unresolved_overlap": "Future MASSIVE TRAIN candidates require a new source-ID audit before their training or reuse of this diagnostic",
        "source_question_rows": {"zh": 284, "ru": 3778},
        "eligible_question_rows": len(prompts),
        "observable_group_count": len({row["group_id"] for row in targets}),
        "grouping_limit": "Chinese item IDs and recovered MASSIVE source IDs are known; other Russian tasks group identical task/state text only. These are not certified independent source examples.",
        "by_language_type": {
            f"{language}/{kind}": count
            for (language, kind), count in sorted(counts.items())
        },
        "by_task": dict(sorted(task_counts.items())),
        "files_sha256": {
            "prompts.jsonl": file_digest(prompt_path),
            "targets.private.jsonl": file_digest(target_path),
        },
        "native_rule": "Choice exact criterion key; Noul P(true)>=0.5; Score argmax over every ordinal level. Missing/invalid answer is wrong.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _native_choice(answer: Any, target: dict[str, Any]) -> tuple[bool, bool]:
    if (
        not isinstance(answer, dict)
        or answer.get("type") != target["task_type"]
        or "error" in answer
    ):
        return False, False
    kind = target["task_type"]
    if kind == "choice":
        label = answer.get("choice")
        valid = isinstance(label, str) and label in target["options"]
        return valid, bool(valid and label == target["gold"])
    if kind == "noul":
        value = answer.get("noul", answer.get("probability"))
        valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
        return valid, bool(valid and (value >= 0.5) == target["gold"])
    probabilities = answer.get("probabilities")
    valid = (
        isinstance(probabilities, dict)
        and set(probabilities) == set(target["options"])
        and all(
            type(p) in (int, float) and math.isfinite(p) and p >= 0
            for p in probabilities.values()
        )
        and 0.99 <= sum(probabilities.values()) <= 1.01
    )
    return valid, bool(
        valid and int(max(probabilities, key=probabilities.get)) == target["gold"]
    )


def score(panel: Path, predictions: Path) -> dict[str, Any]:
    manifest_path = panel / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("release_panel_eligible") is not False
        or manifest.get("adapter_source_sha256") != file_digest(Path(__file__))
    ):
        raise ValueError("Unknown or release-mislabelled public DEV panel")
    for name, expected in manifest["files_sha256"].items():
        if file_digest(panel / name) != expected:
            raise ValueError(f"{name}: frozen panel changed")
    prompts = _read_jsonl(panel / "prompts.jsonl")
    targets = _read_jsonl(panel / "targets.private.jsonl")
    answers = _read_jsonl(predictions, allow_empty=True)
    by_prompt = {row["id"]: row for row in prompts}
    by_target = {row["id"]: row for row in targets}
    by_answer = {row["id"]: row for row in answers}
    if (
        len(by_prompt) != len(prompts)
        or len(by_target) != len(targets)
        or len(by_answer) != len(answers)
        or set(by_prompt) != set(by_target)
        or not set(by_answer) <= set(by_prompt)
    ):
        raise ValueError("Duplicate/extra/mismatched prompt, target or prediction ID")
    _validate_serialized_inputs(prompts, targets)
    identity = {key: answers[0].get(key) for key in MODEL_KEYS} if answers else {}
    if answers and (
        not all(identity.values())
        or any({key: row.get(key) for key in MODEL_KEYS} != identity for row in answers)
    ):
        raise ValueError("Prediction model identity incomplete or inconsistent")
    sections: dict[str, dict[str, Counter[str]]] = {
        "by_task": defaultdict(Counter),
        "by_task_type": defaultdict(Counter),
        "by_language_type": defaultdict(Counter),
    }
    for target in targets:
        prediction = by_answer.get(target["id"])
        valid = correct = False
        if prediction is not None:
            if prediction.get("source_input_sha256") != target["source_input_sha256"]:
                raise ValueError("Prediction input fingerprint mismatch")
            native = prediction.get("answers")
            if isinstance(native, dict) and set(native) == {"decision"}:
                valid, correct = _native_choice(native["decision"], target)
        keys = {
            "by_task": target["task"],
            "by_task_type": f"{target['task']}/{target['task_type']}",
            "by_language_type": f"{target['language']}/{target['task_type']}",
        }
        for section, key in keys.items():
            counter = sections[section][key]
            counter["questions"] += 1
            counter["correct"] += int(correct)
            counter["invalid_or_missing"] += int(not valid)
    return {
        "schema_version": VERSION + "-score/1",
        "scope": "exposed supplementary DEV diagnostic, not release ranking",
        "manifest_sha256": file_digest(manifest_path),
        "predictions_sha256": file_digest(predictions),
        "model": identity,
        "questions": len(targets),
        "observable_group_count": len({row["group_id"] for row in targets}),
        **{
            section: {
                key: {
                    **dict(counts),
                    "accuracy": counts["correct"] / counts["questions"],
                }
                for key, counts in sorted(counters.items())
            }
            for section, counters in sections.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_args = sub.add_parser("build")
    build_args.add_argument("--zh-root", type=Path, required=True)
    build_args.add_argument("--ru-root", type=Path, required=True)
    build_args.add_argument("--massive-archive", type=Path, required=True)
    build_args.add_argument("--massive-v1-train", type=Path, required=True)
    build_args.add_argument("--massive-v2-train", type=Path, required=True)
    build_args.add_argument("--output", type=Path, required=True)
    score_args = sub.add_parser("score")
    score_args.add_argument("--panel", type=Path, required=True)
    score_args.add_argument("--predictions", type=Path, required=True)
    score_args.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(
            args.zh_root,
            args.ru_root,
            args.massive_archive,
            args.output,
            args.massive_v1_train,
            args.massive_v2_train,
        )
        print(
            json.dumps(
                {
                    "manifest_sha256": file_digest(args.output / "manifest.json"),
                    "eligible_questions": result["eligible_question_rows"],
                    "held": result["held_upstream_rows"],
                },
                sort_keys=True,
            )
        )
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = score(args.panel, args.predictions)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "score_sha256": file_digest(args.output),
                    "questions": result["questions"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
