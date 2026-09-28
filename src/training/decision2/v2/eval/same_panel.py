"""Formal same-panel runner: JevArena v3 (typed FINAL + CSS15) and JevBench public 231.

Every number this produces is **post-key same-panel** evidence: the v3 keys were
opened earlier in the project. JevBench public 231 is a public-subset rerun, not
the official sealed JevBench rank. Missing, invalid and over-budget answers stay
in every denominator.

Each step writes one receipt into the run directory:

  collect  native inference per panel (inside the pinned image)   -> COLLECT.json
  adopt    reuse earlier predictions whose identity was verified  -> ADOPT.json
  seal     gold-free coverage and identity seal                   -> SEAL.json
  report   frozen scorers, v3 composite and slices (after SEAL)   -> REPORT.json/.md
  compare  joint paired bootstrap against a comparator run        -> PAIRED-vs-<name>.json
  table    same-panel matrix from several REPORT.json files       -> markdown/JSON
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2.eval import adapters as adapter_registry
from v2.eval import panels

FORMAL_PANELS = ("typed-final", "css15", "public231")
REPORT_SCHEMA = "dev2-same-panel-report/1"
LABEL = "post-key same-panel"
LONG_INPUT_CHARS = 4000
PAIRED_REPLICATES = 5000
PAIRED_SEED = 20260927
SOURCE_ROOT = Path(__file__).resolve().parents[2]
TRACKED_SOURCES = (
    "v2/eval/same_panel.py",
    "v2/eval/panels.py",
    "v2/eval/adapters.py",
    "benchmark/score.py",
    "benchmark/generate.py",
    "transfer/score.py",
    "transfer/build.py",
    "jev_arena/jevbench_public.py",
    "jev_arena/compare_v3.py",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    return panels.sha_file(path)


def input_digest(state: Any, questions: Any) -> str:
    encoded = json.dumps(
        {"state": state, "questions": questions},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> str:
    data = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb" if exclusive else "wb") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    return hashlib.sha256(data).hexdigest()


def source_hashes(root: Path = SOURCE_ROOT) -> dict[str, str | None]:
    return {
        name: sha_file(root / name) if (root / name).is_file() else None
        for name in TRACKED_SOURCES
    }


def mirror_receipt(root: Path = SOURCE_ROOT) -> dict[str, Any] | None:
    for candidate in (root, *root.parents):
        receipt = candidate / ".dev2-mirror.json"
        if receipt.is_file():
            return json.loads(receipt.read_text(encoding="utf-8"))
    return None


def prediction_path(run_dir: Path, panel: str) -> Path:
    return run_dir / "output" / f"{panel}.predictions.jsonl"


# ---------------------------------------------------------------- collect


def runtime_probe(python: str, env: dict[str, str]) -> dict[str, Any]:
    code = (
        "import json,sys\nout={'python':sys.version.split()[0]}\n"
        "for m in ('torch','transformers','tokenizers','safetensors','numpy','fla','triton'):\n"
        "    try:\n        mod=__import__(m); out[m]=getattr(mod,'__version__','?')\n"
        "    except Exception: out[m]=None\n"
        "try:\n    import torch; out['hip']=torch.version.hip; out['devices']=torch.cuda.device_count()\n"
        "    out['device_name']=torch.cuda.get_device_name(0) if out['devices'] else None\n"
        "except Exception: pass\nprint(json.dumps(out))\n"
    )
    try:
        result = subprocess.run(
            [python, "-c", code], env=env, capture_output=True, text=True, timeout=300
        )
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def collector_env(extra_pythonpath: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    path = [str(SOURCE_ROOT), *extra_pythonpath]
    if env.get("PYTHONPATH"):
        path.append(env["PYTHONPATH"])
    env.update(
        PYTHONPATH=":".join(path),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def collect(args: argparse.Namespace) -> int:
    adapter = adapter_registry.load(args.adapter, args.adapter_spec)
    run_dir: Path = args.run_dir
    receipt_path = run_dir / ("SMOKE.json" if args.max_items else "COLLECT.json")
    if receipt_path.exists():
        raise FileExistsError(f"{receipt_path} exists; use a new run directory")
    values = {
        "model": str(args.model_path),
        "revision": args.revision,
        "device": args.device,
        "model_id": args.model_id or adapter.model_id or "",
    }
    for entry in args.extra:
        key, _, value = entry.partition("=")
        values[key] = value
    env = collector_env(args.pythonpath)
    receipt: dict[str, Any] = {
        "schema": "dev2-same-panel-collect/1",
        "label": LABEL,
        "started_utc": utc_now(),
        "adapter": adapter.describe(),
        "adapter_module_sha256": adapter_registry.module_sha256(SOURCE_ROOT, adapter),
        "model_path": str(args.model_path),
        "model_revision": args.revision,
        "model_id": values["model_id"] or None,
        "extra": {
            k: v for k, v in values.items() if k not in {"model", "revision", "device"}
        },
        "image_id": os.environ.get("DEV2_IMAGE_ID"),
        "visible_devices": {
            key: os.environ.get(key)
            for key in (
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "gpu_label": os.environ.get("DEV2_GPU_LABEL"),
        "source_mirror": mirror_receipt(),
        "sources": source_hashes(),
        "runtime": runtime_probe(adapter.python, env),
        "panels": [],
    }
    status = 0
    for panel in args.panels:
        prompts = panels.path(args.panel_root, panel, "prompts")
        if sha_file(prompts) != panels.ALL[panel]["prompts_sha256"]:
            raise ValueError(f"{panel}: gold-free prompts differ from the frozen panel")
        output_dir = run_dir / ("smoke" if args.max_items else "output")
        output = output_dir / f"{panel}.predictions.jsonl"
        log = run_dir / "logs" / f"{panel}{'.smoke' if args.max_items else ''}.log"
        output_dir.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = adapter.command({**values, "input": str(prompts), "output": str(output)})
        if args.max_items:
            argv += ["--max-items", str(args.max_items)]
        started, wall = utc_now(), time.perf_counter()
        with log.open("ab") as log_file:
            code = subprocess.run(
                argv,
                cwd=SOURCE_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            ).returncode
        entry = {
            "panel": panel,
            "argv": argv,
            "utc_start": started,
            "utc_end": utc_now(),
            "wall_seconds": time.perf_counter() - wall,
            "exit_code": code,
            "output_sha256": sha_file(output) if output.is_file() else None,
            "output_rows": len(read_jsonl(output)) if output.is_file() else 0,
            "expected_rows": panels.ALL[panel]["originals"],
            "log_sha256": sha_file(log),
        }
        receipt["panels"].append(entry)
        print(
            json.dumps(
                {
                    k: entry[k]
                    for k in ("panel", "exit_code", "wall_seconds", "output_rows")
                }
            )
        )
        if code != 0:
            status = code
            receipt["stopped_after_failure"] = panel
            break
    receipt["ended_utc"] = utc_now()
    receipt["gpu_wall_seconds"] = sum(p["wall_seconds"] for p in receipt["panels"])
    write_json(receipt_path, receipt)
    return status


# ------------------------------------------------------------------ adopt


def adopt(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir
    if (run_dir / "ADOPT.json").exists() or (run_dir / "COLLECT.json").exists():
        raise FileExistsError("run directory already has predictions")
    entries = {}
    for panel in FORMAL_PANELS:
        source = getattr(args, panel.replace("-", "_"))
        if source is None:
            continue
        expected = dict(item.split("=", 1) for item in args.expect_sha256).get(panel)
        digest = sha_file(source)
        if expected and digest != expected:
            raise ValueError(f"{panel}: {digest} differs from the expected {expected}")
        target = prediction_path(run_dir, panel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entries[panel] = {"source": str(source), "sha256": digest}
    receipts = {}
    for item in args.prior_receipt:
        path = Path(item)
        receipts[path.name] = sha_file(path)
    write_json(
        run_dir / "ADOPT.json",
        {
            "schema": "dev2-same-panel-adopt/1",
            "label": LABEL,
            "adopted_utc": utc_now(),
            "reason": args.reason,
            "predictions": entries,
            "prior_receipts_sha256": receipts,
            "prior_gpu_hours": args.prior_gpu_hours,
        },
    )
    return 0


# ------------------------------------------------------------------- seal


def seal(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir
    target = run_dir / "SEAL.json"
    if target.exists():
        raise FileExistsError(target)
    provenance = next(
        (name for name in ("COLLECT.json", "ADOPT.json") if (run_dir / name).is_file()),
        None,
    )
    if provenance is None:
        raise FileNotFoundError("seal needs COLLECT.json or ADOPT.json")
    result: dict[str, Any] = {
        "schema": "dev2-same-panel-seal/1",
        "label": LABEL,
        "sealed_utc": utc_now(),
        "gold_or_scores_read": False,
        "provenance": {provenance: sha_file(run_dir / provenance)},
        "sources": source_hashes(),
        "panels": {},
    }
    for panel in FORMAL_PANELS:
        predictions = prediction_path(run_dir, panel)
        if not predictions.is_file():
            continue
        prompts = read_jsonl(panels.path(args.panel_root, panel, "prompts"))
        expected = {row["id"]: row for row in prompts}
        seen: set[str] = set()
        identity: dict[str, Counter] = defaultdict(Counter)
        null_slots = 0
        for row in read_jsonl(predictions):
            item_id = row.get("id")
            if item_id not in expected or item_id in seen:
                raise ValueError(
                    f"{panel}: unknown or duplicate prediction id {item_id!r}"
                )
            prompt = expected[item_id]
            if row.get("source_input_sha256") != input_digest(
                prompt["state"], prompt["questions"]
            ):
                raise ValueError(
                    f"{panel}: {item_id} was predicted from different input"
                )
            answers = row.get("answers")
            if not isinstance(answers, dict) or set(answers) != set(
                prompt["questions"]
            ):
                raise ValueError(
                    f"{panel}: {item_id} answer keys differ from question keys"
                )
            null_slots += sum(value is None for value in answers.values())
            for key in (
                "model_id",
                "model_revision",
                "adapter_version",
                "backend",
                "model_config_sha256",
            ):
                if key in row:
                    identity[key][json.dumps(row[key])] += 1
            seen.add(item_id)
        result["panels"][panel] = {
            "prompts_sha256": panels.ALL[panel]["prompts_sha256"],
            "predictions_sha256": sha_file(predictions),
            "originals": len(expected),
            "predicted_originals": len(seen),
            "missing_originals": len(expected) - len(seen),
            "answer_slots": sum(len(row["questions"]) for row in prompts),
            "null_answer_slots": null_slots,
            "identity": {key: dict(counts) for key, counts in identity.items()},
        }
    if not result["panels"]:
        raise FileNotFoundError("no prediction files to seal")
    digest = write_json(target, result)
    print(json.dumps({"seal_sha256": digest, "panels": sorted(result["panels"])}))
    return 0


# ----------------------------------------------------------------- report

_STOPWORDS = {
    "en": "the and of to is in that it for with as was on be this are not you have but",
    "es": "el la de que y en los las por una con para es del se no lo como más pero",
    "fr": "le la les des et est une que pour dans pas qui sur au du avec ce il elle sont",
    "de": "der die das und ist nicht ein eine zu mit sich auf den von für dem im des auch wird",
    "pt": "o a os as de que e do da em um uma para com não por se mais como mas",
    "it": "il di che e la per un una non sono con del della le gli si come ma anche più",
    "nl": "de het een en van is dat niet op te met zijn voor ook maar als bij er aan om",
}
_STOP_SETS = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}


def text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in text_values(item)]
    if isinstance(value, list):
        return [text for item in value for text in text_values(item)]
    return []


def language_guess(prompt: dict[str, Any]) -> str:
    """Frozen heuristic screen: script share first, then stopword votes."""
    text = " ".join(
        text_values(prompt.get("state")) + text_values(prompt.get("questions"))
    )
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        latin = sum("LATIN" in unicodedata.name(ch, "") for ch in letters) / len(
            letters
        )
        if latin < 0.8:
            return "non-latin"
    tokens = re.findall(r"[^\W\d_]+", text.lower())
    votes = {
        lang: sum(token in words for token in tokens)
        for lang, words in _STOP_SETS.items()
    }
    best = max(votes, key=lambda lang: (votes[lang], lang == "en"))
    return best if votes[best] >= 5 and votes[best] > votes["en"] else "en"


def input_chars(prompt: dict[str, Any]) -> int:
    return len(
        json.dumps(
            {"state": prompt["state"], "questions": prompt["questions"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def slice_summary(outcomes: list[tuple[bool, bool]]) -> dict[str, Any]:
    n = len(outcomes)
    return {
        "n": n,
        "correct": sum(correct for correct, _ in outcomes),
        "invalid": sum(not valid for _, valid in outcomes),
        "accuracy_all": (sum(correct for correct, _ in outcomes) / n) if n else None,
    }


def latency_summary(rows: list[dict[str, Any]], slots: int) -> dict[str, Any]:
    values = [
        float(r["latency_ms"])
        for r in rows
        if type(r.get("latency_ms")) in (int, float) and math.isfinite(r["latency_ms"])
    ]
    total_s = sum(values) / 1000
    return {
        "n": len(values),
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": statistics.mean(values) if values else None,
        "sum_seconds": total_s,
        "prompts_per_second": len(values) / total_s if total_s else None,
        "answer_slots_per_second": slots / total_s if total_s else None,
    }


def count_safetensors(root: Path) -> dict[str, Any]:
    total, files = 0, []
    for file in sorted(root.rglob("*.safetensors")):
        with file.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        count = sum(
            math.prod(meta["shape"])
            for name, meta in header.items()
            if name != "__metadata__"
        )
        total += count
        files.append({"file": str(file.relative_to(root)), "parameters": count})
    return {"parameters": total, "files": files}


def typed_item_outcomes(
    gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]
):
    from benchmark.score import evaluate_answer

    per_item: dict[str, list[tuple[bool, bool]]] = {}
    for item_id, item in gold.items():
        answers = (predictions.get(item_id) or {}).get("answers")
        outcomes = []
        for key, question in item["questions"].items():
            if not isinstance(answers, dict) or set(answers) != set(item["questions"]):
                outcomes.append((False, False))
                continue
            result = evaluate_answer(question, item["gold"][key], answers[key])
            outcomes.append((bool(result.get("correct")), result.get("status") == "ok"))
        per_item[item_id] = outcomes
    return per_item


def identity_value(seal_panel: dict[str, Any], key: str) -> Any:
    values = seal_panel["identity"].get(key, {})
    if len(values) > 1:
        raise ValueError(f"predictions mix several {key} values: {sorted(values)}")
    return json.loads(next(iter(values))) if values else None


def report(args: argparse.Namespace) -> int:
    from benchmark.score import load_jsonl, score_suite
    from jev_arena import jevbench_public
    from transfer.score import evaluate as evaluate_css
    from transfer.score import score as score_css

    run_dir: Path = args.run_dir
    seal_path = run_dir / "SEAL.json"
    sealed = json.loads(seal_path.read_text(encoding="utf-8"))
    for panel, entry in sealed["panels"].items():
        if sha_file(prediction_path(run_dir, panel)) != entry["predictions_sha256"]:
            raise ValueError(f"{panel}: predictions changed after the seal")
    panels.verify(args.panel_root, list(sealed["panels"]))
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    first = next(iter(sealed["panels"].values()))
    model_id = args.model_id or identity_value(first, "model_id")
    revision = args.revision or identity_value(first, "model_revision")
    out: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "label": LABEL,
        "model": {
            "label": args.label,
            "tier": args.tier,
            "family": args.family,
            "model_id": model_id,
            "revision": revision,
            "adapter_version": identity_value(first, "adapter_version"),
            "backend": identity_value(first, "backend"),
        },
        "seal_sha256": sha_file(seal_path),
        "provenance": sealed["provenance"],
        "sources": source_hashes(),
        "panels": {},
        "slices": {"long_input_chars_threshold": LONG_INPUT_CHARS},
        "latency": {},
        "invalid": {},
    }
    provenance_file = run_dir / next(iter(sealed["provenance"]))
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    out["gpu_wall_seconds"] = provenance.get("gpu_wall_seconds")
    out["reused"] = provenance.get("schema", "").endswith("adopt/1")
    out["prior_gpu_hours"] = provenance.get("prior_gpu_hours")
    out["runtime"] = provenance.get("runtime")
    out["image_id"] = provenance.get("image_id")
    out["batch_policy"] = args.batch_policy or (provenance.get("adapter") or {}).get(
        "batch_policy"
    )

    params: dict[str, Any] = {
        "loaded": args.loaded_parameters,
        "active": args.active_parameters,
        "source": args.parameter_source,
    }
    if args.count_safetensors:
        counted = count_safetensors(args.count_safetensors)
        params["safetensors_count"] = counted["parameters"]
        if params["loaded"] is None:
            params["loaded"] = counted["parameters"]
            params["source"] = params["source"] or "safetensors header element count"
    out["parameters"] = params

    languages: dict[str, Counter] = {}
    if "typed-final" in sealed["panels"]:
        gold_path = panels.path(args.panel_root, "typed-final", "gold")
        pred_path = prediction_path(run_dir, "typed-final")
        typed = score_suite(
            gold_path,
            pred_path,
            model_id or "absent",
            revision or "absent",
            out["model"]["backend"] or "native",
        )
        write_json(scores_dir / "typed-final.score.json", typed, exclusive=False)
        by_type = {
            k: {
                "correct": v["correct_n"],
                "n": v["n"],
                "accuracy": v["accuracy_all"],
                "brier": v["brier"],
                "ece_10": v["ece_10"],
            }
            for k, v in typed["by_type"].items()
        }
        out["panels"]["typed-final"] = {
            "T": typed["macro_family_accuracy"],
            "by_type": by_type,
            "by_family": {k: v["accuracy_all"] for k, v in typed["by_family"].items()},
            "answer_slot_accuracy": typed["overall"]["accuracy_all"],
            "brier": typed["overall"]["brier"],
            "ece_10": typed["overall"]["ece_10"],
            "robustness_pairs": typed["pairs"],
            "score_sha256": sha_file(scores_dir / "typed-final.score.json"),
        }
        out["invalid"]["typed-final"] = {
            "slots": typed["overall"]["n"],
            "invalid_or_missing": typed["overall"]["invalid_or_missing_n"],
            "reasons": typed["invalid_reasons"],
        }
        gold = load_jsonl(gold_path)
        predictions = {row["id"]: row for row in read_jsonl(pred_path)}
        outcomes = typed_item_outcomes(gold, predictions)
        prompts = {
            row["id"]: row
            for row in read_jsonl(
                panels.path(args.panel_root, "typed-final", "prompts")
            )
        }
        long_ids = {i for i, p in prompts.items() if input_chars(p) >= LONG_INPUT_CHARS}
        out["slices"]["typed-final"] = {
            "long": slice_summary([o for i in long_ids for o in outcomes[i]]),
            "short": slice_summary(
                [o for i in prompts if i not in long_ids for o in outcomes[i]]
            ),
        }
        languages["typed-final"] = Counter(language_guess(p) for p in prompts.values())
        out["latency"]["typed-final"] = latency_summary(
            list(predictions.values()), typed["overall"]["n"]
        )

    if "css15" in sealed["panels"]:
        gold_path = panels.path(args.panel_root, "css15", "gold")
        pred_path = prediction_path(run_dir, "css15")
        css = score_css(gold_path, pred_path)
        write_json(scores_dir / "css15.score.json", css, exclusive=False)
        role = css["roles"]["evaluation"]
        tasks = {
            name: {
                "macro_f1": v["macro_f1_all"],
                "accuracy": v["accuracy_all"],
                "n": v["n"],
                "invalid": v["invalid_or_missing_n"],
                "brier_sum": v["brier_sum"],
                "ece_pmax_15": v["ece_pmax_15"],
            }
            for name, v in css["tasks"].items()
            if v["role"] == "evaluation"
        }
        out["panels"]["css15"] = {
            "H": role["median_task_macro_f1_all"],
            "micro_accuracy": role["micro_accuracy_all"],
            "median_task_brier_sum": role["median_task_brier_sum"],
            "median_task_ece_pmax_15": role["median_task_ece_pmax_15"],
            "tasks": tasks,
            "score_sha256": sha_file(scores_dir / "css15.score.json"),
        }
        reasons: Counter = Counter()
        for value in css["tasks"].values():
            reasons.update(value["invalid_reasons"])
        out["invalid"]["css15"] = {
            "items": role["items"],
            "invalid_or_missing": role["items"] - role["valid_items"],
            "reasons": dict(reasons),
        }
        gold_rows = {row["id"]: row for row in read_jsonl(gold_path)}
        predictions = {row["id"]: row for row in read_jsonl(pred_path)}
        prompts = {
            row["id"]: row
            for row in read_jsonl(panels.path(args.panel_root, "css15", "prompts"))
        }
        per_item = {}
        for item_id, row in gold_rows.items():
            result = evaluate_css(row, predictions.get(item_id))
            per_item[item_id] = (bool(result.get("correct")), bool(result["valid"]))
        long_ids = {i for i, p in prompts.items() if input_chars(p) >= LONG_INPUT_CHARS}
        out["slices"]["css15"] = {
            "long": slice_summary([per_item[i] for i in long_ids]),
            "short": slice_summary([per_item[i] for i in prompts if i not in long_ids]),
        }
        languages["css15"] = Counter(language_guess(p) for p in prompts.values())
        out["latency"]["css15"] = latency_summary(
            list(predictions.values()), role["items"]
        )

    if "public231" in sealed["panels"]:
        pred_path = prediction_path(run_dir, "public231")
        seal_public = sealed["panels"]["public231"]
        public_id = identity_value(seal_public, "model_id")
        public_output = scores_dir / "public231.score.json"
        if public_output.exists():
            public_output.unlink()
        public = jevbench_public.score(
            args.panel_root / panels.FORMAL["public231"]["panel_dir"],
            pred_path,
            public_id if public_id is not None else jevbench_public.ABSENT_MODEL_ID,
            identity_value(seal_public, "model_revision"),
            public_output,
        )
        out["panels"]["public231"] = {
            "correct": public["correct"],
            "items": public["items"],
            "valid": public["valid"],
            "tier_macro_accuracy": public["tier_macro_accuracy"],
            "tiers": {
                name: {"correct": t["correct"], "items": t["items"]}
                for name, t in public["tiers"].items()
            },
            "brier_valid": public["brier_valid"],
            "ece_pmax_15": public["ece_pmax_15"],
            "score_sha256": sha_file(public_output),
        }
        out["invalid"]["public231"] = {
            "items": public["items"],
            "invalid_or_missing": public["items"] - public["valid"],
        }
        prompts = {
            row["id"]: row
            for row in read_jsonl(panels.path(args.panel_root, "public231", "prompts"))
        }
        per_item = {
            entry["id"]: (bool(entry["correct"]), bool(entry["valid"]))
            for entry in public["per_item"]
        }
        long_ids = {i for i, p in prompts.items() if input_chars(p) >= LONG_INPUT_CHARS}
        out["slices"]["public231"] = {
            "long": slice_summary([per_item[i] for i in long_ids]),
            "short": slice_summary([per_item[i] for i in prompts if i not in long_ids]),
        }
        languages["public231"] = Counter(language_guess(p) for p in prompts.values())
        out["latency"]["public231"] = latency_summary(
            read_jsonl(pred_path), public["items"]
        )

    out["slices"]["language_screen"] = {
        panel: dict(counts) for panel, counts in languages.items()
    }
    out["slices"]["multilingual_measurable"] = any(
        sum(v for k, v in counts.items() if k != "en") > 0
        for counts in languages.values()
    )
    if {"typed-final", "css15"} <= set(out["panels"]):
        t, h = out["panels"]["typed-final"]["T"], out["panels"]["css15"]["H"]
        out["v3"] = {
            "T": t,
            "H": h,
            "score": 100 * math.sqrt(t * h),
            "formula": "100*sqrt(T*H)",
        }
    write_json(run_dir / "REPORT.json", out, exclusive=False)
    (run_dir / "REPORT.md").write_text(render_markdown([out]), encoding="utf-8")
    print(
        json.dumps(
            {
                "label": args.label,
                "v3": out.get("v3"),
                "public": (out["panels"].get("public231") or {}).get("correct"),
            }
        )
    )
    return 0


# ---------------------------------------------------------------- compare


def compare(args: argparse.Namespace) -> int:
    from jev_arena.compare_v3 import compare as compare_v3

    sides = []
    for run_dir in (args.run_dir, args.comparator_run_dir):
        sealed = json.loads((run_dir / "SEAL.json").read_text(encoding="utf-8"))
        for panel in ("typed-final", "css15"):
            if (
                sha_file(prediction_path(run_dir, panel))
                != sealed["panels"][panel]["predictions_sha256"]
            ):
                raise ValueError(
                    f"{run_dir}: {panel} predictions changed after the seal"
                )
        sides.append(run_dir)
    result = compare_v3(
        panels.path(args.panel_root, "typed-final", "gold"),
        panels.path(args.panel_root, "css15", "gold"),
        prediction_path(sides[0], "typed-final"),
        prediction_path(sides[0], "css15"),
        prediction_path(sides[1], "typed-final"),
        prediction_path(sides[1], "css15"),
        left_name=args.left_name,
        right_name=args.right_name,
        replicates=PAIRED_REPLICATES,
        seed=PAIRED_SEED,
    )
    result["label"] = LABEL
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", args.right_name)
    write_json(args.run_dir / f"PAIRED-vs-{safe}.json", result, exclusive=False)
    print(json.dumps({"delta": result["point"]["delta"], "ci95": result["ci95"]}))
    return 0


# ------------------------------------------------------------------ table


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(reports: list[dict[str, Any]]) -> str:
    header = (
        "| Model | Tier | Loaded params | v3 (post-key) | T | H | Choice | Noul | Score | Public 231 (E/S/H) "
        "| Typed Brier/ECE | Order consistency | Invalid typed/CSS/public | Long CSS acc | p50 ms (typed/CSS) | Source |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | --- | --- |\n"
    )
    rows = []
    for rep in sorted(
        reports,
        key=lambda r: (
            r["model"].get("tier") or "",
            -((r.get("v3") or {}).get("score") or 0),
        ),
    ):
        typed = rep["panels"].get("typed-final", {})
        public = rep["panels"].get("public231", {})
        types = typed.get("by_type", {})
        tiers = public.get("tiers", {})
        order = (typed.get("robustness_pairs") or {}).get("order_invariance", {})
        inv = rep.get("invalid", {})
        params = rep.get("parameters", {})
        loaded = params.get("loaded")
        loaded_text = f"{loaded / 1e9:.3f}B" if isinstance(loaded, int) else "—"
        if params.get("active"):
            loaded_text += f" ({params['active'] / 1e9:.2f}B active)"
        public_text = (
            f"{public['correct']}/{public['items']} ({'/'.join(str(tiers[t]['correct']) for t in ('easy', 'standard', 'hard') if t in tiers)})"
            if public
            else "—"
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    rep["model"].get("label") or "—",
                    rep["model"].get("tier") or "—",
                    loaded_text,
                    fmt((rep.get("v3") or {}).get("score")),
                    fmt((rep.get("v3") or {}).get("T"), 4),
                    fmt((rep.get("v3") or {}).get("H"), 4),
                    *(
                        f"{types[t]['correct']}/{types[t]['n']}" if t in types else "—"
                        for t in ("choice", "noul", "score")
                    ),
                    public_text,
                    f"{fmt(typed.get('brier'))}/{fmt(typed.get('ece_10'))}",
                    fmt(order.get("relation_consistency_all")),
                    "/".join(
                        fmt((inv.get(p) or {}).get("invalid_or_missing"))
                        for p in FORMAL_PANELS
                    ),
                    fmt(
                        (
                            (rep.get("slices", {}).get("css15") or {}).get("long") or {}
                        ).get("accuracy_all")
                    ),
                    "/".join(
                        fmt((rep.get("latency", {}).get(p) or {}).get("p50_ms"), 0)
                        for p in ("typed-final", "css15")
                    ),
                    "reused" if rep.get("reused") else "run",
                ]
            )
            + " |"
        )
    return header + "\n".join(rows) + "\n"


def table(args: argparse.Namespace) -> int:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.report]
    text = render_markdown(reports)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


# ------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)

    p = commands.add_parser("collect")
    common(p)
    p.add_argument("--adapter")
    p.add_argument("--adapter-spec", type=Path)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--model-id")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--panels", type=lambda s: s.split(","), default=list(FORMAL_PANELS))
    p.add_argument(
        "--extra",
        action="append",
        default=[],
        help="KEY=VALUE for adapter placeholders",
    )
    p.add_argument("--pythonpath", action="append", default=["/opt/decision-fla"])
    p.add_argument("--max-items", type=int, help="smoke run into <run-dir>/smoke/")
    p.set_defaults(func=collect)

    p = commands.add_parser("adopt")
    common(p)
    for panel in FORMAL_PANELS:
        p.add_argument(f"--{panel}", dest=panel.replace("-", "_"), type=Path)
    p.add_argument("--expect-sha256", action="append", default=[], help="PANEL=SHA256")
    p.add_argument("--prior-receipt", action="append", default=[])
    p.add_argument("--prior-gpu-hours", type=float)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=adopt)

    p = commands.add_parser("seal")
    common(p)
    p.set_defaults(func=seal)

    p = commands.add_parser("report")
    common(p)
    p.add_argument("--label", required=True)
    p.add_argument("--tier", required=True)
    p.add_argument("--family", default="")
    p.add_argument("--model-id")
    p.add_argument("--revision")
    p.add_argument("--loaded-parameters", type=int)
    p.add_argument("--active-parameters", type=int)
    p.add_argument("--parameter-source")
    p.add_argument("--count-safetensors", type=Path)
    p.add_argument("--batch-policy")
    p.set_defaults(func=report)

    p = commands.add_parser("compare")
    common(p)
    p.add_argument("--comparator-run-dir", type=Path, required=True)
    p.add_argument("--left-name", required=True)
    p.add_argument("--right-name", required=True)
    p.set_defaults(func=compare)

    p = commands.add_parser("table")
    p.add_argument("--report", action="append", required=True)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=table)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
