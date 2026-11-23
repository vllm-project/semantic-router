"""Misroute attribution: error taxonomy over a labeled frozen dataset.

Classes (recipe-lab design §6, per signal family):

- E1_SIGNAL          a truth signal contradicts the record's observed signal
                     evidence (F2: judged per signal from signal_values);
                     requires signal-level truth (truth = {"signals": ...})
- DECISION_MISMATCH  matched decision != expected decision, judged without
                     usable signal-level truth (F1 fallback class; with F2
                     signal truth that is fully verified a decision miss is
                     an E2 policy error instead)
- E2_POLICY          decision or routed model deviates from the oracle-policy
                     expectation, answer wrong
- E3_CAPABILITY      routing exactly as intended, answer still wrong
- SUBOPTIMAL         routing deviated from the policy but the answer was
                     still correct (counted separately, never as an error)
- NO_EVIDENCE        answer correctness unknown (no saved answer /
                     interrupted run); evidence-first: incomplete evidence is
                     never counted as an error
- CORRECT            routed as intended and answered correctly
- POLICY_GAP         truth (label or signal facts) not covered by the
                     declared oracle policy

Input records (one JSON object per line) follow a minimal schema; the
router_replay adapter produces exactly this shape:

  {"case_id": str, "truth": str | {"signals": {...}},
   "matched_decision": str, "routed": str,
   "family": str | absent, "answer_ok": true | false | null,
   "signal_values": {...}, "signal_confidences": {...}}

Truth is either a string label (legacy: the expected decision, resolved via
the oracle policy's label mapping) or an object {"signals": {<name>: fact}}
whose facts are label strings, numbers, or numeric predicates ({gte: 3});
facts drive both the policy's rule evaluation and E1 signal judgment (F2).

Verified against the router source (conclusions carried over from the
replay adapter, tools/recipe-lab/recipe_lab/adapters/router_replay.py):

- signal evidence keys are "<type>:<name>" with an optional ":<label>"
  segment for labelled signals; per key the value falls back from
  signal_values to signal_confidences (pkg/decision engine.go
  signalPredicateValue). Choice-type decision signals store one probability
  per choice under "decision:<name>:<choice>" (hit label = argmax); score-
  type decision signals keep the numeric score under "decision:<name>".
  Other observed key types: classifier (fixed-head, "classifier:<name>:
  <label>"), projection (d3), embedding, structure, keyword, action.
- signal family is therefore derived from the evidence key's type segment
  (F2: no longer a --family CLI constant): classifier -> vela_classify,
  decision -> vela_decisions, projection -> d3; other type segments surface
  verbatim. Records without signal evidence keep the input family tag.

Records written before F1 remain readable: a top-level "pred" field is
accepted as matched_decision.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import (
    ContractError,
    Expectation,
    OraclePolicy,
    matches_condition,
    validate_signal_fact,
)

E1_SIGNAL = "E1_signal"
DECISION_MISMATCH = "decision_mismatch"
E2_POLICY = "E2_policy"
E3_CAPABILITY = "E3_capability"
SUBOPTIMAL = "suboptimal_routing"
NO_EVIDENCE = "no_evidence"
CORRECT = "correct"
POLICY_GAP = "policy_gap"

ERROR_CLASSES = (E1_SIGNAL, DECISION_MISMATCH, E2_POLICY, E3_CAPABILITY)
ALL_CLASSES = (
    CORRECT,
    E1_SIGNAL,
    DECISION_MISMATCH,
    E2_POLICY,
    E3_CAPABILITY,
    SUBOPTIMAL,
    NO_EVIDENCE,
    POLICY_GAP,
)

REQUIRED_FIELDS = ("case_id", "truth", "matched_decision", "routed", "answer_ok")

# Evidence key type -> signal family (verified key format, see module
# docstring). Unlisted type segments surface verbatim as the family.
KEY_TYPE_PRIORITY = (
    "decision",
    "classifier",
    "projection",
    "embedding",
    "structure",
    "keyword",
    "action",
)
FAMILY_BY_KEY_TYPE = {
    "classifier": "vela_classify",
    "decision": "vela_decisions",
    "projection": "d3",
}

# Per-family fix directions for the report (design §6 taxonomy).
FAMILY_HINTS = {
    "vela_classify": "threshold calibration (fixed head)",
    "vela_decisions": "threshold calibration / signal-question engineering",
    "d3": "projection-side calibration",
}


def _require_number_map(value: Any, where: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ContractError(
            f"{where} must be a mapping of signal key -> number, got {value!r}"
        )
    for key, number in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ContractError(f"{where} keys must be non-empty strings, got {key!r}")
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ContractError(
                f"{where}[{key!r}] must be a finite number, got {number!r}"
            )
    return dict(value)


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    truth: str | dict
    matched_decision: str
    routed: str
    answer_ok: bool | None
    family: str | None = None
    signal_values: dict[str, float] = field(default_factory=dict)
    signal_confidences: dict[str, float] = field(default_factory=dict)

    @property
    def truth_signals(self) -> dict[str, Any] | None:
        """The case's truth signal facts, or None for string (label) truth."""
        if isinstance(self.truth, dict):
            return self.truth.get("signals")
        return None

    @classmethod
    def from_dict(cls, raw: dict, lineno: int) -> CaseInput:
        if "matched_decision" not in raw and "pred" in raw:
            # pre-F1 field name; kept readable for compatibility (F1)
            raw = {**raw, "matched_decision": raw["pred"]}
        for name in REQUIRED_FIELDS:
            if name not in raw:
                raise ContractError(
                    f"attribution record line {lineno} is missing required field: {name}"
                )
        answer_ok = raw["answer_ok"]
        if answer_ok is not None and not isinstance(answer_ok, bool):
            raise ContractError(
                f"attribution record line {lineno}: answer_ok must be true, false or null (no evidence), "
                f"got {answer_ok!r}"
            )
        for name in ("case_id", "matched_decision", "routed"):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ContractError(
                    f"attribution record line {lineno}: {name} must be a non-empty string"
                )
        truth = _parse_truth(raw["truth"], lineno)
        family = raw.get("family")
        if family is not None and (not isinstance(family, str) or not family.strip()):
            raise ContractError(
                f"attribution record line {lineno}: family must be a non-empty string when present"
            )
        where = f"attribution record line {lineno}"
        return cls(
            case_id=raw["case_id"],
            truth=truth,
            matched_decision=raw["matched_decision"],
            routed=raw["routed"],
            answer_ok=answer_ok,
            family=family,
            signal_values=_require_number_map(
                raw.get("signal_values", {}), f"{where}: signal_values"
            ),
            signal_confidences=_require_number_map(
                raw.get("signal_confidences", {}), f"{where}: signal_confidences"
            ),
        )


def _parse_truth(truth: Any, lineno: int) -> str | dict:
    """Validate the truth field: a non-empty string label, or an object with
    a non-empty "signals" mapping of validated facts (F2)."""
    where = f"attribution record line {lineno}: truth"
    if isinstance(truth, str):
        if not truth.strip():
            raise ContractError(f"{where} must be a non-empty string")
        return truth
    if isinstance(truth, dict):
        unexpected = [k for k in truth if k != "signals"]
        if unexpected:
            raise ContractError(
                f"{where} object supports only the 'signals' key, got: {unexpected}"
            )
        signals = truth.get("signals")
        if not isinstance(signals, dict) or not signals:
            raise ContractError(
                f"{where} object must carry a non-empty 'signals' mapping"
            )
        for name, fact in signals.items():
            if not isinstance(name, str) or not name.strip():
                raise ContractError(
                    f"{where}.signals names must be non-empty strings, got {name!r}"
                )
            try:
                validate_signal_fact(fact, f"{where}.signals[{name!r}]")
            except ContractError as exc:
                raise ContractError(f"{exc} (record line {lineno})") from exc
        return truth
    raise ContractError(
        f"{where} must be a string label or an object with a 'signals' mapping, got {truth!r}"
    )


def merged_signal_evidence(case: CaseInput) -> dict[str, float]:
    """Merge the record's signal evidence with the router's per-key fallback
    order (signal_values first, then signal_confidences)."""
    merged: dict[str, float] = dict(case.signal_confidences)
    merged.update(case.signal_values)
    return merged


def observe_signal(name: str, evidence: dict[str, float]) -> dict[str, Any]:
    """Read one signal's observed state from replay signal evidence.

    Returns {"label": hit-label-or-None, "score": score-or-None,
    "key_type": type segment of the best evidence key-or-None}. The hit
    label is the argmax choice over "<type>:<name>:<label>" keys; the score
    comes from the bare "<type>:<name>" key. Key-type search follows
    KEY_TYPE_PRIORITY (verified key format, see module docstring)."""
    label: str | None = None
    score: float | None = None
    key_type: str | None = None
    for key_type_candidate in KEY_TYPE_PRIORITY:
        if score is None:
            value = evidence.get(f"{key_type_candidate}:{name}")
            if value is not None:
                score = value
                key_type = key_type_candidate
        prefix = f"{key_type_candidate}:{name}:"
        if label is None:
            label_items = [(k, v) for k, v in evidence.items() if k.startswith(prefix)]
            if label_items:
                best_key, _ = max(label_items, key=lambda item: item[1])
                label = best_key[len(prefix) :]
                key_type = key_type_candidate
    return {"label": label, "score": score, "key_type": key_type}


def family_for_signal(name: str, evidence: dict[str, float]) -> str | None:
    """Signal family for one signal, derived from its evidence key's type
    segment (verified key format; F2 replaces the --family constant)."""
    observed = observe_signal(name, evidence)
    if observed["key_type"] is None:
        return None
    return FAMILY_BY_KEY_TYPE.get(observed["key_type"], observed["key_type"])


def derive_family(case: CaseInput) -> str | None:
    """Case family: derived from the record's own signal evidence (first
    truth signal, else the first evidence key's type segment); records
    without evidence keep the input family tag."""
    evidence = merged_signal_evidence(case)
    signals = case.truth_signals
    if signals:
        for name in signals:
            family = family_for_signal(name, evidence)
            if family is not None:
                return family
    for key in evidence:
        key_type = key.split(":", 1)[0]
        return FAMILY_BY_KEY_TYPE.get(key_type, key_type)
    return case.family


def _judge_signal(name: str, fact: Any, evidence: dict[str, float]) -> dict[str, Any]:
    """Judge one truth-signal fact against the observed evidence.

    Returns {signal, truth, observed, ok, judged}; judged=False means the
    record carries no readable evidence for this signal, which is never an
    E1 (evidence-first), but leaves the case's signal verification
    incomplete."""
    observed = observe_signal(name, evidence)
    if isinstance(fact, str):
        if observed["label"] is None:
            return {
                "signal": name,
                "truth": fact,
                "observed": None,
                "ok": False,
                "judged": False,
            }
        return {
            "signal": name,
            "truth": fact,
            "observed": observed["label"],
            "ok": observed["label"] == fact,
            "judged": True,
        }
    if observed["score"] is None:
        return {
            "signal": name,
            "truth": fact,
            "observed": None,
            "ok": False,
            "judged": False,
        }
    return {
        "signal": name,
        "truth": fact,
        "observed": observed["score"],
        "ok": matches_condition(observed["score"], fact),
        "judged": True,
    }


def _expectation_for(case: CaseInput, policy: OraclePolicy) -> Expectation | None:
    """Resolve the oracle-policy expectation for one case: signal-fact truth
    evaluates the declared rules top-down (F2); string truth uses the legacy
    label mapping (the label is itself the intended decision)."""
    signals = case.truth_signals
    if signals is not None:
        return policy.evaluate(signals)
    expected_model = policy.expected(case.truth)
    if expected_model is None:
        return None
    return Expectation(decision=case.truth, models=(expected_model,))


def classify_case(case: CaseInput | dict, policy: OraclePolicy) -> dict[str, Any]:
    """Classify one case into the attribution taxonomy (branch-explicit).

    Order (evidence-first; F2 multi-signal oracle):
      policy_gap -> E1_signal (a truth signal contradicts observed evidence)
      -> no_evidence (answer unknown) -> correct / suboptimal_routing /
      E2_policy / E3_capability; a decision miss without fully verified
      signal truth stays decision_mismatch (F1 fallback).

    Accepts a CaseInput or a raw record dict."""
    if isinstance(case, dict):
        case = CaseInput.from_dict(case, 0)
    expectation = _expectation_for(case, policy)
    evidence = merged_signal_evidence(case)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "truth": case.truth,
        "matched_decision": case.matched_decision,
        "routed": case.routed,
        "expected": (
            (expectation.models[0] if expectation.models else expectation.decision)
            if expectation
            else None
        ),
        "expected_decision": expectation.decision if expectation else None,
        "expected_models": list(expectation.models) if expectation else [],
        "family": derive_family(case),
        "answer_ok": case.answer_ok,
    }
    if expectation is None:
        row["error_class"] = POLICY_GAP
        return row

    signals = case.truth_signals
    verified = False
    if signals is not None and evidence:
        judgments = [
            _judge_signal(name, fact, evidence) for name, fact in signals.items()
        ]
        row["signals"] = judgments
        contradictions = [j for j in judgments if j["judged"] and not j["ok"]]
        if contradictions:
            failing = contradictions[0]
            row["error_class"] = E1_SIGNAL
            row["e1_signal"] = failing["signal"]
            row["e1_observed"] = failing["observed"]
            row["family"] = (
                family_for_signal(failing["signal"], evidence) or row["family"]
            )
            return row
        verified = all(j["judged"] for j in judgments)

    if case.answer_ok is None:
        row["error_class"] = NO_EVIDENCE
        return row
    decision_ok = case.matched_decision == expectation.decision
    models_ok = not expectation.models or case.routed in expectation.models
    routed_as_expected = decision_ok and models_ok
    if case.answer_ok:
        row["error_class"] = CORRECT if routed_as_expected else SUBOPTIMAL
        return row
    if routed_as_expected:
        row["error_class"] = E3_CAPABILITY
        return row
    if not decision_ok and not verified:
        # the decision missed and the signal facts are not fully verified:
        # a signal error cannot be ruled out (F1 fallback class)
        row["error_class"] = DECISION_MISMATCH
        return row
    row["error_class"] = E2_POLICY
    return row


def load_records(path: str | Path) -> list[CaseInput]:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"attribution records file not found: {path}")
    cases: list[CaseInput] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"attribution records line {lineno} is not valid JSON: {exc}"
                ) from exc
            cases.append(CaseInput.from_dict(raw, lineno))
    return cases


def attribute(
    cases: list[CaseInput], policy: OraclePolicy, dataset_hash: str = ""
) -> list[dict[str, Any]]:
    rows = [classify_case(c, policy) for c in cases]
    for row in rows:
        if dataset_hash:
            row["dataset_hash"] = dataset_hash
    return rows


def summarize(rows: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    counts = Counter(row["error_class"] for row in rows)
    answered_classes = (CORRECT, E2_POLICY, E3_CAPABILITY, SUBOPTIMAL)
    n_answered = sum(counts[c] for c in answered_classes)
    n_correct = counts[CORRECT]
    accuracy = (n_correct / n_answered) if n_answered else None

    e1_rows = [r for r in rows if r["error_class"] == E1_SIGNAL]
    # confusion pairs use the expected decision (string truth resolves to the
    # same value; signal-fact truth has no bare label to print)
    confusion = Counter(
        (r["expected_decision"], r["matched_decision"]) for r in e1_rows
    )
    mm_rows = [r for r in rows if r["error_class"] == DECISION_MISMATCH]
    mm_confusion = Counter(
        (r["expected_decision"], r["matched_decision"]) for r in mm_rows
    )
    e2_rows = [r for r in rows if r["error_class"] == E2_POLICY]
    deviations = Counter((r["routed"], r["expected"]) for r in e2_rows)
    e3_by_family = Counter(
        r.get("family") or "unknown" for r in rows if r["error_class"] == E3_CAPABILITY
    )
    e1_by_family = Counter(r.get("family") or "unknown" for r in e1_rows)
    e1_by_signal = Counter(r.get("e1_signal") or "unknown" for r in e1_rows)

    return {
        "total": len(rows),
        "class_counts": {cls: counts.get(cls, 0) for cls in ALL_CLASSES},
        "answered": n_answered,
        "correct": n_correct,
        "accuracy": accuracy,
        "e1_by_family": dict(e1_by_family),
        "e1_by_signal": dict(e1_by_signal),
        "e3_by_family": dict(e3_by_family),
        "e1_confusion_top": [
            {"expected_decision": d, "matched_decision": m, "count": n}
            for (d, m), n in confusion.most_common(top_k)
        ],
        "decision_mismatch_top": [
            {"expected_decision": d, "matched_decision": m, "count": n}
            for (d, m), n in mm_confusion.most_common(top_k)
        ],
        "e2_deviations_top": [
            {"routed": r, "expected": e, "count": n}
            for (r, e), n in deviations.most_common(top_k)
        ],
    }


def render_markdown(summary: dict[str, Any], dataset_hash: str = "") -> str:
    counts = summary["class_counts"]
    accuracy = summary["accuracy"]
    accuracy_text = (
        f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A (no answered cases)"
    )
    lines = [
        "# Misroute attribution report",
        "",
        f"- dataset_hash: `{dataset_hash or 'n/a'}`",
        f"- total cases: {summary['total']}",
        f"- routed-outcome accuracy: **{accuracy_text}**"
        f" ({summary['correct']}/{summary['answered']} answered cases)",
        "",
        "## Class distribution",
        "",
        "| class | count |",
        "|---|---|",
    ]
    for cls in ALL_CLASSES:
        lines.append(f"| {cls} | {counts[cls]} |")
    lines += ["", "## E1 by signal family", ""]
    if summary["e1_by_family"]:
        for family, n in sorted(summary["e1_by_family"].items()):
            hint = FAMILY_HINTS.get(family, "family-specific calibration")
            lines.append(f"- `{family}`: {n} cases -> directional hint: {hint}")
    else:
        lines.append("- none")
    lines += ["", "## E1 by failing signal", ""]
    if summary["e1_by_signal"]:
        for signal, n in sorted(summary["e1_by_signal"].items()):
            lines.append(f"- `{signal}`: {n} cases")
    else:
        lines.append("- none")
    lines += ["", "## E1 confusion pairs (top)", ""]
    if summary["e1_confusion_top"]:
        for item in summary["e1_confusion_top"]:
            lines.append(
                f"- expected `{item['expected_decision']}` but decision matched `{item['matched_decision']}`: {item['count']}"
            )
    else:
        lines.append("- none")
    lines += ["", "## decision_mismatch confusion pairs (top)", ""]
    if summary["decision_mismatch_top"]:
        for item in summary["decision_mismatch_top"]:
            lines.append(
                f"- expected `{item['expected_decision']}` but decision matched `{item['matched_decision']}`: {item['count']}"
            )
    else:
        lines.append("- none")
    lines += ["", "## E2 routing deviations (top)", ""]
    if summary["e2_deviations_top"]:
        for item in summary["e2_deviations_top"]:
            lines.append(
                f"- routed `{item['routed']}` but policy expects `{item['expected']}`: {item['count']}"
            )
    else:
        lines.append("- none")
    lines += ["", "## E3 by signal family", ""]
    if summary["e3_by_family"]:
        for family, n in sorted(summary["e3_by_family"].items()):
            lines.append(
                f"- `{family}`: {n} cases (routing not at fault; candidate new-signal area)"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return path
