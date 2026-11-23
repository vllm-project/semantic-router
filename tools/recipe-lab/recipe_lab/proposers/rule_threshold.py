"""Rule-level threshold proposer: move every predicate site of one signal.

The signal-level proposer (threshold_search) moves a single decision-signal
default. Real recipes instead re-state the threshold inside decision rules,
often under NOT branches (BUG-3: four `difficulty gte 2.5` rule sites across
four decisions, two of them negated, all calibrated to 1.75 in PR #5568).
This proposer finds those sites, moves all of them (or a selected subset of
decisions) to one new value, and records each moved site in the candidate
sidecar so a later verdict can be attributed to the exact wiring change.

Only decision rules are sites: the signal-level `predicate` default is not
perturbed here. Deterministic by construction, like threshold_search.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..contract import ContractError

_PRED_OPS = ("gte", "gt", "lte", "lt")


@dataclass(frozen=True)
class ThresholdSite:
    """One predicate occurrence inside a decision's rules.

    `path` locates the predicate dict from the config root, e.g.
    ("routing", "decisions", 1, "rules", "conditions", 1, "predicate").
    """

    decision: str
    signal: str
    op: str
    value: float
    negated: bool
    path: tuple


@dataclass(frozen=True)
class RuleThresholdCandidate:
    signal_name: str
    op: str
    old_values: tuple[float, ...]
    new_value: float
    sites: tuple[ThresholdSite, ...]
    ref: str
    config: dict[str, Any]

    def summary(self) -> str:
        olds = "/".join(str(v) for v in self.old_values)
        return (
            f"rule thresholds: signal {self.signal_name!r} {self.op} {olds} -> "
            f"{self.new_value} ({len(self.sites)} site(s))"
        )


def find_threshold_sites(config: dict[str, Any]) -> list[ThresholdSite]:
    """Every decision-rule predicate site, in config order."""
    sites: list[ThresholdSite] = []
    decisions = config.get("routing", {}).get("decisions", [])
    for d_idx, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            continue
        rules = decision.get("rules")
        if not isinstance(rules, dict):
            continue
        _walk(
            rules,
            str(decision.get("name", f"decision-{d_idx}")),
            False,
            ("routing", "decisions", d_idx, "rules"),
            sites,
        )
    return sites


def _walk(
    node: Any,
    decision_name: str,
    negated: bool,
    path: tuple,
    sites: list[ThresholdSite],
) -> None:
    if not isinstance(node, dict):
        return
    operator = node.get("operator")
    if operator:
        child_negated = negated or operator == "NOT"
        conditions = node.get("conditions")
        if isinstance(conditions, list):
            for c_idx, child in enumerate(conditions):
                _walk(
                    child,
                    decision_name,
                    child_negated,
                    (*path, "conditions", c_idx),
                    sites,
                )
        return
    if node.get("type") != "decision":
        return
    predicate = node.get("predicate")
    if not isinstance(predicate, dict):
        return  # a bare signal reference: it uses the signal-level default
    for op in _PRED_OPS:
        if op not in predicate:
            continue
        value = predicate[op]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(
                f"decision {decision_name!r} rule for signal {node.get('name')!r} "
                f"has a non-numeric {op}: {value!r}"
            )
        sites.append(
            ThresholdSite(
                decision=decision_name,
                signal=str(node.get("name")),
                op=op,
                value=float(value),
                negated=negated,
                path=(*path, "predicate"),
            )
        )


def group_sites(
    sites: list[ThresholdSite],
) -> dict[tuple[str, str, float], list[ThresholdSite]]:
    """Group by (signal, op, value): the variables a candidate can move."""
    groups: dict[tuple[str, str, float], list[ThresholdSite]] = {}
    for site in sites:
        groups.setdefault((site.signal, site.op, site.value), []).append(site)
    return groups


def signal_range(config: dict[str, Any], signal_name: str) -> tuple[float, float]:
    """Threshold bounds implied by the signal question; explicit --lo/--hi wins."""
    signals = config.get("routing", {}).get("signals", {}).get("decision", [])
    for signal in signals:
        if isinstance(signal, dict) and signal.get("name") == signal_name:
            question = signal.get("question") or {}
            qtype = question.get("type")
            if qtype == "score":
                levels = question.get("levels") or []
                if levels:
                    return (0.0, float(len(levels) - 1))
            elif qtype in ("noul", "choice", "set"):
                return (0.0, 1.0)
            break
    raise ContractError(
        f"cannot infer the threshold range for signal {signal_name!r}; pass --lo/--hi explicitly"
    )


def propose_rule_thresholds(
    base_config: dict[str, Any],
    signal_name: str,
    *,
    values: list[float] | None = None,
    deltas: list[float] | None = None,
    decisions: list[str] | None = None,
    allow_partial: bool = False,
    from_value: float | None = None,
    op: str | None = None,
    lo: float | None = None,
    hi: float | None = None,
    ref_prefix: str = "config-rev",
) -> list[RuleThresholdCandidate]:
    """One candidate per target value; every selected site moves to it.

    values take precedence over deltas. With deltas the selected sites must
    share one old value (otherwise the delta is ambiguous). Clamped to the
    signal range (or explicit lo/hi), rounded to 6 decimals, deduped, and the
    original value is never a candidate.
    """
    if values is None and deltas is None:
        raise ContractError("propose_rule_thresholds needs one of values or deltas")
    sites = [s for s in find_threshold_sites(base_config) if s.signal == signal_name]
    if from_value is not None:
        sites = [s for s in sites if s.value == float(from_value)]
    if op is not None:
        sites = [s for s in sites if s.op == op]
    elif sites and len({s.op for s in sites}) > 1:
        ops = ", ".join(sorted({s.op for s in sites}))
        raise ContractError(
            f"signal {signal_name!r} has rule sites with mixed operators ({ops}); pass --op"
        )
    if decisions is not None:
        covered = {s.decision for s in sites}
        missing = [d for d in decisions if d not in covered]
        if missing:
            detail = ", ".join(missing)
            if not allow_partial:
                raise ContractError(
                    f"no rule threshold site for signal {signal_name!r} in decision(s) {detail}; "
                    "pass --allow-partial to move only the covered decisions"
                )
            print(
                f"warning: no rule threshold site for signal {signal_name!r} in "
                f"decision(s) {detail}; moving only the covered ones",
                file=sys.stderr,
            )
        sites = [s for s in sites if s.decision in decisions]
    if not sites:
        hint = f" at value {from_value}" if from_value is not None else ""
        raise ContractError(
            f"no rule threshold sites for signal {signal_name!r}{hint}; "
            "run with --list-sites to inspect the config"
        )
    old_values = tuple(sorted({s.value for s in sites}))
    if deltas is not None and len(old_values) > 1:
        raise ContractError(
            f"signal {signal_name!r} rule sites hold mixed values "
            f"({', '.join(str(v) for v in old_values)}); pass --values instead of --deltas"
        )
    if lo is None or hi is None:
        range_lo, range_hi = signal_range(base_config, signal_name)
        lo = range_lo if lo is None else lo
        hi = range_hi if hi is None else hi
    if values is not None:
        targets = [round(min(hi, max(lo, float(v))), 6) for v in values]
    else:
        targets = [round(min(hi, max(lo, old_values[0] + float(d))), 6) for d in deltas]
    deduped: list[float] = []
    for target in targets:
        if target not in old_values and target not in deduped:
            deduped.append(target)
    candidates: list[RuleThresholdCandidate] = []
    for new_value in deduped:
        config = copy.deepcopy(base_config)
        for site in sites:
            node: Any = config
            for key in site.path:
                node = node[key]
            node[site.op] = new_value
        candidates.append(
            RuleThresholdCandidate(
                signal_name=signal_name,
                op=sites[0].op,
                old_values=old_values,
                new_value=new_value,
                sites=tuple(sites),
                ref=f"{ref_prefix}:{signal_name}={new_value}",
                config=config,
            )
        )
    return candidates


def _site_record(site: ThresholdSite, new_value: float) -> dict[str, Any]:
    return {
        "decision": site.decision,
        "signal": site.signal,
        "op": site.op,
        "from": site.value,
        "to": new_value,
        "negated": site.negated,
    }


def write_candidates(
    candidates: list[RuleThresholdCandidate], output_dir: str | Path
) -> list[Path]:
    """Write each candidate config plus its sidecar metadata; returns config paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, candidate in enumerate(candidates):
        path = (
            out_dir
            / f"candidate-{index:03d}-{candidate.signal_name}-{candidate.new_value}.yaml"
        )
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(candidate.config, fh, allow_unicode=True, sort_keys=False)
        sidecar = {
            "kind": "config",
            "ref": candidate.ref,
            "change_summary": candidate.summary(),
            "provenance": {"proposer": "rule-threshold", "insight_refs": []},
            "sites": [_site_record(s, candidate.new_value) for s in candidate.sites],
        }
        path.with_suffix(".candidate.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths
