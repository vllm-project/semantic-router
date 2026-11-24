"""Distribution-calibrated threshold proposer (D).

The rule-level proposer (rule_threshold) moves sites to values a human (or
an agent) names. This proposer names the values itself: read the observed
distribution of one decision signal out of a diagnosis artifact and put the
threshold at a chosen quantile of it (BUG-3: `difficulty gte 2.5` -> the
observed p75, 1.75). Two spellings of the same intent:

- `--quantiles 0.5,0.75,0.9`: one candidate per listed quantile;
- `--target-share 0.25`: the quantile that leaves roughly 25% of the traffic
  on the other side, i.e. quantile 0.75 (maintainers think in "how much
  traffic should reach the lane", the distribution speaks in quantiles).

The proposal itself is delegated to rule_threshold.propose_rule_thresholds,
so sites sharing (signal, op, value) still move together, values are clamped
to the signal range and the old value is never proposed. Quantile values are
rounded to 2 decimals before proposing (1.746 -> 1.75), so the candidate
matches what a maintainer would write into the recipe.

Each candidate sidecar cites the diagnosis it came from
(`diag:<path>@sha256:<hex>#threshold.<decision>.<signal>`); the refs are
verified at generation time, so a ledger never inherits an unverifiable one.
A diagnosis whose config hash does not match the base config is rejected:
stale evidence must not recalibrate a new config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contract import ContractError
from ..diagnose import DIAGNOSIS_KIND, percentile
from ..insights import resolve_insight_refs
from . import rule_threshold

_ANCHOR_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


@dataclass(frozen=True)
class CalibratedProposal:
    """One calibrated candidate plus the evidence text and refs for its sidecar."""

    candidate: rule_threshold.RuleThresholdCandidate
    basis: str
    insight_refs: tuple[str, ...]


def verify_diagnosis_config(diagnosis: dict[str, Any], config_hash: str) -> None:
    """The diagnosis must have been produced over exactly this base config."""
    if diagnosis.get("kind") != DIAGNOSIS_KIND:
        raise ContractError(f"not a {DIAGNOSIS_KIND} artifact")
    recorded = (diagnosis.get("inputs") or {}).get("config_hash") or ""
    if not recorded:
        raise ContractError(
            "diagnosis carries no config hash; regenerate it with 'recipe-lab diagnose'"
        )
    if recorded != config_hash:
        raise ContractError(
            f"diagnosis config hash mismatch: diagnosis recorded {recorded}, "
            f"base config is {config_hash}"
        )


def _signal_entry(diagnosis: dict[str, Any], signal_name: str) -> dict[str, Any]:
    """The distribution entry for one signal; the bare `decision:<name>` key
    wins over per-choice keys, which stay ambiguous on purpose."""
    entries = (diagnosis.get("signal_distribution") or {}).get("signals") or []
    candidates = [e for e in entries if e.get("signal") == signal_name]
    if not candidates:
        raise ContractError(
            f"diagnosis has no distribution for signal {signal_name!r}; "
            "check the signal name or re-run diagnose"
        )
    exact = [e for e in candidates if e.get("key") == f"decision:{signal_name}"]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    keys = ", ".join(str(e.get("key")) for e in candidates)
    raise ContractError(
        f"signal {signal_name!r} is ambiguous in the diagnosis ({keys}); "
        "calibration needs a single score-type distribution"
    )


def calibrated_quantiles(
    quantiles: list[float] | None, target_share: float | None
) -> list[tuple[float, str]]:
    """[(quantile, basis note)] from one of the two calibration intents."""
    if quantiles is not None:
        if not quantiles:
            raise ContractError("--quantiles must list at least one quantile")
        notes: list[tuple[float, str]] = []
        for q in quantiles:
            if not 0.0 < q <= 1.0:
                raise ContractError(f"quantile {q} is outside (0, 1]")
            notes.append((q, f"p{q * 100:g}"))
        return notes
    if target_share is not None:
        if not 0.0 < target_share < 1.0:
            raise ContractError(f"target share {target_share} is outside (0, 1)")
        q = round(1.0 - target_share, 6)
        return [(q, f"p{q * 100:g} (target-share {target_share:g})")]
    raise ContractError("one of quantiles or target_share is required")


def _anchor(name: str) -> str:
    # the diag anchor charset forbids colons (G2 regex): refuse instead of mangling
    if not name or any(ch not in _ANCHOR_SAFE for ch in name):
        raise ContractError(
            f"cannot build a diag anchor for name {name!r}: restricted charset"
        )
    return name


def _site_refs(
    candidate: rule_threshold.RuleThresholdCandidate, diagnosis_ref: str
) -> tuple[str, ...]:
    """One insight ref per moved site: diag:<path>@sha256:<hex>#threshold.<decision>.<signal>."""
    if not diagnosis_ref.startswith("diag:") or "@sha256:" not in diagnosis_ref:
        raise ContractError(
            f"diagnosis ref must look like diag:<path>@sha256:<hex>, got {diagnosis_ref!r}"
        )
    refs: list[str] = []
    for site in candidate.sites:
        anchor = f"threshold.{_anchor(site.decision)}.{_anchor(site.signal)}"
        ref = f"{diagnosis_ref}#{anchor}"
        if ref not in refs:
            refs.append(ref)
    return tuple(refs)


def _basis(candidate: rule_threshold.RuleThresholdCandidate, note: str, n: int) -> str:
    olds = "/".join(f"{v:g}" for v in candidate.old_values)
    return f"{candidate.signal_name} {candidate.op} {olds} -> {candidate.new_value:g} ({note} of n={n})"


def propose_calibrated(
    diagnosis: dict[str, Any],
    base_config: dict[str, Any],
    signal_name: str,
    *,
    diagnosis_ref: str,
    config_hash: str,
    quantiles: list[float] | None = None,
    target_share: float | None = None,
    from_value: float | None = None,
    op: str | None = None,
    lo: float | None = None,
    hi: float | None = None,
    ref_prefix: str = "calibrated",
) -> list[CalibratedProposal]:
    """Rule threshold candidates at observed quantiles of one signal.

    The quantiles are recomputed from the stored sample with the same
    percentile routine diagnose used, then rounded to 2 decimals and handed
    to rule_threshold.propose_rule_thresholds (same-group sites move
    together, range clamps and dedupe apply). Raises ContractError when the
    diagnosis does not match the base config hash.
    """
    verify_diagnosis_config(diagnosis, config_hash)
    entry = _signal_entry(diagnosis, signal_name)
    stored = entry.get("values")
    if not isinstance(stored, list) or not stored:
        raise ContractError(
            f"diagnosis signal entry for {signal_name!r} carries no stored values; "
            "regenerate the diagnosis with a compatible 'recipe-lab diagnose'"
        )
    sample = sorted(float(v) for v in stored)
    targets: list[tuple[float, str]] = []
    seen: set[float] = set()
    for q, note in calibrated_quantiles(quantiles, target_share):
        target = round(percentile(sample, q * 100.0), 2)
        if target not in seen:
            seen.add(target)
            targets.append((target, note))
    candidates = rule_threshold.propose_rule_thresholds(
        base_config,
        signal_name,
        values=[t for t, _ in targets],
        from_value=from_value,
        op=op,
        lo=lo,
        hi=hi,
        ref_prefix=ref_prefix,
    )
    notes = dict(targets)
    proposals = []
    for candidate in candidates:
        note = notes.get(candidate.new_value, "clamped")
        refs = _site_refs(candidate, diagnosis_ref)
        # verified now, so the ledger never inherits an unverifiable ref
        resolve_insight_refs(list(refs), ledger=None)
        proposals.append(
            CalibratedProposal(
                candidate=candidate,
                basis=_basis(candidate, note, len(sample)),
                insight_refs=refs,
            )
        )
    return proposals


def write_calibrated_candidates(
    proposals: list[CalibratedProposal], output_dir: str | Path
) -> list[Path]:
    """Write each candidate like the rule-level proposer does, then stamp the
    sidecar with the calibration basis and its diagnosis insight refs."""
    paths = rule_threshold.write_candidates(
        [p.candidate for p in proposals], output_dir
    )
    for proposal, path in zip(proposals, paths, strict=False):
        sidecar_path = path.with_suffix(".candidate.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["change_summary"] = proposal.basis
        sidecar["provenance"] = {
            "proposer": "distribution-calibration",
            "insight_refs": list(proposal.insight_refs),
        }
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return paths
