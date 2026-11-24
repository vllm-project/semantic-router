"""C9: acceptance bundle packaging (final live PASS + transfer guards).

`recipe-lab bundle` packages the acceptance evidence for exactly one ledger
record into a reviewable directory. It is deliberately hard to run: only a
live-tier PASS carrying the ledger's ``final`` promotion flag with every
declared guard passing is eligible - anything else is a contract error
(exit 2), never a warning.

Artifacts (a byte-stable pure function of the inputs; no wall-clock content):
config.yaml / validate.txt / config.diff / report.md / profile.json /
ledger.jsonl (campaign excerpt) / provenance.json / PROMOTION.md / REVIEW.md.

Honesty rules the bundle enforces:
- the packaged inputs are bound to the evidence (Y4): --config must hash to
  the record candidate's config_sha256, and --skeleton/--profile must hash
  to the scaffold/profile pair the candidate's bootstrap provenance sidecar
  cites;
- the packaged config must classify like the evaluated one (Y2): its
  classification projection hash (deployments, system, modules.classifier,
  modules.prompt_guard, embeddings) must match the projection the bootstrap
  sidecar recorded for the candidate;
- provenance cites hashes and basenames only - no absolute personal paths;
- every artifact is scanned for private content (personal user directories,
  non-local literal-IP URLs) plus any --private-pattern regexes before
  anything is written; a match fails the bundle;
- missing evidence (router commit, model revisions, model_eval script) is
  recorded as null, never guessed.
"""

from __future__ import annotations

import difflib
import hashlib
import ipaddress
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .classification import classification_projection_hash
from .contract import ContractError, ExperimentSpec
from .ledger import Ledger
from .promotion_evidence import verify_promotion_evidence
from .provenance import portable_path
from .report import render_promotion_draft

_LIMITATIONS_TEXT = (
    "Replayed from the recipe-lab ledger only. Metrics are descriptive; "
    "statistical significance is not established. A PASS is a recommendation, "
    "not a promotion: promotion remains a human decision (see PROMOTION.md "
    "and REVIEW.md)."
)

_CACHE_TOOLS_TEXT = (
    "The acceptance protocol evaluates the router with response caching and "
    "tool-call features disabled on the serving stack; the packaged metrics "
    "are only valid under that configuration."
)

_REVIEW_TEMPLATE = """# Recipe bundle review (V5)

- record: {record_id}
- candidate ref: {ref} ({kind})
- evidence draft: PROMOTION.md
- hashes: provenance.json

## Reviewer checklist (fill in during V5)

- [ ] PROMOTION.md human checklist complete
- [ ] provenance.json hashes verified against the archived inputs
- [ ] transfer guards and constraints reviewed in report.md
- [ ] decision (promote / hold / rework):
"""

# C9 private-content scan: personal user directories and non-local literal-IP
# URLs. Custom --private-pattern regexes extend these; a match fails the
# bundle before anything is written.
_BASE_PRIVATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("windows-user-path", re.compile(r'[A-Za-z]:\\Users\\[^\s/:*?"<>|\\]+')),
    ("unix-user-path", re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+")),
)
_IPV4_LITERAL = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?![\w.])")


def _content_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _run_config_validate(config_path: Path, validate_bin: str) -> str:
    """Run `<validate-bin> config validate` and return its output text.

    A failing validation is a bundle rejection, not a packaged artifact.
    """
    try:
        proc = subprocess.run(
            [validate_bin, "config", "validate", "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=300.0,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ContractError(f"validator binary not found: {validate_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ContractError("config validate timed out") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        raise ContractError(
            f"bundle rejected: config validate exit {proc.returncode}: {tail}"
        )
    # The real validator echoes an absolute input path in its Path column.
    # Rich can wrap the displayed filename over two lines. Restrict redaction
    # to the known argument or its displayed prefix; diagnostics stay intact.
    text = proc.stdout + proc.stderr
    path = str(config_path.absolute())
    text = text.replace(path, portable_path(path))

    def portable_input(match: re.Match) -> str:
        label, value = match.groups()
        if path.startswith(value.rstrip()):
            return label + portable_path(value)
        return match.group()

    return re.sub(r"(?m)^([ \t]*Path[ \t]*:?[ \t]+)([^\r\n]+)", portable_input, text)


def scan_private_content(
    artifacts: dict[str, str], extra_patterns: list[str] | None = None
) -> None:
    """Fail the bundle when any artifact carries private content.

    The error lists offending artifact:pattern pairs; the matched content is
    never echoed (it must not leak into logs either).
    """
    compiled = list(_BASE_PRIVATE_PATTERNS)
    for index, raw in enumerate(extra_patterns or []):
        try:
            compiled.append((f"custom[{index}]", re.compile(raw)))
        except re.error as exc:
            raise ContractError(f"invalid --private-pattern {raw!r}: {exc}") from exc
    hits: list[str] = []
    for name in sorted(artifacts):
        text = artifacts[name]
        for pattern_name, pattern in compiled:
            if pattern.search(text):
                hits.append(f"{name}:{pattern_name}")
        for match in _IPV4_LITERAL.finditer(text):
            try:
                address = ipaddress.IPv4Address(match.group())
            except ipaddress.AddressValueError:
                continue
            if not (address.is_loopback or address.is_unspecified):
                hits.append(f"{name}:non-local-ip")
                break
    if hits:
        raise ContractError(
            "bundle artifacts contain private content "
            "(artifact:pattern; content not echoed): " + ", ".join(sorted(set(hits)))
        )


def _metrics_json(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "n/a"
    return json.dumps(metrics, sort_keys=True, ensure_ascii=False)


def _routing_distribution(notes: str) -> dict[str, Any] | None:
    marker = "routing_distribution="
    if not notes or marker not in notes:
        return None
    tail = notes.rsplit(marker, 1)[1]
    try:
        parsed = json.loads(tail)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _repetitions_from_notes(notes: str) -> int | None:
    match = re.search(r"repetitions=(\d+)", notes or "")
    return int(match.group(1)) if match else None


def render_bundle_report(
    spec: ExperimentSpec,
    records: list[dict[str, Any]],
    record: dict[str, Any],
    lanes: dict[str, Any] | None = None,
) -> str:
    """Deterministic acceptance report rendered from the ledger + spec only."""
    candidate = record.get("candidate") or {}
    paired = record.get("paired") or {}
    gates = record.get("gates") or {}
    lines = [
        "# recipe-lab acceptance bundle report",
        "",
        f"- record: `{record.get('record_id', 'unrecorded')}` (final verify, tier live)",
        f"- verdict: **{record.get('verdict')}**",
        f"- workload: {record.get('workload') or '-'} / split: {record.get('split') or '-'}",
        f"- candidate ref: `{candidate.get('ref')}` ({candidate.get('kind')})",
        f"- dataset: {record.get('dataset_hash')}",
        f"- baseline ref: {record.get('baseline_ref') or '-'}",
        "",
        "## Verdict detail",
        "",
        f"- metrics: {_metrics_json(record.get('metrics'))}",
        f"- gates: {', '.join(f'{k}={v}' for k, v in gates.items()) or 'none declared'}",
        f"- bench evidence: {', '.join(record.get('bench_refs') or []) or '-'}",
    ]
    if paired:
        lines += [
            f"- paired cases: {paired.get('paired_cases')}",
            f"- quality delta: {paired.get('quality_delta')}",
            f"- quality delta CI95: {paired.get('quality_delta_ci95')}",
        ]
        if paired.get("quality_delta_bootstrap_ci95") is not None:
            lines.append(
                f"- quality delta bootstrap CI95: {paired.get('quality_delta_bootstrap_ci95')}"
            )
    constraints = record.get("constraints") or {}
    if constraints:
        lines += ["", "### Constraints", ""]
        for name, evidence in constraints.items():
            lines.append(
                f"- {name}: {evidence.get('status', '?')} "
                f"(candidate={evidence.get('candidate')}, bound={evidence.get('bound_value')})"
            )
    reasons = record.get("reasons") or []
    if reasons:
        lines += ["", "### Verdict reasons", ""]
        lines += [f"- {reason}" for reason in reasons]
    if lanes is not None:
        # Template path (sub-issue-8 §2.2 step 10): the report must say which
        # lanes the case set actually measured, which were assigned by the
        # default rule (untested), and which were explicitly excluded.
        measured = lanes.get("measured_lanes") or []
        unmeasured = lanes.get("unmeasured_lanes") or []
        excluded = lanes.get("excluded_lanes") or []
        lines += ["", "### Lane coverage", ""]
        lines.append(f"- measured lanes: {', '.join(measured) if measured else 'none'}")
        lines.append(
            "- unmeasured lanes (assigned by the default rule, untested): "
            + (", ".join(unmeasured) if unmeasured else "none")
        )
        lines.append(
            "- excluded lanes (--exclude-decision): "
            + (", ".join(excluded) if excluded else "none")
        )
    distribution = _routing_distribution(record.get("notes") or "")
    if distribution:
        lines += ["", "### Observed routing distribution (final verify)", ""]
        lines += [
            f"- {model}: {count}" for model, count in sorted(distribution.items())
        ]
    lines += [
        "",
        "## Campaign evidence (ledger replay)",
        "",
        "| record | round | tier | workload | verdict | metrics | gates |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in records:
        row_gates = row.get("gates") or {}
        gates_text = ", ".join(f"{k}={v}" for k, v in row_gates.items()) or "-"
        lines.append(
            f"| {row.get('record_id', '-')} | {row.get('round')} | {row.get('tier')} "
            f"| {row.get('workload') or '-'} | {row.get('verdict')} "
            f"| {_metrics_json(row.get('metrics'))} | {gates_text} |"
        )
    counts: dict[str, dict[str, int]] = {}
    for row in records:
        name = row.get("workload") or "default"
        verdict = str(row.get("verdict"))
        counts.setdefault(name, {})
        counts[name][verdict] = counts[name].get(verdict, 0) + 1
    lines += ["", "### Per-workload verdict counts", ""]
    for name in sorted(counts):
        detail = ", ".join(
            f"{count}x {verdict}" for verdict, count in sorted(counts[name].items())
        )
        lines.append(f"- {name}: {detail}")
    lines += ["", "## Evidence quality", ""]
    repetitions = _repetitions_from_notes(record.get("notes") or "")
    lines.append(
        f"- verify repetitions: {repetitions if repetitions is not None else 'unknown'}"
    )
    if paired:
        paired_cases = paired.get("paired_cases")
        min_cases = spec.objective.min_cases
        verdict_text = (
            "meets"
            if isinstance(paired_cases, int) and paired_cases >= min_cases
            else "below or unknown"
        )
        lines.append(
            f"- paired cases {paired_cases} vs objective.min_cases {min_cases}: {verdict_text}"
        )
    warnings = [
        f"{row.get('record_id')}: coverage_warning"
        for row in records
        if row.get("coverage_warning")
    ]
    warnings += [
        f"{row.get('record_id')}: constraint_warning"
        for row in records
        if row.get("constraint_warning")
    ]
    lines.append("- warnings: " + ("; ".join(warnings) if warnings else "none"))
    lines += ["", "## Assumptions", ""]
    if spec.cost_basis is not None:
        text = f"- cost basis: {spec.cost_basis.source}"
        if spec.cost_basis.assumed_pricing:
            text += " (assumed pricing, not metered bills)"
        lines.append(text)
    if spec.references is not None:
        lines.append(
            f"- reference arms: strong={spec.references.strong} weak={spec.references.weak}"
        )
    lines.append(
        "- guard floors come from the spec's gates section; the ledger record "
        "carries the measured outcomes"
    )
    lines += [
        "",
        "## Limitations",
        "",
        _LIMITATIONS_TEXT,
        "",
        "## Cache and tools",
        "",
        _CACHE_TOOLS_TEXT,
        "",
    ]
    return "\n".join(lines)


def build_bundle(
    *,
    spec: ExperimentSpec,
    ledger_path: str,
    record_id: str,
    out_dir: str | Path,
    config_path: str | Path,
    skeleton_path: str | Path | None = None,
    profile_path: str | Path,
    validate_bin: str,
    router_commit: str | None = None,
    model_revisions: dict[str, str] | None = None,
    model_eval_script: str | None = None,
    candidate_sidecar: str | Path,
    bindings_path: str | Path | None = None,
    extra_private_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the C9 bundle for one final live PASS record.

    Two candidate modes are supported, selected by the sidecar's schema:

    - **legacy bootstrap sidecar** (recipe-lab-bootstrap-v1): ``--skeleton``
      and ``--profile`` are bound to the sidecar's scaffold/profile hashes
      and the packaged config must classify like the evaluated candidate
      (Y2 projection hash).
    - **template bindings sidecar** (recipe-lab-bindings-v1, sub-issue-8
      §2.2 step 10): ``--bindings`` replaces ``--skeleton``; it is bound to
      the sidecar's bindings hash, the profile to the sidecar's profile
      hash, and the config to both the ledger candidate and the sidecar's
      candidate hash. The provenance carries the template bundle hash and
      the per-lane measured/unmeasured/excluded lists; the report states
      the lane coverage.

    Every rejection is a ContractError (exit 2 through the CLI): unknown
    record, non-live tier, non-PASS verdict, no final flag, a transfer
    guard not passing, a packaged input that is not the artifact the
    evidence cites (Y4), a failing config validate, or private content in
    any artifact. Nothing is written unless every check passes.
    """
    records = Ledger(ledger_path).read()
    matches = [row for row in records if row.get("record_id") == record_id]
    if not matches:
        raise ContractError(f"record {record_id} not found in ledger {ledger_path}")
    record = matches[0]
    if record.get("tier") != "live":
        raise ContractError(f"record {record_id} is not a live-tier record")
    if record.get("verdict") != "PASS":
        raise ContractError(
            f"record {record_id} verdict is {record.get('verdict')!r}; "
            "bundle packages only PASS records"
        )
    if not record.get("final"):
        # Y4: finality is the ledger's own promotion flag, not something the
        # bundle reverse-engineers from the contract's split roles.
        raise ContractError(
            f"record {record_id} carries no final flag; bundle packages only "
            "live --final promotion records (the ledger writes final: true)"
        )
    not_passing = sorted(
        f"{name}={value}"
        for name, value in (record.get("gates") or {}).items()
        if value != "pass"
    )
    if not_passing:
        raise ContractError(
            f"record {record_id} has transfer guards not passing: "
            + ", ".join(not_passing)
        )
    verify_promotion_evidence(spec, record)

    config_file = Path(config_path)
    profile_file = Path(profile_path)
    # Y4: every packaged input must be the exact artifact the evidence cites.
    # The config is bound to the ledger's candidate (the sha256 the verdict
    # recorded); the skeleton and profile are bound to the bootstrap sidecar
    # the candidate was generated with.
    recorded_config_hash = (record.get("candidate") or {}).get("config_sha256")
    if not isinstance(recorded_config_hash, str) or not recorded_config_hash:
        raise ContractError(
            f"record {record_id}'s candidate carries no config_sha256; bundle "
            "packages only verdicts that named the config they graded (Y4)"
        )
    packaged_config_hash = _content_hash(config_file)
    if packaged_config_hash != recorded_config_hash:
        raise ContractError(
            f"--config hashes to {packaged_config_hash}, but the record's "
            f"candidate names {recorded_config_hash}"
        )
    sidecar_file = Path(candidate_sidecar)
    try:
        sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"cannot read the candidate provenance sidecar {sidecar_file}: {exc}"
        ) from exc
    template_mode = sidecar.get("schema") == "recipe-lab-bindings-v1"
    skeleton_file = Path(skeleton_path) if skeleton_path is not None else None
    bindings_file = Path(bindings_path) if bindings_path is not None else None
    if template_mode:
        if bindings_file is None or skeleton_file is not None:
            raise ContractError(
                "the candidate sidecar is a template-bindings sidecar "
                "(recipe-lab-bindings-v1): pass --bindings (not --skeleton); "
                "the template path has no result_to_config scaffold"
            )
    else:
        if skeleton_file is None or bindings_file is not None:
            raise ContractError(
                "the candidate sidecar is a legacy bootstrap sidecar: pass "
                "--skeleton (not --bindings); the template path cites a "
                "recipe-lab-bindings-v1 sidecar instead"
            )
        # Y4 (legacy): the scaffold and profile are bound to the bootstrap
        # sidecar the candidate was generated with.
        for flag, role, path in (
            ("--skeleton", "scaffold", skeleton_file),
            ("--profile", "profile", profile_file),
        ):
            cited = (sidecar.get(role) or {}).get("sha256")
            if not isinstance(cited, str) or not cited:
                raise ContractError(
                    f"the candidate sidecar cites no {role} sha256; bundle cannot "
                    f"bind {flag} to the evidence"
                )
            packaged = _content_hash(path)
            if packaged != cited:
                raise ContractError(
                    f"{flag} hashes to {packaged}, but the candidate sidecar cites {cited}"
                )
    if template_mode:
        # Y4 (template): the bindings and the profile are bound to the
        # bindings sidecar; the config additionally to the sidecar's own
        # candidate hash (the ledger binding above already pinned it).
        for flag, role, path in (
            ("--bindings", "bindings", bindings_file),
            ("--profile", "profile", profile_file),
        ):
            cited = (sidecar.get(role) or {}).get("sha256")
            if not isinstance(cited, str) or not cited:
                raise ContractError(
                    f"the bindings sidecar cites no {role} sha256; bundle cannot "
                    f"bind {flag} to the evidence"
                )
            packaged = _content_hash(path)
            if packaged != cited:
                raise ContractError(
                    f"{flag} hashes to {packaged}, but the bindings sidecar cites {cited}"
                )
        sidecar_candidate = (sidecar.get("candidate") or {}).get("sha256")
        if packaged_config_hash != sidecar_candidate:
            raise ContractError(
                f"--config hashes to {packaged_config_hash}, but the "
                f"bindings sidecar's candidate names {sidecar_candidate}"
            )
    if not template_mode:
        # Y2 (legacy only): the packaged config must classify exactly like
        # the config the evidence was produced with - the bootstrap sidecar
        # cites the candidate's classification projection hash. The template
        # path has no shared classification source; `recipe builtin init`
        # validated the composed config, and the bindings hash pins its
        # routing structure instead.
        sidecar_classification = sidecar.get("classification") or {}
        sidecar_projection = sidecar_classification.get("projection_sha256")
        if not isinstance(sidecar_projection, str) or not sidecar_projection:
            raise ContractError(
                "the candidate sidecar cites no classification projection_sha256; "
                "bundle cannot prove the packaged config classifies like the "
                "evaluated candidate (Y2)"
            )
        try:
            packaged_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ContractError(f"--config is not valid YAML: {exc}") from exc
        packaged_projection = classification_projection_hash(packaged_config or {})
        if packaged_projection != sidecar_projection:
            raise ContractError(
                f"--config's classification projection hashes to {packaged_projection}, "
                f"but the candidate sidecar cites {sidecar_projection}"
            )
    validate_text = _run_config_validate(config_file, validate_bin)

    config_bytes = config_file.read_bytes()
    profile_bytes = profile_file.read_bytes()
    if template_mode:
        diff_text = None
    else:
        skeleton_text = skeleton_file.read_text(encoding="utf-8")
        diff_text = "".join(
            difflib.unified_diff(
                skeleton_text.splitlines(keepends=True),
                config_bytes.decode("utf-8").splitlines(keepends=True),
                fromfile="skeleton",
                tofile="config.yaml",
            )
        )
    campaign_hash = record.get("dataset_hash")
    excerpt = [row for row in records if row.get("dataset_hash") == campaign_hash]
    ledger_text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in excerpt
    )
    revisions = dict(model_revisions or {})
    models_revision: dict[str, str | None] = {
        model.name: revisions.pop(model.name, None) for model in spec.models
    }
    models_revision.update(revisions)
    provenance = {
        "schema_version": "v1",
        "kind": "recipe-lab-bundle",
        "id": f"recipe-lab-bundle:{record_id}",
        "record_id": record_id,
        "router_commit": router_commit,
        "models": models_revision,
        "datasets": {
            "record_dataset_hash": campaign_hash,
            "workloads": {
                workload.name: {
                    split: split_spec.content_hash
                    for split, split_spec in workload.splits.items()
                }
                for workload in spec.workloads
            },
        },
        "artifacts": {
            "skeleton": (
                {
                    "name": skeleton_file.name,
                    "sha256": _content_hash(skeleton_file),
                }
                if skeleton_file is not None
                else None
            ),
            "bindings": (
                {
                    "name": bindings_file.name,
                    "sha256": _content_hash(bindings_file),
                }
                if bindings_file is not None
                else None
            ),
            "config": {"name": "config.yaml", "sha256": _content_hash(config_file)},
            "profile": {
                "name": "profile.json",
                "sha256": _content_hash(profile_file),
            },
            "ledger": {
                "name": Path(ledger_path).name,
                "sha256": _content_hash(Path(ledger_path)),
            },
        },
        # Y2: hashes only - the projection this bundle verified the packaged
        # config against, plus the classification source's content identity.
        # Template mode has no shared classification source (the bindings
        # hash pins the routing structure); classification is null there.
        "classification": {
            "projection_sha256": sidecar_projection if not template_mode else None,
            "source_sha256": (
                sidecar_classification.get("sha256") if not template_mode else None
            ),
        },
        # Template mode (sub-issue-8 §2.2 step 10): the template identity and
        # the per-lane measured/unmeasured/excluded lists the report cites.
        "template": sidecar.get("template") if template_mode else None,
        "lanes": (
            {
                "measured_lanes": sidecar.get("measured_lanes") or [],
                "unmeasured_lanes": sidecar.get("unmeasured_lanes") or [],
                "excluded_lanes": sidecar.get("excluded_lanes") or [],
            }
            if template_mode
            else None
        ),
        "run_ids": {
            "baseline_ref": record.get("baseline_ref"),
            "bench_refs": list(record.get("bench_refs") or []),
        },
        "model_eval_script": (
            {
                "name": Path(model_eval_script).name,
                "sha256": _content_hash(Path(model_eval_script)),
            }
            if model_eval_script
            else None
        ),
    }

    lanes_for_report = provenance["lanes"] if template_mode else None
    artifacts: dict[str, str] = {
        "validate.txt": validate_text,
        "report.md": render_bundle_report(spec, records, record, lanes_for_report),
        "ledger.jsonl": ledger_text,
        "provenance.json": json.dumps(
            provenance, sort_keys=True, ensure_ascii=False, indent=1
        )
        + "\n",
        "PROMOTION.md": render_promotion_draft(record),
        "REVIEW.md": _REVIEW_TEMPLATE.format(
            record_id=record_id,
            ref=(record.get("candidate") or {}).get("ref"),
            kind=(record.get("candidate") or {}).get("kind"),
        ),
        # byte copies are scanned through their decoded text as well
        "config.yaml": "",
        "profile.json": "",
    }
    if template_mode:
        artifacts["bindings.yaml"] = bindings_file.read_text(encoding="utf-8")
    else:
        artifacts["config.diff"] = diff_text
    artifacts["config.yaml"] = config_bytes.decode("utf-8")
    artifacts["profile.json"] = profile_bytes.decode("utf-8")
    scan_private_content(artifacts, extra_private_patterns)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_bytes(config_bytes)
    (out / "profile.json").write_bytes(profile_bytes)
    if template_mode:
        (out / "bindings.yaml").write_bytes(bindings_file.read_bytes())
    written = [
        "validate.txt",
        "report.md",
        "ledger.jsonl",
        "provenance.json",
        "PROMOTION.md",
        "REVIEW.md",
    ]
    if not template_mode:
        written.append("config.diff")
    for name in written:
        (out / name).write_text(artifacts[name], encoding="utf-8")
    return {
        "bundle": str(out),
        "record": record_id,
        "artifacts": sorted(artifacts),
    }
