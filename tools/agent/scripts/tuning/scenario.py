"""Scenario ABC and immutable-runtime candidate tuning pipeline.

A Scenario defines *what* to tune and *how to score* it.
CandidateTuner evaluates one live snapshot, proposes one offline candidate, and
validates that candidate without mutating the running router.
"""

from __future__ import annotations

import abc
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml
from router_calibration_support import run_validate

from . import engine, engine_selection
from .client import RouterClient
from .probes import load_probes


class Scenario(abc.ABC):
    """Pluggable evaluation scenario that defines tuning behavior.

    Subclasses must implement:
      - name: identifier for this scenario
      - severity(probe) -> int: severity weighting for the loss function
    And may override:
      - adapt_result(): customize how router responses are parsed
      - display_iteration(): print per-iteration diagnostics
      - build_output(): enrich the final JSON output
    """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def severity(self, probe: dict) -> int: ...

    def adapt_result(self, probe: dict, resp: dict) -> dict | None:
        """Return a custom result dict, or None to use the default adapter."""
        return None

    def display_iteration(
        self,
        iteration: int,
        results: list[dict],
        diagnoses: list[dict],
        fix: Any,
    ) -> None:
        """Hook for scenario-specific diagnostic display.

        Override in subclasses to print custom diagnostics for the candidate pass.
        The default implementation is a no-op.
        """
        return

    def build_output(self, base: dict) -> dict:
        """Hook to enrich the output JSON with scenario-specific data."""
        return base


class CandidateTuner:
    """Propose and validate one candidate for an immutable live runtime."""

    def __init__(
        self,
        scenario: Scenario,
        router: RouterClient,
        config_path: Path,
        probes_path: Path,
        candidate_path: Path,
    ):
        self.cs = scenario
        self.router = router
        self.config_path = config_path.resolve()
        self.probes_path = probes_path.resolve()
        self.candidate_path = candidate_path.resolve()
        if self.config_path == self.candidate_path:
            raise ValueError("candidate config must not overwrite the active manifest")

    def run(self) -> dict:
        """Evaluate once and write at most one offline-validated candidate."""
        probes = load_probes(self.probes_path)
        print(f"\nLoaded {len(probes)} probes from {self.probes_path}")

        adapter = self._make_adapter()
        print(f"\n{'='*60}")
        print("  Immutable runtime evaluation")
        print(f"{'='*60}")
        results = self.router.run_probes(probes, adapter)
        self._print_summary(results)

        cfg_raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(cfg_raw, dict):
            raise ValueError("config must decode to a YAML mapping")
        dsl = engine.load_dsl_config(cfg_raw)
        failures = [result for result in results if not result["correct"]]
        diagnoses = [
            engine_selection.diagnose_probe(result, dsl) for result in failures
        ]
        fix = engine_selection.select_fix(
            diagnoses,
            results,
            dsl,
            severity_fn=self.cs.severity,
        )
        self.cs.display_iteration(0, results, diagnoses, fix)

        validation = None
        if fix is None:
            print("  No beneficial candidate found.")
        else:
            self._apply_fix(fix, cfg_raw, dsl)
            self._write_candidate(cfg_raw)
            validation = run_validate(None, self.candidate_path)
            status = "valid" if validation.get("valid") else "invalid"
            print(f"  Offline validation: {status}")

        output = {
            "scenario": self.cs.name,
            "method": "analytical_trace_diagnosis",
            "pipeline": "observe → diagnose → candidate → offline validate",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "router_endpoint": self.router.endpoint,
            "config_path": str(self.config_path),
            "num_probes": len(probes),
            "evaluation": self._evaluation_record(results),
            "selected_fix": self._fix_to_dict(fix) if fix else None,
            "candidate_config": str(self.candidate_path) if fix else None,
            "validation": validation,
        }
        return self.cs.build_output(output)

    # -- internal helpers --------------------------------------------------

    def _make_adapter(self):
        cs = self.cs

        def adapter(probe, resp):
            return cs.adapt_result(probe, resp)

        return adapter

    def _print_summary(self, results):
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        pct = round(100 * correct / total, 1) if total else 0
        weighted_loss = sum(self.cs.severity(r) for r in results if not r["correct"])
        print(
            f"  Accuracy: {correct}/{total} ({pct}%)" f"  severity_loss={weighted_loss}"
        )

    def _evaluation_record(self, results):
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        return {
            "accuracy": correct,
            "total": total,
            "pct": round(100 * correct / total, 1) if total else 0,
            "severity_weighted_loss": sum(
                self.cs.severity(r) for r in results if not r["correct"]
            ),
            "probe_details": [
                {
                    "id": r["id"],
                    "expected": r["expected"],
                    "actual": r["actual"],
                    "correct": r["correct"],
                }
                for r in results
            ],
        }

    def _apply_fix(self, fix, cfg_raw, dsl):
        if isinstance(fix, engine.StructuralFix):
            updated = engine_selection.apply_structural_fix(cfg_raw, fix)
            cfg_raw.update(updated)
            print(f"  Applied structural fix: {fix.description}")
        elif isinstance(fix, engine.Fix):
            updated = engine_selection.apply_fix_to_config(cfg_raw, fix, dsl)
            cfg_raw.update(updated)
            print(f"  Applied parametric fix: {fix.explanation}")

    def _write_candidate(self, cfg_raw):
        target_path = self.candidate_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(target_path.parent),
            suffix=".tmp",
            prefix=target_path.stem,
        )
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(
                    cfg_raw,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target_path)
        except BaseException:
            os.unlink(tmp)
            raise
        print(f"  Candidate written to {target_path}")

    @staticmethod
    def _fix_to_dict(fix) -> dict:
        if isinstance(fix, engine.StructuralFix):
            return {
                "fix_type": "structural_rule_change",
                "decision": fix.decision_name,
                "action": fix.action,
                "description": fix.description,
                "remove_signals": fix.remove_signals,
            }
        elif isinstance(fix, engine.Fix):
            return {
                "fix_type": fix.fix_type,
                "target": fix.target,
                "param_path": fix.param_path,
                "old_value": fix.old_value,
                "new_value": fix.new_value,
                "net_improvement": fix.net_improvement,
                "regressions": fix.regressions,
                "explanation": fix.explanation,
            }
        return {}
