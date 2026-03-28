"""Scenario ABC and TuningLoop — pluggable analytical tuning pipeline.

A Scenario defines *what* to tune and *how to score* it.
The TuningLoop orchestrates the generic observe → diagnose → fix → verify cycle.
"""

from __future__ import annotations

import abc
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from . import engine
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
        """Hook for scenario-specific per-iteration display.

        Override in subclasses to print custom diagnostics per tuning round.
        The default implementation is a no-op.
        """
        return

    def build_output(self, base: dict) -> dict:
        """Hook to enrich the output JSON with scenario-specific data."""
        return base


class TuningLoop:
    """Generic observe → diagnose → analytical-fix → verify loop.

    Works with any Scenario implementation.  The loop:
      1. Runs probes against the live router
      2. Diagnoses failures using the analytical engine
      3. Selects the best analytical fix (structural or parametric)
      4. Applies the fix and reloads the router
      5. Repeats until convergence or max iterations
    """

    def __init__(
        self,
        scenario: Scenario,
        router: RouterClient,
        config_path: Path,
        probes_path: Path,
        deploy_config: Path | None = None,
        router_pid: int = 0,
        max_iterations: int = 10,
    ):
        self.cs = scenario
        self.router = router
        self.config_path = config_path
        self.probes_path = probes_path
        self.deploy_config = deploy_config
        self.router_pid = router_pid
        self.max_iterations = max_iterations

    def run(self) -> dict:
        """Execute the full tuning loop. Returns the output dict."""
        probes = load_probes(self.probes_path)
        print(f"\nLoaded {len(probes)} probes from {self.probes_path}")

        adapter = self._make_adapter()
        iterations: list[dict] = []
        all_fixes: list[dict] = []
        trajectory: list[dict] = []

        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"  Iteration {iteration}")
            print(f"{'='*60}")

            results = self.router.run_probes(probes, adapter)
            self._print_summary(iteration, results)

            correct = sum(1 for r in results if r["correct"])
            total = len(results)
            pct = round(100 * correct / total, 1) if total else 0

            weighted_loss = sum(
                self.cs.severity(r) for r in results if not r["correct"]
            )

            if correct == total:
                print("  All probes pass — converged!")
                iterations.append(self._iteration_record(iteration, results, [], None))
                trajectory.append(
                    {
                        "iteration": iteration,
                        "accuracy": correct,
                        "total": total,
                        "pct": pct,
                        "severity_weighted_loss": weighted_loss,
                    }
                )
                break

            cfg_raw = yaml.safe_load(self.config_path.read_text())
            dsl = engine.load_dsl_config(cfg_raw)

            failures = [r for r in results if not r["correct"]]
            diagnoses = [engine.diagnose_probe(r, dsl) for r in failures]

            fix = engine.select_fix(
                diagnoses,
                results,
                dsl,
                severity_fn=self.cs.severity,
            )

            self.cs.display_iteration(iteration, results, diagnoses, fix)

            iterations.append(
                self._iteration_record(iteration, results, diagnoses, fix)
            )
            trajectory.append(
                {
                    "iteration": iteration,
                    "accuracy": correct,
                    "total": total,
                    "pct": pct,
                    "severity_weighted_loss": weighted_loss,
                }
            )

            if fix is None:
                print("  No beneficial fix found — converged.")
                break

            self._apply_fix(fix, cfg_raw, dsl)
            all_fixes.append(self._fix_to_dict(fix))
            self._write_and_reload(cfg_raw)

        final_verification = self._run_final_verification(probes, adapter)

        output = {
            "scenario": self.cs.name,
            "method": "analytical_trace_diagnosis",
            "pipeline": "engine.py — observe → diagnose → fix → verify",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "router_endpoint": self.router.endpoint,
            "config_path": str(self.config_path),
            "num_probes": len(probes),
            "iterations": iterations,
            "final_verification": final_verification,
            "all_fixes_applied": all_fixes,
            "trajectory": trajectory,
        }
        return self.cs.build_output(output)

    # -- internal helpers --------------------------------------------------

    def _run_final_verification(self, probes, adapter) -> dict:
        print(f"\n{'='*60}")
        print("  Final verification")
        print(f"{'='*60}")
        results = self.router.run_probes(probes, adapter)
        self._print_summary("final", results)
        correct = sum(1 for r in results if r["correct"])
        weighted_loss = sum(self.cs.severity(r) for r in results if not r["correct"])
        return {
            "accuracy": correct,
            "total": len(results),
            "pct": (round(100 * correct / len(results), 1) if results else 0),
            "severity_weighted_loss": weighted_loss,
        }

    def _make_adapter(self):
        cs = self.cs

        def adapter(probe, resp):
            return cs.adapt_result(probe, resp)

        return adapter

    def _print_summary(self, label, results):
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        pct = round(100 * correct / total, 1) if total else 0
        weighted_loss = sum(self.cs.severity(r) for r in results if not r["correct"])
        print(
            f"  Accuracy: {correct}/{total} ({pct}%)" f"  severity_loss={weighted_loss}"
        )

    def _iteration_record(self, iteration, results, diagnoses, fix):
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        return {
            "iteration": iteration,
            "accuracy": correct,
            "total": total,
            "pct": round(100 * correct / total, 1) if total else 0,
            "severity_weighted_loss": sum(
                self.cs.severity(r) for r in results if not r["correct"]
            ),
            "num_diagnoses": len(diagnoses),
            "selected_fix": self._fix_to_dict(fix) if fix else None,
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
            updated = engine.apply_structural_fix(cfg_raw, fix)
            cfg_raw.update(updated)
            print(f"  Applied structural fix: {fix.description}")
        elif isinstance(fix, engine.Fix):
            updated = engine.apply_fix_to_config(cfg_raw, fix, dsl)
            cfg_raw.update(updated)
            print(f"  Applied parametric fix: {fix.explanation}")

    def _write_and_reload(self, cfg_raw):
        target = self.deploy_config or self.config_path
        target_path = Path(target)
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
            os.replace(tmp, str(target_path))
        except BaseException:
            os.unlink(tmp)
            raise
        print(f"  Config written to {target}")
        if self.router_pid:
            ok = self.router.hot_reload(self.router_pid)
            print(f"  Hot reload: {'success' if ok else 'timeout/failed'}")
        else:
            print("  No router PID — skipping hot reload")

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
