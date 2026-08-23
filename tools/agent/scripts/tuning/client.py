"""Read-only client for evaluating an immutable semantic-router runtime."""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib import request


class RouterClient:
    """Stateless client for the router's evaluation API."""

    def __init__(self, endpoint: str = "http://localhost:8080"):
        self.endpoint = endpoint.rstrip("/")

    def eval_probe(self, query: str, trace: bool = True) -> dict:
        """Send a single query to /api/v1/eval and return the full response."""
        url = f"{self.endpoint}/api/v1/eval"
        if trace:
            url += "?trace=true"
        body = json.dumps({"text": query}).encode()
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def run_probes(
        self,
        probes: list[dict],
        result_adapter: Callable[[dict, dict], dict] | None = None,
    ) -> list[dict]:
        """Run all probes through the router and return result dicts.

        Args:
            probes: list of {"id", "query", "expected_decision", ...} dicts.
            result_adapter: optional (probe, response) -> dict to customize
                the result shape.  Return None to use the default adapter.
        """
        results: list[dict] = []
        for probe in probes:
            query = probe.get("query", "")
            expected = probe.get("expected_decision", "")
            try:
                resp = self.eval_probe(query, trace=True)
            except Exception as exc:
                results.append(
                    {
                        "id": probe.get("id", ""),
                        "query": query[:200],
                        "expected": expected,
                        "actual": "ERROR",
                        "correct": False,
                        "error": str(exc),
                        "tags": probe.get("tags", []),
                    }
                )
                continue

            if result_adapter:
                custom = result_adapter(probe, resp)
                if custom is not None:
                    results.append(custom)
                    continue

            dr = resp.get("decision_result", {})
            actual = dr.get("decision_name", "NONE")
            results.append(
                {
                    "id": probe.get("id", ""),
                    "query": query[:200],
                    "expected": expected,
                    "actual": actual,
                    "correct": actual == expected,
                    "signal_confidences": resp.get("signal_confidences", {}),
                    "projection_scores": resp.get("projection_scores", {}),
                    "projection_bands": resp.get("projection_bands", {}),
                    "eval_trace": dr.get("eval_trace", []),
                    "matched_signals": dr.get("matched_signals", {}),
                    "unmatched_signals": dr.get("unmatched_signals", {}),
                    "tags": probe.get("tags", []),
                }
            )
        return results
