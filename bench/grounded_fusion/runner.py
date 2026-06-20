"""Drives the semantic router's fusion looper and parses the ``fusion`` trace.

Sends one DRACO problem to the router's OpenAI endpoint and extracts the final
synthesized answer plus the per-panel-response grounding scores / drop decisions
(``body["fusion"]`` -- see fusion.go formatFusionJSONResponse and the
FusionGroundingTrace it carries).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests


@dataclass
class PanelEntry:
    model: str
    content: str
    grounding_score: float | None = None
    dropped: bool = False
    flagged: list[str] = field(default_factory=list)


@dataclass
class FusionResult:
    sample_id: str
    final_answer: str
    panel: list[PanelEntry] = field(default_factory=list)
    grounding_present: bool = False
    reference_mode: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    models_used: list[str] = field(default_factory=list)
    error: str | None = None


def run_fusion(
    endpoint: str,
    sample_id: str,
    problem: str,
    model: str = "auto",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int = 1800,
    sentinel: str = "deliberate-eval",
) -> FusionResult:
    """POST one problem to the router fusion endpoint and parse the trace.

    The router routes to the bench fusion decision via a regex keyword rule, so the
    ``sentinel`` is prepended to the prompt (see make_configs.py). Use model="auto"
    so the request goes through classification into the fusion decision.
    """
    content = f"{sentinel}\n\n{problem}" if sentinel else problem
    start = time.time()
    try:
        resp = requests.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers={"Authorization": "Bearer test"},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        return FusionResult(sample_id=sample_id, final_answer="", error=str(e))

    latency_ms = (time.time() - start) * 1000
    if resp.status_code != requests.codes.ok:
        return FusionResult(
            sample_id=sample_id,
            final_answer="",
            latency_ms=latency_ms,
            error=f"HTTP {resp.status_code}: {resp.text[:300]}",
        )
    return _parse(sample_id, resp.json(), latency_ms)


def _parse(sample_id: str, body: dict, latency_ms: float) -> FusionResult:
    result = FusionResult(sample_id=sample_id, final_answer="", latency_ms=latency_ms)
    try:
        result.final_answer = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        result.error = "no choices in response"
    result.usage = body.get("usage", {}) or {}

    fusion = body.get("fusion")
    if not isinstance(fusion, dict):
        return result  # plain (non-fusion) response or fusion trace suppressed

    responses = fusion.get("responses") or []
    grounding = (
        fusion.get("grounding") if isinstance(fusion.get("grounding"), dict) else None
    )
    score_by_model: dict[str, dict] = {}
    if grounding:
        result.grounding_present = True
        result.reference_mode = grounding.get("reference_mode", "")
        for s in grounding.get("scores") or []:
            score_by_model[s.get("model", "")] = s

    content_by_model = {
        r.get("model", ""): r.get("content", "") or "" for r in responses
    }
    if grounding and score_by_model:
        # grounding.scores lists EVERY panel response with its drop decision, while
        # the trace's `responses` holds only the KEPT (post-filter) responses. Build
        # the panel from the scores so dropped responses (and thus the contested
        # slice) stay visible. Content exists only for kept responses — the judge
        # never saw the dropped ones — which is fine: we only need their drop flag.
        for model, s in score_by_model.items():
            result.panel.append(
                PanelEntry(
                    model=model,
                    content=content_by_model.get(model, ""),
                    grounding_score=s.get("score"),
                    dropped=bool(s.get("dropped", False)),
                    flagged=s.get("flagged_spans") or [],
                )
            )
    else:
        for r in responses:
            result.panel.append(
                PanelEntry(model=r.get("model", ""), content=r.get("content", "") or "")
            )
    return result
