"""Native-inference adapters for open board entrants (calibration anchors) and our own checkpoints.

Every adapter returns a kit ``/v1/systemone`` response ``{"answers": {key: answer}}`` from the entrant's
own prompt, readout and limits (their packaged code where it exists). Over-limit inputs are refused,
never truncated (the request is recorded ``unsupported``), as on the board.

``make_engine(name)``: ``pplx``, ``ckpt:<dir>`` (any code-readout checkpoint, e.g. ours), ``vega2``,
``lux2``, ``nox2``, ``torchcast``, ``clef``, ``jebadiah``, ``jade``, ``quyet``. ``BOARD`` maps names to
board engine ids.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from d25.vega.common import decision_format as df

QWEN_LOCAL = "/models/Qwen3.8-27B"
MODELS = "/data/d25/vega/proxy/models"
# 2026-10-03 release commits (the 10-08 commits only edit card metadata, which the package manifest rejects)
RELEASE = {
    "vega2": "7aec49ae11a18741706da549ab626b9052795fe7",
    "lux2": "78bf3c03d9147aeb30b641edfe0e30ed04887ca5",
    "nox2": "25e8f67d1b486c647222df3aac640d2d5d736bbe",
}
REPOS = {
    "pplx": "perplexity-ai/pplx-decider-v1.1-27b",
    "vega2": "vllm-sr/Decision-2.0-Vega-27B",
    "lux2": "vllm-sr/Decision-2.0-Lux-9B",
    "nox2": "vllm-sr/Decision-2.0-Nox-4B",
    "torchcast": "torchcast-ai/torchcast-decision-27b",
    "clef": "Cloudflare/clef",
    "jebadiah": "frontier-infra/jebadiah-27b",
    "jade": "theunnecessarythings/JADE",
    "quyet": "chinhnc/Quyet-1.0-Large",
    "kev": "jaredpalmer/kev-27b",
    "eikos": "caiovicentino1/Eikos-27B-FP8",
    "deck31b": "google/gemma-4-31B-it",
}
BOARD = {
    "pplx": "pplx-decider-v1.1-27b",
    "vega2": "decision-2.0-vega-27b",
    "lux2": "decision-2.0-lux-9b",
    "nox2": "decision-2.0-nox-4b",
    "torchcast": "torchcast-decision-27b",
    "clef": "clef",
    "jebadiah": "jebadiah-27b",
    "jade": "jade",
    "quyet": "quyet-1.0-large",
    "kev": "kev-27b",
    "eikos": "eikos-27b-fp8",
    "deck31b": "deck31b",
}


def local_repo(name: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(REPOS[name], local_files_only=True)


class Unsupported(Exception):
    pass


def _answer(question, probs):
    return df.to_answer(question, probs)


class ReadoutEngine:
    """Code-readout checkpoints through the shared engine; batches every question of a chunk of requests."""

    def __init__(self, path, device, **kw):
        from d25.vega.eval.engine import CodeReadoutModel

        self.model = CodeReadoutModel(path, device=device, **kw)

    def batch(self, rows):
        flat, owner = [], []
        for i, r in enumerate(rows):
            for k, q in r["questions"].items():
                flat.append({"state": r["state"], "question": q})
                owner.append((i, k))
        probs = self.model.predict(flat, on_over_limit="none")
        answers = [dict() for _ in rows]
        bad = set()
        for (i, k), p in zip(owner, probs):
            if p is None:
                bad.add(i)
            else:
                answers[i][k] = _answer(rows[i]["questions"][k], p)
        return [
            (
                ("unsupported", None, "over the input limit")
                if i in bad
                else ("ok", {"answers": answers[i]}, None)
            )
            for i in range(len(rows))
        ]


class Decision2Engine:
    """Our released Decision 2.0 packages (native System One runtime, T = 1)."""

    def __init__(self, name, device):
        # the package verifier refuses symlinks, so it needs a real copy (snapshot_download(local_dir=...))
        rev = RELEASE[name]
        path = os.path.join(MODELS, REPOS[name].split("/")[1] + "@" + rev[:8])
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            snapshot_download(REPOS[name], revision=rev, local_dir=path)
        sys.path.insert(0, path)
        from decision2 import Decision2

        base = QWEN_LOCAL if name == "vega2" and Path(QWEN_LOCAL).exists() else None
        self.model = Decision2.from_pretrained(path, device=device, base_path=base)

    def __call__(self, state, questions):
        try:
            return self.model.system_one(state=state, questions=questions)
        except ValueError as e:
            if "token" in str(e).lower() or "budget" in str(e).lower():
                raise Unsupported(str(e)) from e
            raise


def _repeat(x, rows):
    if isinstance(x, dict):
        return {k: _repeat(v, rows) for k, v in x.items()}
    return x.repeat(rows, *[1] * (x.dim() - 1))


def _torchcast_rows_cache(self, shared, rows):
    """Torchcast's own ``_rows_cache`` with one compatibility change: transformers 5.17 keeps linear-attention
    conv/recurrent states in dicts, so every tensor inside is repeated (their code assumed plain tensors).
    """
    from transformers import DynamicCache
    from transformers.cache_utils import LinearAttentionLayer

    cache = DynamicCache(config=self.body.config)
    for i, layer in enumerate(shared.layers):
        if not isinstance(layer, LinearAttentionLayer):
            cache.layers[i] = layer
            continue
        copy = LinearAttentionLayer()
        for name in ("conv_states", "recurrent_states"):
            setattr(copy, name, _repeat(getattr(layer, name), rows))
        copy.dtype, copy.device = layer.dtype, layer.device
        copy.max_batch_size, copy.conv_kernel_size = rows, layer.conv_kernel_size
        copy.is_conv_states_initialized = copy.is_recurrent_states_initialized = (
            copy.has_previous_state
        ) = True
        cache.layers[i] = copy
    return cache


class TorchcastEngine:
    def __init__(self, device):
        path = local_repo("torchcast")
        os.environ.setdefault("STARTLUX_ALLOW_SLOW", "1")
        sys.path.insert(0, path)
        from torchcast_decision import model as M

        M.TorchcastDecision._rows_cache = _torchcast_rows_cache
        self.model = M.TorchcastDecision(path, device=device, images=False)

    def __call__(self, state, questions):
        answers, _ = self.model.decide(state, questions)
        return {"answers": answers}


class ClefEngine:
    def __init__(self, device):
        path = local_repo("clef")
        sys.path.insert(0, path)
        import joint_schema_model as J

        self.J = J
        self.model, self.processor = J.load_release_model(path, device=device)
        self.model.eval()

    def __call__(self, state, questions):
        import torch

        try:
            with torch.inference_mode():
                # the board's clef run had no unsupported requests, so its limit exceeded systemone()'s 16,384 default
                return self.J.systemone(
                    self.model,
                    self.processor,
                    {"model": "clef", "state": state, "questions": questions},
                    max_length=131072,
                )
        except ValueError as e:
            if "length" in str(e).lower() or "token" in str(e).lower():
                raise Unsupported(str(e)) from e
            raise


class JebadiahEngine:
    def __init__(self, device):
        import torch

        path = local_repo("jebadiah")
        sys.path.insert(0, os.path.join(path, "scripts"))
        from jebadiah_model import Scorer, load_base, load_tokenizer, read_temperatures
        from jebadiah_prompt import answer_from_probs

        self.answer_from_probs = answer_from_probs
        tok = load_tokenizer(path)
        model = load_base(
            path, attn_implementation="sdpa", dtype=torch.bfloat16, device=device
        )
        self.scorer = Scorer(
            model, tok, temperatures=read_temperatures(path), device=device
        )

    def __call__(self, state, questions):
        # mechanical translation the served route applies: structured instructions / option descriptions as JSON text
        text = lambda v: (
            v
            if isinstance(v, str) or v is None
            else json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        )
        qs = {
            k: {
                **q,
                "instructions": text(q.get("instructions")),
                **(
                    {"criteria": {c: text(d) for c, d in q["criteria"].items()}}
                    if q.get("criteria")
                    else {}
                ),
            }
            for k, q in questions.items()
        }
        res = self.scorer.score(state, qs)
        return {
            "answers": {
                qid: self.answer_from_probs(qs[qid], keys, probs)
                for qid, (keys, probs) in res.items()
            }
        }


class JadeEngine:
    """vLLM-based (run one shard per 1-GPU pod)."""

    def __init__(self, device):
        path = local_repo("jade")
        sys.path.insert(0, path)
        from jade.engine import JadeEngine as Native

        self.native = Native(path, gpu_memory_utilization=0.88)

    def __call__(self, state, questions):
        try:
            response, _ = self.native(state, questions)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "UnsupportedInput":
                raise Unsupported(str(e)) from e
            raise
        return response


class QuyetEngine:
    """``quyet`` package with the 10-option cap lifted, as the lab ran it for the board."""

    def __init__(self, device):
        import quyet
        import quyet.questions

        quyet.questions.MAX_OPTIONS = 255
        self.model = quyet.load(local_repo("quyet"), device=device)

    def __call__(self, state, questions):
        text = lambda v: (
            v
            if isinstance(v, str) or v is None
            else json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        )
        qs = {
            k: {
                **q,
                "instructions": text(q.get("instructions")),
                **(
                    {"criteria": {c: text(d) for c, d in q["criteria"].items()}}
                    if q.get("criteria")
                    else {}
                ),
            }
            for k, q in questions.items()
        }
        return self.model.predict(state if state != "" else {}, qs)


def _spawn_local_server(cmd, cwd, env, log):
    import subprocess

    return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)


def _probe_local(url, timeout):
    import urllib.request

    urllib.request.urlopen(url, timeout=timeout)


class ServerEngine:
    """Start the entrant's own /v1/systemone server in this (1-GPU) pod and post every request to it."""

    def __init__(
        self,
        cmd,
        port,
        cwd=None,
        env=None,
        model="anchor",
        timeout_s=3600,
        health="/v1/models",
    ):
        import time

        self.url = f"http://127.0.0.1:{port}/v1/systemone"
        self.model = model
        log = open(os.path.join(cwd or ".", f"server-{port}.log"), "a")
        self.proc = _spawn_local_server(cmd, cwd, {**os.environ, **(env or {})}, log)
        t = time.time()
        while time.time() - t < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}")
            try:
                _probe_local(f"http://127.0.0.1:{port}{health}", 5)
                return
            except Exception:  # noqa: BLE001
                time.sleep(10)
        raise TimeoutError("server did not become healthy")

    def __call__(self, state, questions):
        import json as _json
        import urllib.error
        import urllib.request

        body = _json.dumps(
            {"model": self.model, "state": state, "questions": questions}
        ).encode()
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return _json.loads(r.read())
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            if e.code in (400, 413, 422) and any(
                m in text.lower()
                for m in ("context", "token", "too long", "options", "limit", "exceed")
            ):
                raise Unsupported(text[:300]) from e
            raise RuntimeError(f"HTTP {e.code}: {text[:300]}") from e


def kev_engine(device):
    import subprocess

    src = "/data/d25/vega/proxy/src/kev"
    if not os.path.isdir(src):
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/jaredpalmer/kev.git",
                src,
            ]
        )
    rev = subprocess.check_output(
        ["git", "-C", src, "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"kev source commit {rev}", flush=True)
    env = {"PYTHONPATH": src + ":" + os.environ.get("PYTHONPATH", "")}
    return ServerEngine(
        [sys.executable, "-m", "kev.serve", "--run", REPOS["kev"], "--port", "8008"],
        8008,
        cwd=src,
        env=env,
        model="kev-27b",
    )


def make_engine(name: str, device: str = "cuda:0", options: dict | None = None):
    options = options or {}
    if name == "pplx":
        return ReadoutEngine(local_repo("pplx"), device, readout_dtype="bfloat16")
    if name.startswith("ckpt:"):
        return ReadoutEngine(name[5:], device, **options)
    if name in ("vega2", "lux2", "nox2"):
        return Decision2Engine(name, device)
    return {
        "torchcast": TorchcastEngine,
        "clef": ClefEngine,
        "jebadiah": JebadiahEngine,
        "jade": JadeEngine,
        "quyet": QuyetEngine,
        "kev": kev_engine,
    }[name](device)
