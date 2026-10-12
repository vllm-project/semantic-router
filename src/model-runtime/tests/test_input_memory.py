"""Peak memory of the runtime process on the Response API's 5 MiB input.

The router's response cache embeds a request's text with ``truncate`` and no
token budget, so ``response-api-edge-large-input`` (5 MiB + 1 KiB of English)
reaches the runtime whole. This serves a tiny embedding package with the Vela
Embedding window (32,768 tokens) in a server process, sends that input as the
router does, and the same size of base64 and a rejected input, and measures
the process's peak RSS growth during the request. Read whole, the package's
word-level tokenizer reads 1.17 million tokens from the English input.
"""

from __future__ import annotations

import base64
import json
import random
from pathlib import Path

import pytest
from vllm_srun.testing.embed_packages import write_embedding_package

from .test_server import UnixConnection, request, start, wait_ready

LARGE_INPUT = (5 << 20) + 1024
# Peak RSS growth of the request, measured on a 12-core x86 VM: 630 to 665 MiB
# when the runtime tokenized the whole English input, 176 to 188 MiB reading it
# only as far as its budget, 88 MiB of which is the model's forward over its
# 32,768-token window; 208 to 210 MiB for 5 MiB of base64, which has no space.
BUDGET_MIB = 256
# A rejected input costs its request body (the bytes, the decoded text and the
# parsed JSON, 20 MiB) and the prefix that decides its budget: 50 MiB. The
# fixture's word-level tokenizer drops whitespace, so it cannot reject early.
REJECTED_BUDGET_MIB = 80


def status(pid: int) -> dict[str, int]:
    out = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            out[key] = int(value.split()[0])
    return out


def peak_growth_mib(pid: int, send) -> tuple[float, tuple[int, bytes]]:
    """The process's peak RSS growth while ``send`` runs, from a reset high-water mark."""
    base = status(pid)["VmRSS"]
    Path(f"/proc/{pid}/clear_refs").write_text("5")
    answer = send()
    return (status(pid)["VmHWM"] - base) / 1024, answer


def large_text() -> str:
    sentence = "The quick brown fox jumps over the lazy dog. "
    return (sentence * (LARGE_INPUT // len(sentence) + 1))[:LARGE_INPUT]


def large_base64() -> str:
    """A 5 MiB run without a space, as an inlined file reaches the router."""
    return base64.b64encode(random.Random(0).randbytes(LARGE_INPUT * 3 // 4)).decode()


@pytest.fixture()
def serve(tmp_path):
    if not Path("/proc/self/clear_refs").exists():
        pytest.skip("peak RSS needs Linux procfs")
    package = write_embedding_package(tmp_path / "embedding")
    processes = []

    def run(*args: str):
        path = str(tmp_path / f"run{len(processes)}" / "runtime.sock")
        process = start(
            [str(package), "--uds", path, "--max-request-bytes", str(64 << 20), *args]
        )
        processes.append(process)
        wait_ready(lambda: UnixConnection(path), process)

        def embed(text: str, overflow: str = "truncate") -> tuple[int, bytes]:
            body = {"input": [text], "options": {"overflow": overflow}}
            return request(
                UnixConnection(path, timeout=300), "POST", "/v1/embeddings", body
            )

        assert embed("warm up")[0] == 200
        return process, embed

    yield run
    for process in processes:
        process.terminate()
        process.wait(30)


@pytest.mark.parametrize("kind", ["english", "base64"])
def test_a_5_mib_input_grows_the_runtime_less_than_its_budget(serve, kind):
    process, embed = serve()
    text = large_text() if kind == "english" else large_base64()
    growth, (code, raw) = peak_growth_mib(process.pid, lambda: embed(text))
    assert code == 200, raw[:500]
    usage = json.loads(raw)["data"][0]["input"]
    assert usage["processed_tokens"] == 32768 and usage["truncated"]
    assert usage["tokens_lower_bound"] and usage["tokens"] > 32768
    assert growth < BUDGET_MIB, f"the 5 MiB input grew the runtime by {growth:.0f} MiB"


def test_a_5_mib_input_over_a_reject_budget_costs_little_more_than_its_body(serve):
    process, embed = serve()
    growth, (code, raw) = peak_growth_mib(
        process.pid, lambda: embed(large_text(), overflow="reject")
    )
    assert code == 200
    assert json.loads(raw)["data"][0]["error"] == "max_length_exceeded"
    assert (
        growth < REJECTED_BUDGET_MIB
    ), f"the rejected input grew the runtime by {growth:.0f} MiB"
