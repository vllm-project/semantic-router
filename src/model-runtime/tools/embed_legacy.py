"""Legacy comparison for embedders and rerankers: the router's native facade vs the runtime.

The legacy side runs the router's native facade (``pkg/modelruntime/native``
at the legacy commit) on the CPU as the router did (candle for Vela and Qwen3;
ONNX Runtime on the prepared bundle for Omni text, images and audio), in a Go
test written into a copy of that commit's tree with its built bindings (the
setup of ``tools/legacy_parity.py``). The runtime side serves the same
packages in-process through ``Runtime.call`` with its result cache off (Omni's
published snapshot on the native engine, or its prepared bundle on
onnxruntime with ``--omni-engine onnxruntime``), timed
inside an event loop that runs forever on its own thread, as the server's
handler is. Both sides build every input's request before timing. Both
answer one request at a time (one text per embedding call, a query with its
documents per rerank call), repeated, on the same inputs; ``--concurrency``
adds a timed closed-loop load on both: goroutines on the legacy side, and
on the runtime side concurrent requests on that event loop, as the server
serves concurrent connections.

    python3 tools/embed_legacy.py legacy --tree TREE --cache HF --flat DIR --out legacy.jsonl
    python3 tools/embed_legacy.py runtime --cache HF --prepared DIR --out runtime.jsonl [--device rocm:0]
    python3 tools/embed_legacy.py compare --legacy legacy.jsonl --runtime runtime.jsonl --out record.json

``build`` compiles the legacy side's test binary for ``ab``, which alternates
legacy and runtime calls on the same cores (the order flips every round, and
load windows rotate) so both sides see the same moment's contention on a
shared host; ``--gap-ms`` pauses before every call so neither side's
spinning thread pools share the cores with the other side's call. ``ab``
reports latency and throughput only (``compare`` checks values). Confine
both sides with a cgroup cpuset (a container or a systemd scope): ONNX
Runtime pins its threads to CPUs it reads from the host, past a ``taskset``
mask.

    python3 tools/embed_legacy.py build --tree TREE --cache HF --flat DIR --out legacy.test
    python3 tools/embed_legacy.py ab --binary legacy.test --cache HF --out ab.json [--legacy-cpus 16] [--gap-ms 50]
    python3 tools/embed_legacy.py ab --baseline-engine onnxruntime --cache HF --prepared DIR --out ab.json

With ``--baseline-engine`` the baseline side is the runtime itself in a
second process (the ``serve`` command, which speaks the legacy binary's line
protocol), with Omni on that engine: the same alternation compares two
runtime engines.

``legacy`` and ``build`` need only the standard library (the runner image has
a bare Python); ``compare`` needs NumPy.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed_corpus import PARAGRAPHS, QUERIES, RERANK, texts
from legacy_parity import REPO, flat_copy, percentile, read_lines

GO_TEST = "zz_legacy_embed_dump_test.go"
EMBEDDING = (
    "vllm-sr/Vela-1.0-Encoder-307M-Embedding",
    "1e57cebf5a7b7fec6e6973f05bbca97c5cca4436",
)
RERANKER = (
    "vllm-sr/Vela-1.0-Encoder-307M-Reranker",
    "a388e41cbbd5dc5f16b6389fa76d0b8b8a38a8bf",
)
QWEN3 = ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
OMNI_NANO = ("vllm-sr/Vela-1.0-Omni-Nano", "2ff2d66385dbdd661a560ec3e8bcb45a0527d92e")
OMNI_MINI = ("vllm-sr/Vela-1.0-Omni-Mini", "801bae3ad28df6891408f0e0441c676b30e132e3")


class Job(NamedTuple):
    """One legacy binding: model (repo, revision), task mode, legacy adapter, view, input modality."""

    model: tuple[str, str]
    mode: str
    adapter: str
    view: tuple[int, int] = (0, 0)  # (dimension, layer); 0 is the model's default
    modality: str = "text"


JOBS: dict[str, Job] = {
    "embedding": Job(EMBEDDING, "embedding", "mmbert"),
    "embedding_d256": Job(EMBEDDING, "embedding", "mmbert", (256, 0)),
    "embedding_l11": Job(EMBEDDING, "embedding", "mmbert", (0, 11)),
    "rerank": Job(RERANKER, "rerank", "vela_reranker", (768, 22)),
    "rerank_l6d256": Job(RERANKER, "rerank", "vela_reranker", (256, 6)),
    "qwen3": Job(QWEN3, "embedding", "qwen3"),
    **{
        f"omni_{size}_{modality}": Job(model, "omni", "vela_omni", modality=modality)
        for size, model in (("nano", OMNI_NANO), ("mini", OMNI_MINI))
        for modality in ("text", "image", "audio")
    },
}
PROVIDERS = {"embedding": "candle", "rerank": "candle", "omni": "ort"}
LEGACY_PATHS = {
    "candle": "router native facade, candle on the CPU (one call per text, per query)",
    "ort": "router native facade, ONNX Runtime on the prepared Omni bundle (one call per input)",
}
MAX_TOKENS = {"embedding": 8192, "rerank": 8192, "omni": 512}
# Design section 17 (None: not gated); a rerank tie is a legacy logit margin under ``tie``.
CPU_THRESHOLDS = {"min_cosine": 0.99999, "max_abs": 1e-4, "tie": 1e-3}
ROCM_THRESHOLDS = {"min_cosine": 0.9995, "max_abs": None, "tie": 2e-2}

GO_TEMPLATE = r"""//go:build !windows && cgo

package native

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type embedDumpInput struct {
	ID        string    `json:"id"`
	Text      string    `json:"text"`
	Query     string    `json:"query"`
	Documents []string  `json:"documents"`
	Image     string    `json:"image"`
	PCM       []float32 `json:"pcm"`
	Rate      int       `json:"rate"`
	Channels  int       `json:"channels"`
}

type embedDumpJob struct {
	Job, Path, Revision, Mode, Adapter, Provider, Modality string
	MaxTokens, Dimension, Layer                            int
	Inputs                             []embedDumpInput
	Repeats, Concurrency               int
	Seconds                            float64
}

type embedDumpLine struct {
	Job       string          `json:"job"`
	ID        string          `json:"id"`
	Result    json.RawMessage `json:"result,omitempty"`
	Error     string          `json:"error,omitempty"`
	LatencyNS []int64         `json:"latency_ns,omitempty"`
}

type embedDumpLoad struct {
	Job         string  `json:"job"`
	Concurrency int     `json:"concurrency"`
	Calls       int     `json:"calls"`
	Seconds     float64 `json:"seconds"`
	LatencyNS   []int64 `json:"latency_ns"`
}

type embedDumpCall func(context.Context, embedDumpInput) (any, error)

func TestLegacyEmbedDump(t *testing.T) {
	raw, err := os.ReadFile(os.Getenv("LEGACY_EMBED_JOBS"))
	if err != nil {
		t.Fatal(err)
	}
	var jobs []embedDumpJob
	if err := json.Unmarshal(raw, &jobs); err != nil {
		t.Fatal(err)
	}
	out, err := os.Create(os.Getenv("LEGACY_EMBED_OUT"))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	encoder := json.NewEncoder(out)
	runtime := New(nil)
	ctx := context.Background()
	for _, job := range jobs {
		call, closeTask := embedDumpTask(t, runtime, job)
		for range 3 {
			_, _ = call(ctx, job.Inputs[0])
		}
		for _, input := range job.Inputs {
			line := embedDumpLine{Job: job.Job, ID: input.ID}
			for repeat := 0; repeat < max(job.Repeats, 1); repeat++ {
				start := time.Now()
				result, callErr := call(ctx, input)
				line.LatencyNS = append(line.LatencyNS, time.Since(start).Nanoseconds())
				if repeat > 0 {
					continue
				}
				if callErr != nil {
					line.Error = callErr.Error()
				} else if line.Result, err = json.Marshal(result); err != nil {
					t.Fatal(err)
				}
			}
			if err := encoder.Encode(line); err != nil {
				t.Fatal(err)
			}
		}
		if job.Concurrency > 0 {
			if err := encoder.Encode(embedDumpLoadRun(ctx, job, call)); err != nil {
				t.Fatal(err)
			}
		}
		closeTask()
	}
}

// TestLegacyEmbedServe answers stdin commands so a driver can alternate legacy and runtime
// work: "call<TAB>job<TAB>id" prints the call's latency in nanoseconds (-1 on error) and
// "load<TAB>job<TAB>concurrency<TAB>seconds" a load window as JSON.
func TestLegacyEmbedServe(t *testing.T) {
	raw, err := os.ReadFile(os.Getenv("LEGACY_EMBED_JOBS"))
	if err != nil {
		t.Fatal(err)
	}
	var jobs []embedDumpJob
	if err := json.Unmarshal(raw, &jobs); err != nil {
		t.Fatal(err)
	}
	runtime := New(nil)
	ctx := context.Background()
	calls := map[string]embedDumpCall{}
	byName := map[string]embedDumpJob{}
	inputs := map[string]embedDumpInput{}
	for _, job := range jobs {
		call, closeTask := embedDumpTask(t, runtime, job)
		defer closeTask()
		calls[job.Job], byName[job.Job] = call, job
		for _, input := range job.Inputs {
			inputs[job.Job+"\t"+input.ID] = input
		}
		for range 3 {
			_, _ = call(ctx, job.Inputs[0])
		}
	}
	fmt.Println("READY")
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		fields := strings.Split(scanner.Text(), "\t")
		if fields[0] == "load" {
			job := byName[fields[1]]
			job.Concurrency, _ = strconv.Atoi(fields[2])
			job.Seconds, _ = strconv.ParseFloat(fields[3], 64)
			report, _ := json.Marshal(embedDumpLoadRun(ctx, job, calls[job.Job]))
			fmt.Println(string(report))
			continue
		}
		start := time.Now()
		_, callErr := calls[fields[1]](ctx, inputs[fields[1]+"\t"+fields[2]])
		latency := time.Since(start).Nanoseconds()
		if callErr != nil {
			latency = -1
		}
		fmt.Println(latency)
	}
}

func embedDumpLoadRun(ctx context.Context, job embedDumpJob, call embedDumpCall) embedDumpLoad {
	deadline := time.Now().Add(time.Duration(job.Seconds * float64(time.Second)))
	var mutex sync.Mutex
	var wait sync.WaitGroup
	report := embedDumpLoad{Job: job.Job, Concurrency: job.Concurrency}
	start := time.Now()
	for worker := range job.Concurrency {
		wait.Add(1)
		go func(worker int) {
			defer wait.Done()
			for index := worker; time.Now().Before(deadline); index += job.Concurrency {
				began := time.Now()
				_, _ = call(ctx, job.Inputs[index%len(job.Inputs)])
				elapsed := time.Since(began).Nanoseconds()
				mutex.Lock()
				report.Calls++
				report.LatencyNS = append(report.LatencyNS, elapsed)
				mutex.Unlock()
			}
		}(worker)
	}
	wait.Wait()
	report.Seconds = time.Since(start).Seconds()
	return report
}

func embedDumpSpec(job embedDumpJob, contract, overflow string) config.ResolvedModelBinding {
	return config.ResolvedModelBinding{
		Recipe: "parity", Name: job.Job,
		Binding: config.ModelBinding{Deployment: job.Job, Adapter: job.Adapter, Contract: contract},
		Deployment: config.ModelDeployment{
			Artifact: job.Path, Revision: job.Revision, Provider: job.Provider, Device: "cpu", Precision: "native",
			Input: config.ModelInputBudget{MaxTokens: job.MaxTokens, Overflow: overflow},
		},
	}
}

func embedDumpTask(t *testing.T, runtime *Runtime, job embedDumpJob) (embedDumpCall, func()) {
	ctx := context.Background()
	switch job.Mode {
	case "embedding":
		provider, err := runtime.Embedding(ctx, embedDumpSpec(job, "embedding.v1", "truncate"), job.Dimension, job.Layer)
		if err != nil {
			t.Fatal(err)
		}
		options := embedding.Options{Dimension: job.Dimension, Layer: job.Layer}
		return func(ctx context.Context, in embedDumpInput) (any, error) {
			return provider.EmbedWithOptions(ctx, in.Text, options)
		}, func() { _ = provider.Close() }
	case "rerank":
		spec := embedDumpSpec(job, config.RelevanceScoresContract, "reject")
		spec.Binding.PairScorer = &config.PairScorerSelection{Layer: job.Layer, Dimension: job.Dimension}
		scorer, err := runtime.Relevance(ctx, spec)
		if err != nil {
			t.Fatal(err)
		}
		return func(ctx context.Context, in embedDumpInput) (any, error) {
			pairs := make([]tasks.QueryDocument, len(in.Documents))
			for i, document := range in.Documents {
				pairs[i] = tasks.QueryDocument{Query: in.Query, Document: document}
			}
			return scorer.ScorePairs(ctx, "parity", pairs)
		}, func() { _ = scorer.Close() }
	case "omni":
		provider, err := runtime.Embedding(ctx, embedDumpSpec(job, "embedding.v1", "reject"), 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		return func(ctx context.Context, in embedDumpInput) (any, error) {
			switch job.Modality {
			case "image":
				data, err := base64.StdEncoding.DecodeString(in.Image)
				if err != nil {
					return nil, err
				}
				return provider.EmbedImage(ctx, data, 0)
			case "audio":
				return provider.EmbedAudio(ctx, embedding.AudioRequest{PCM: in.PCM, SampleRate: in.Rate, Channels: in.Channels})
			}
			return provider.EmbedWithOptions(ctx, in.Text, embedding.Options{})
		}, func() { _ = provider.Close() }
	}
	t.Fatalf("unknown mode %q", job.Mode)
	return nil, nil
}
"""


def snapshot(cache: Path, repo: str, revision: str) -> Path:
    path = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not (path / "config.json").is_file():
        raise SystemExit(f"{repo}@{revision} is not in {cache}")
    return path


def omni_inputs(bundle: Path, modality: str) -> list[dict[str, Any]]:
    """The bundle's golden images (encoded bytes) or channel-major PCM, or the short corpus texts."""
    golden = bundle / "golden"
    index = json.loads((golden / "index.json").read_text(encoding="utf-8"))
    if modality == "image":
        return [
            {
                "id": f"i{i}",
                "image": base64.b64encode(
                    (golden / case["file"]).read_bytes()
                ).decode(),
                "format": Path(case["file"]).suffix[1:].replace("jpg", "jpeg"),
            }
            for i, case in enumerate(index["images"])
        ]
    if modality == "audio":
        inputs = []
        for i, case in enumerate(index["audio"]):
            shape = case["pcm"]["shape"]
            pcm = array("f", (golden / case["pcm"]["file"]).read_bytes())
            channels = shape[0] if len(shape) == 2 else 1
            inputs.append(
                {
                    "id": f"a{i}",
                    "pcm": pcm.tolist(),
                    "rate": case["sampling_rate"],
                    "channels": channels,
                }
            )
        return inputs
    return [
        {"id": f"t{i}", "text": text} for i, text in enumerate(QUERIES + PARAGRAPHS)
    ]


def job_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    corpus = [{"id": f"t{i}", "text": text} for i, text in enumerate(texts())]
    sets = [
        {"id": f"q{i}", "query": query, "documents": list(documents)}
        for i, (query, documents) in enumerate(RERANK)
    ]
    specs = []
    for name in args.jobs.split(",") if args.jobs else JOBS:
        job = JOBS[name]
        repo, revision = job.model
        if job.mode == "omni":
            path = Path(args.prepared) / repo.split("/")[-1].lower()
            inputs = omni_inputs(path, job.modality)
        else:
            path = snapshot(Path(args.cache), repo, revision)
            if getattr(args, "flat", None):
                path = flat_copy(
                    path, Path(args.flat) / f"{repo.split('/')[-1]}-{revision[:12]}"
                )
            inputs = corpus if job.mode == "embedding" else sets
        specs.append(
            {
                "Job": name,
                "Repo": repo,
                "Path": str(path),
                "Revision": revision,
                "Mode": job.mode,
                "Adapter": job.adapter,
                "Provider": PROVIDERS[job.mode],
                "Modality": job.modality,
                "MaxTokens": MAX_TOKENS[job.mode],
                "Dimension": job.view[0],
                "Layer": job.view[1],
                "Inputs": inputs[: args.limit] if args.limit else inputs,
                "Repeats": args.repeats,
                "Concurrency": args.concurrency,
                "Seconds": args.seconds,
            }
        )
    return specs


def legacy_env(tree: Path, jobs: Path) -> dict[str, str]:
    """The cgo and loader environment of the legacy tree's bindings, plus the jobs file."""
    libraries = [
        tree / name / "target" / "release"
        for name in ("candle-binding", "ml-binding", "nlp-binding", "onnx-binding")
    ]
    return {
        **os.environ,
        "CGO_ENABLED": "1",
        "CGO_CFLAGS": f"-I{tree / 'candle-binding'}",
        "CGO_LDFLAGS": " ".join(f"-L{p}" for p in libraries)
        + " -lcandle_semantic_router -lml_semantic_router -lnlp_binding",
        "LD_LIBRARY_PATH": ":".join(
            [*(str(p) for p in libraries), os.environ.get("LD_LIBRARY_PATH", "")]
        ),
        "LEGACY_EMBED_JOBS": str(jobs),
    }


def write_legacy_test(args: argparse.Namespace, jobs: Path) -> Path:
    """The Go test in the legacy tree and the jobs file; returns the router module."""
    tree = Path(args.tree)
    package = tree / "src" / "semantic-router" / "pkg" / "modelruntime" / "native"
    (package / GO_TEST).write_text(GO_TEMPLATE, encoding="utf-8")
    jobs.write_text(json.dumps(job_specs(args)), encoding="utf-8")
    return tree / "src" / "semantic-router"


def run_legacy(args: argparse.Namespace) -> None:
    """Run the jobs through the legacy facade with the built bindings."""
    jobs = Path(args.out).with_suffix(".jobs.json")
    module = write_legacy_test(args, jobs)
    env = {
        **legacy_env(Path(args.tree), jobs),
        "LEGACY_EMBED_OUT": str(Path(args.out).resolve()),
    }
    command = ["go", "test", "-count=1", "-timeout", "6h"]
    command += ["-run", "^TestLegacyEmbedDump$", "./pkg/modelruntime/native/"]
    subprocess.run(command, cwd=module, env=env, check=True)


def build_legacy(args: argparse.Namespace) -> None:
    """Compile the legacy side's test binary (serve mode for ``ab``) next to its jobs file."""
    jobs = Path(args.out).with_suffix(".jobs.json")
    module = write_legacy_test(args, jobs)
    command = ["go", "test", "-c", "-o", str(Path(args.out).resolve())]
    command.append("./pkg/modelruntime/native/")
    subprocess.run(
        command, cwd=module, env=legacy_env(Path(args.tree), jobs), check=True
    )


def float_wav(pcm: list[float], rate: int, channels: int) -> bytes:
    """Channel-major PCM as an interleaved IEEE float WAV (what the router forwards)."""
    frames = len(pcm) // channels
    data = array("f", bytes(4 * len(pcm)))
    for channel in range(channels):
        data[channel::channels] = array(
            "f", pcm[channel * frames : (channel + 1) * frames]
        )
    fmt = struct.pack(
        "<HHIIHH", 3, channels, rate, rate * 4 * channels, 4 * channels, 32
    )
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", 4 * len(pcm)) + data.tobytes()
    return b"RIFF" + struct.pack("<I", len(body)) + body


def runtime_request(
    spec: dict[str, Any], item: dict[str, Any], model: str
) -> tuple[str, dict[str, Any]]:
    """The surface and body of one legacy call's equivalent."""
    if spec["Mode"] == "rerank":
        body = {
            "model": model,
            "query": item["query"],
            "documents": item["documents"],
            "layer": spec["Layer"],
            "dimensions": spec["Dimension"],
        }
        return "rerank", body
    if "image" in item:
        url = f"data:image/{item['format']};base64,{item['image']}"
        value: Any = [{"type": "image_url", "image_url": {"url": url}}]
    elif "pcm" in item:
        wav = float_wav(item["pcm"], item["rate"], item["channels"])
        audio = {"data": base64.b64encode(wav).decode(), "format": "wav"}
        value = [{"type": "input_audio", "input_audio": audio}]
    else:
        value = item["text"]
    options = {"max_tokens": spec["MaxTokens"], "overflow": "truncate"}
    body = {
        "model": model,
        "input": value,
        "options": {"overflow": "reject"} if spec["Mode"] == "omni" else options,
    }
    if spec["Dimension"]:
        body["dimensions"] = spec["Dimension"]
    if spec["Layer"]:
        body["layer"] = spec["Layer"]
    return "embeddings", body


def wire_size(body: dict[str, Any]) -> int:
    """The encoded body size the HTTP server passes to ``Runtime.call`` (small bodies plan inline)."""
    return len(json.dumps(body, separators=(",", ":")).encode("utf-8"))


class Request(NamedTuple):
    """One input's ``Runtime.call`` arguments; the server reads the size off the request it received."""

    surface: str
    body: dict[str, Any]
    size: int


def runtime_requests(spec: dict[str, Any], model: str) -> list[Request]:
    """Every input's request, built before any timing, as the legacy side's inputs are."""
    requests = []
    for item in spec["Inputs"]:
        surface, body = runtime_request(spec, item, model)
        requests.append(Request(surface, body, wire_size(body)))
    return requests


class LoopThread:
    """An event loop running forever on its own thread, as the server's does; calls are timed inside it.

    asyncio's loop: over HTTP the server answers as fast on asyncio as on
    uvloop, but uvloop driven from another thread adds a wake-up per call
    that the server never pays.
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def call(self, runtime: Any, request: Request) -> tuple[int, int, dict[str, Any]]:
        """(nanoseconds inside the loop, HTTP status, body) of one ``Runtime.call``."""

        async def timed() -> tuple[int, int, dict[str, Any]]:
            start = time.perf_counter_ns()
            status, out = await runtime.call(
                request.surface, request.body, request.size
            )
            return time.perf_counter_ns() - start, status, out

        return self.run(timed())

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


def runtime_result(spec: dict[str, Any], out: dict[str, Any]) -> Any:
    if spec["Mode"] == "rerank":
        by_index = {r["index"]: r["logit"] for r in out["results"]}
        return [by_index[i] for i in range(len(by_index))]
    return out["data"][0]["embedding"]


def serve_runtime(
    args: argparse.Namespace, specs: list[dict[str, Any]]
) -> tuple[Any, dict[str, str]]:
    """A started in-process runtime serving every job's model, and the served names.

    Omni runs its published snapshot on the native engine, or with
    ``--omni-engine onnxruntime`` its prepared bundle (``--prepared``). The
    runtime is this checkout's, or with ``--runtime-src`` another tree's.
    """
    sys.path.insert(0, args.runtime_src or str(REPO / "src" / "model-runtime"))
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    names = {spec["Repo"]: spec["Repo"].split("/")[-1] for spec in specs}
    graphs = getattr(args, "omni_engine", "native") == "onnxruntime"
    paths = {
        spec["Repo"]: spec["Path"]
        for spec in specs
        if spec["Mode"] == "omni" and graphs
    }
    models = tuple(
        ModelConfig(
            model=paths.get(repo, repo),
            name=name,
            device=args.device,
            profile=args.profile,
            engine="onnxruntime" if repo in paths else "auto",
        )
        for repo, name in names.items()
    )
    config = ServeConfig(
        models=models,
        threads=args.threads,
        result_cache_entries=0,
        cache_dir=args.cache,
        offline=True,
    )
    runtime = Runtime(config)
    runtime.start(background=False)
    return runtime, names


def run_runtime(args: argparse.Namespace) -> None:
    specs = job_specs(args)
    runtime, names = serve_runtime(args, specs)
    loop = LoopThread()

    def call(request: Request) -> tuple[int, dict[str, Any]]:
        elapsed, status, out = loop.call(runtime, request)
        if status != 200:
            raise RuntimeError(json.dumps(out))
        return elapsed, out

    with open(args.out, "w", encoding="utf-8") as stream:
        for spec in specs:
            requests = runtime_requests(spec, names[spec["Repo"]])
            for _ in range(3):
                call(requests[0])
            for item, request in zip(spec["Inputs"], requests, strict=True):
                line: dict[str, Any] = {
                    "job": spec["Job"],
                    "id": item["id"],
                    "latency_ns": [],
                }
                for repeat in range(max(args.repeats, 1)):
                    elapsed, out = call(request)
                    line["latency_ns"].append(elapsed)
                    if repeat == 0:
                        line["result"] = runtime_result(spec, out)
                stream.write(json.dumps(line) + "\n")
            if args.concurrency:
                load = runtime_load(runtime, loop, spec["Job"], requests, args)
                stream.write(json.dumps(load) + "\n")
    loop.close()
    runtime.stop()


def runtime_load(
    runtime: Any,
    loop: LoopThread,
    job: str,
    requests: list[Request],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """The legacy load's closed loop: ``concurrency`` callers for ``seconds``, as concurrent requests on the server's loop."""

    async def load() -> dict[str, Any]:
        deadline = time.monotonic() + args.seconds
        latencies: list[int] = []

        async def caller(index: int) -> None:
            position = index
            while time.monotonic() < deadline:
                request = requests[position % len(requests)]
                start = time.perf_counter_ns()
                await runtime.call(request.surface, request.body, request.size)
                latencies.append(time.perf_counter_ns() - start)
                position += args.concurrency

        began = time.monotonic()
        await asyncio.gather(*(caller(index) for index in range(args.concurrency)))
        return {
            "job": job,
            "concurrency": args.concurrency,
            "calls": len(latencies),
            "seconds": time.monotonic() - began,
            "latency_ns": latencies,
        }

    return loop.run(load())


def legacy_command(binary: str, cpus: int | None) -> list[str]:
    """The serve-mode binary, seeing ``cpus`` CPUs in sysfs when given (needs root).

    The legacy ONNX Runtime sessions size their thread pools from the host's CPU
    count, never from the cpuset (the router never configured them); a private
    mount namespace shows them only the CPUs the run has.
    """
    command = [binary, "-test.run", "^TestLegacyEmbedServe$", "-test.timeout", "12h"]
    if not cpus:
        return command
    online = Path(tempfile.mkdtemp()) / "online"
    online.write_text(f"0-{cpus - 1}\n", encoding="utf-8")
    mounts = " && ".join(
        f"mount --bind {online} /sys/devices/system/cpu/{name}"
        for name in ("online", "possible", "present")
    )
    return ["unshare", "-m", "sh", "-c", f'{mounts} && exec "$0" "$@"', *command]


def run_serve(args: argparse.Namespace) -> None:
    """The runtime as ``ab``'s baseline side: the legacy binary's line protocol on stdin and stdout.

    ``call<TAB>job<TAB>id`` answers one input and prints its nanoseconds (-1
    on an error); ``load<TAB>job<TAB>callers<TAB>seconds`` prints a load
    window as JSON. ``READY`` follows the load.
    """
    specs = job_specs(args)
    runtime, names = serve_runtime(args, specs)
    loop = LoopThread()
    requests = {
        spec["Job"]: runtime_requests(spec, names[spec["Repo"]]) for spec in specs
    }
    positions = {
        spec["Job"]: {item["id"]: index for index, item in enumerate(spec["Inputs"])}
        for spec in specs
    }
    for spec in specs:
        for _ in range(3):
            loop.call(runtime, requests[spec["Job"]][0])
    print("READY", flush=True)
    for line in sys.stdin:
        command, job, *rest = line.rstrip("\n").split("\t")
        if command == "call":
            elapsed, status, _ = loop.call(
                runtime, requests[job][positions[job][rest[0]]]
            )
            print(elapsed if status == 200 else -1, flush=True)
        elif command == "load":
            args.concurrency, args.seconds = int(rest[0]), float(rest[1])
            window = runtime_load(runtime, loop, job, requests[job], args)
            print(json.dumps(window), flush=True)
    loop.close()
    runtime.stop()


def baseline_command(args: argparse.Namespace) -> list[str]:
    """The ``serve`` child that runs the jobs on ``--baseline-engine`` (the A side)."""
    command = [sys.executable, str(Path(__file__).resolve()), "serve"]
    command += ["--omni-engine", args.baseline_engine, "--cache", args.cache]
    command += ["--prepared", args.prepared, "--jobs", args.jobs, "--out", os.devnull]
    command += ["--device", args.device, "--profile", args.profile]
    if args.threads:
        command += ["--threads", str(args.threads)]
    if args.baseline_runtime:
        command += ["--runtime-src", args.baseline_runtime]
    return command


def run_ab(args: argparse.Namespace) -> None:
    """Alternate baseline and runtime calls per input on the same cores, round after round.

    The baseline (reported as ``legacy``) is the legacy facade's binary, or
    with ``--baseline-engine`` the runtime itself in another process, with
    Omni on that engine.
    """
    if args.baseline_engine:
        specs = job_specs(args)
        overrides = dict(entry.split("=", 1) for entry in args.baseline_env)
        legacy = subprocess.Popen(
            baseline_command(args),
            env={**os.environ, **overrides},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    else:
        jobs = Path(args.binary).with_suffix(".jobs.json")
        specs = json.loads(jobs.read_text(encoding="utf-8"))
        if args.jobs:
            specs = [spec for spec in specs if spec["Job"] in args.jobs.split(",")]
        legacy = subprocess.Popen(
            legacy_command(args.binary, args.legacy_cpus),
            env=legacy_env(Path(args.tree), jobs),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    assert legacy.stdin is not None and legacy.stdout is not None
    while legacy.stdout.readline().strip() != "READY":
        if legacy.poll() is not None:
            raise SystemExit("the legacy binary exited before it was ready")
    runtime, names = serve_runtime(args, specs)
    loop = LoopThread()
    requests = {
        spec["Job"]: runtime_requests(spec, names[spec["Repo"]]) for spec in specs
    }

    def call_legacy(spec: dict[str, Any], index: int) -> int:
        legacy.stdin.write(f"call\t{spec['Job']}\t{spec['Inputs'][index]['id']}\n")
        return int(legacy.stdout.readline())

    def call_runtime(spec: dict[str, Any], index: int) -> int:
        elapsed, status, _ = loop.call(runtime, requests[spec["Job"]][index])
        return elapsed if status == 200 else -1

    calls = {"legacy": call_legacy, "runtime": call_runtime}

    def pair(spec: dict[str, Any], index: int, legacy_first: bool) -> tuple[int, int]:
        """One legacy and one runtime call, each ``--gap-ms`` after the previous call.

        ONNX Runtime workers spin for milliseconds after a run; the gap keeps
        either side's spinning pools off the cores the next call uses.
        """
        sides = ("legacy", "runtime") if legacy_first else ("runtime", "legacy")
        elapsed = {}
        for side in sides:
            time.sleep(args.gap_ms / 1000)
            elapsed[side] = calls[side](spec, index)
        return elapsed["legacy"], elapsed["runtime"]

    report: dict[str, Any] = {}
    for spec in specs:
        for _ in range(3):
            call_runtime(spec, 0)
    for round_index in range(args.rounds):
        for spec in specs:
            entry = report.setdefault(
                spec["Job"], {"legacy": [], "runtime": [], "ratio": []}
            )
            for index in range(len(spec["Inputs"])):
                legacy_ns, runtime_ns = pair(spec, index, round_index % 2 == 0)
                if legacy_ns > 0 and runtime_ns > 0:
                    entry["legacy"].append(legacy_ns)
                    entry["runtime"].append(runtime_ns)
                    entry["ratio"].append(legacy_ns / runtime_ns)
        for spec in specs if args.concurrency else ():
            rates = report[spec["Job"]].setdefault(
                "throughput", {"legacy": [], "runtime": []}
            )
            sides = (
                ("legacy", "runtime") if round_index % 2 == 0 else ("runtime", "legacy")
            )
            for side in sides:
                time.sleep(args.gap_ms / 1000)
                if side == "legacy":
                    command = (
                        f"load\t{spec['Job']}\t{args.concurrency}\t{args.seconds}\n"
                    )
                    legacy.stdin.write(command)
                    window = json.loads(legacy.stdout.readline())
                else:
                    window = runtime_load(
                        runtime, loop, spec["Job"], requests[spec["Job"]], args
                    )
                rates[side].append(window["calls"] / window["seconds"])
    legacy.stdin.close()
    legacy.wait()
    loop.close()
    runtime.stop()
    summary = {
        job: {
            "pairs": len(values["ratio"]),
            "legacy": latency([{"latency_ns": values["legacy"]}]),
            "runtime": latency([{"latency_ns": values["runtime"]}]),
            "median_speedup": (
                statistics.median(values["ratio"]) if values["ratio"] else None
            ),
            "throughput_per_s": {
                side: statistics.median(windows)
                for side, windows in values.get("throughput", {}).items()
            },
            "runtime_minus_legacy_ci95": intervals(values),
        }
        for job, values in report.items()
    }
    calls_ms = {
        job: {
            side: [round(ns / 1e6, 3) for ns in values[side]]
            for side in ("legacy", "runtime")
        }
        for job, values in report.items()
    }
    record = {
        job: {
            **summary[job],
            "latency_ms": calls_ms[job],
            "windows_per_s": report[job].get("throughput", {}),
        }
        for job in summary
    }
    Path(args.out).write_text(json.dumps(record) + "\n", encoding="utf-8")
    print(json.dumps(summary))


def latency(lines: Iterable[dict[str, Any]]) -> dict[str, float]:
    values = [ns for line in lines for ns in line["latency_ns"]]
    return {"p50_ms": percentile(values, 0.5), "p95_ms": percentile(values, 0.95)}


def intervals(values: dict[str, Any], replicates: int = 2000) -> dict[str, list[float]]:
    """Paired-bootstrap 95 % intervals of runtime minus legacy.

    Latency resamples the alternating call pairs (p50 / p95 in ms); throughput
    resamples the rounds, whose two windows ran back to back (median req/s).
    """
    rng = random.Random(0)

    def bounds(draws: list[float]) -> list[float]:
        draws.sort()
        return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]]

    out: dict[str, list[float]] = {}
    legacy, runtime = values["legacy"], values["runtime"]
    if legacy:
        draws = {"p50_ms": [], "p95_ms": []}
        for _ in range(replicates):
            picks = rng.choices(range(len(legacy)), k=len(legacy))
            for name, q in (("p50_ms", 0.5), ("p95_ms", 0.95)):
                draws[name].append(
                    percentile([runtime[i] for i in picks], q)
                    - percentile([legacy[i] for i in picks], q)
                )
        out |= {name: bounds(found) for name, found in draws.items()}
    windows = values.get("throughput", {})
    rounds = list(
        zip(windows.get("legacy", []), windows.get("runtime", []), strict=True)
    )
    if len(rounds) > 1:
        found = []
        for _ in range(replicates):
            picks = rng.choices(rounds, k=len(rounds))
            found.append(
                statistics.median(r for _, r in picks)
                - statistics.median(lg for lg, _ in picks)
            )
        out["per_s"] = bounds(found)
    return out


def load_summary(load: dict[str, Any] | None) -> dict[str, float] | None:
    if load is None:
        return None
    return {
        "concurrency": load["concurrency"],
        "per_s": load["calls"] / load["seconds"],
        "p50_ms": percentile(load["latency_ns"], 0.5),
        "p95_ms": percentile(load["latency_ns"], 0.95),
    }


def inversions(old: list[float], new: list[float], tie: float) -> int:
    """Document pairs the runtime orders against legacy logits more than ``tie`` apart."""
    return sum(
        (old[i] - old[j]) * (new[i] - new[j]) < 0 and abs(old[i] - old[j]) > tie
        for i in range(len(old))
        for j in range(i + 1, len(old))
    )


def compare_job(
    job: str,
    legacy: dict[str, dict],
    runtime: dict[str, dict],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Values (cosine or logits and order), latency and load of one job."""
    import numpy as np

    errors = sorted(
        i for i in legacy if ("error" in legacy[i]) != ("error" in runtime[i])
    )
    shared = [i for i in legacy if "result" in legacy[i] and "result" in runtime[i]]
    mode = JOBS[job].mode
    if mode == "rerank":
        deltas, inverted = [], 0
        for i in shared:
            old = [float(v) for v in legacy[i]["result"]["Scores"]]
            new = runtime[i]["result"]
            deltas.append(float(np.abs(np.subtract(old, new)).max()))
            inverted += inversions(old, new, thresholds["tie"])
        values = {"max_logit_delta": max(deltas), "inversions_outside_ties": inverted}
        passed = inverted == 0
    else:
        cosines, deltas = [], []
        for i in shared:
            old = np.asarray(legacy[i]["result"], dtype=np.float64)
            new = np.asarray(runtime[i]["result"], dtype=np.float64)
            cosines.append(
                float(old @ new / (np.linalg.norm(old) * np.linalg.norm(new)))
            )
            deltas.append(float(np.abs(old - new).max()))
        values = {"min_cosine": min(cosines), "max_abs": max(deltas)}
        gated = thresholds["max_abs"] is not None and mode != "omni"
        passed = values["min_cosine"] >= thresholds["min_cosine"] and (
            not gated or values["max_abs"] <= thresholds["max_abs"]
        )
    old_latency = latency(legacy.values())
    new_latency = latency(runtime.values())
    return {
        "inputs": len(legacy),
        "error_mismatches": errors,
        **values,
        "parity_passed": bool(passed and not errors),
        "legacy": old_latency,
        "runtime": new_latency,
        "latency_passed": all(new_latency[k] <= old_latency[k] for k in new_latency),
    }


def run_compare(args: argparse.Namespace) -> None:
    legacy, legacy_loads = read_lines(args.legacy)
    runtime, runtime_loads = read_lines(args.runtime)
    thresholds = CPU_THRESHOLDS if args.device_class == "cpu" else ROCM_THRESHOLDS
    jobs: dict[str, Any] = {}
    for job in dict.fromkeys(job for job, _ in legacy):
        old = {i: line for (j, i), line in legacy.items() if j == job}
        new = {i: line for (j, i), line in runtime.items() if j == job}
        record = compare_job(job, old, new, thresholds)
        loads = (
            load_summary(legacy_loads.get(job)),
            load_summary(runtime_loads.get(job)),
        )
        if all(loads):
            record["load"] = {"legacy": loads[0], "runtime": loads[1]}
            record["throughput_passed"] = loads[1]["per_s"] >= loads[0]["per_s"]
        jobs[job] = record
    providers = dict.fromkeys(PROVIDERS[JOBS[job].mode] for job in jobs)
    result = {
        "legacy": "; ".join(LEGACY_PATHS[provider] for provider in providers),
        "device_class": args.device_class,
        "thresholds": thresholds,
        "jobs": jobs,
        "passed": all(
            r["parity_passed"]
            and r["latency_passed"]
            and r.get("throughput_passed", True)
            for r in jobs.values()
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                job: {k: v for k, v in r.items() if k.endswith("passed")}
                for job, r in jobs.items()
            }
        )
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    legacy = commands.add_parser("legacy", help="run the jobs on the legacy facade")
    legacy.add_argument("--tree", required=True)
    legacy.add_argument("--flat", default=None)
    build = commands.add_parser("build", help="compile the legacy test binary for ab")
    build.add_argument("--tree", required=True)
    build.add_argument("--flat", default=None)
    runtime = commands.add_parser("runtime", help="run the jobs on the runtime")
    serve = commands.add_parser(
        "serve", help="serve the jobs on the runtime as ab's baseline side"
    )
    ab = commands.add_parser("ab", help="alternate legacy and runtime calls")
    ab.add_argument("--binary", default=None)
    ab.add_argument("--tree", default=None)
    ab.add_argument(
        "--baseline-engine",
        choices=("native", "onnxruntime"),
        default=None,
        help="instead of the legacy binary, the runtime with Omni on this engine",
    )
    ab.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="an environment variable for the baseline runtime process only",
    )
    ab.add_argument(
        "--baseline-runtime",
        default=None,
        metavar="DIR",
        help="the baseline runtime's source (another tree's src/model-runtime)",
    )
    ab.add_argument("--rounds", type=int, default=4)
    ab.add_argument("--legacy-cpus", type=int, default=None)
    ab.add_argument(
        "--gap-ms",
        type=float,
        default=0.0,
        help="pause before every call and load window (the other side's pools settle)",
    )
    for sub in (runtime, serve, ab):
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--profile", default="exact")
        sub.add_argument("--threads", type=int, default=None)
        sub.add_argument(
            "--omni-engine",
            choices=("native", "onnxruntime"),
            default="native",
            help="Omni's published snapshot on native, or its prepared bundle on onnxruntime",
        )
        sub.add_argument(
            "--runtime-src",
            default=None,
            metavar="DIR",
            help="import vllm_srun from this src/model-runtime instead of this checkout's",
        )
    for sub in (legacy, build, runtime, serve, ab):
        sub.add_argument("--cache", required=True)
        sub.add_argument(
            "--prepared",
            required=True,
            help="the directory of the prepared Omni bundles (their goldens are the Omni inputs)",
        )
        sub.add_argument("--out", required=True)
        sub.add_argument("--jobs", default="")
        sub.add_argument("--limit", type=int, default=0)
        sub.add_argument("--repeats", type=int, default=5)
        sub.add_argument("--concurrency", type=int, default=0)
        sub.add_argument("--seconds", type=float, default=20.0)
    compare = commands.add_parser("compare", help="write the comparison record")
    compare.add_argument("--legacy", required=True)
    compare.add_argument("--runtime", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--device-class", choices=("cpu", "rocm"), default="cpu")
    args = parser.parse_args(argv)
    if (
        args.command == "ab"
        and not args.baseline_engine
        and not (args.binary and args.tree)
    ):
        parser.error("ab needs --binary and --tree, or --baseline-engine")
    commands_by_name = {
        "legacy": run_legacy,
        "build": build_legacy,
        "runtime": run_runtime,
        "serve": run_serve,
        "ab": run_ab,
        "compare": run_compare,
    }
    commands_by_name[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
