"""Legacy parity driver: the same inputs through the legacy router path and the runtime.

The legacy side runs the router's native facade (``pkg/modelruntime/native``
at the legacy commit, the code behind ``/api/v1/diagnostics/models/*``) in a Go
test written into a copy of an exact mirror of that commit, with the prepared
bindings the router builds (contract, adapter, deployment budget, windows,
operating point). The reference is exactly what the router served, Vela
Halu's grounded path included (it has no diagnostics route). The runtime side
serves the same packages in-process and answers through its surface API.

  corpus   write the input set: the repository's E2E prompts, plus long and
           grounded inputs derived from them deterministically
  legacy   run the jobs on the legacy facade; write results and latencies
  runtime  run the jobs on the runtime; write results and latencies
  build-legacy  compile the legacy side once, as a test binary ``ab`` serves
  ab       alternate legacy and runtime calls on the same cores, round after
           round; every finished round is saved, so a run resumes or grows
  compare  check design section 17's thresholds; write the parity record

Results are JSON lines ``{job, id, result | error, latency_ns}``; offsets
are converted to code points on the legacy side of ``compare`` (the bindings
report UTF-8 bytes).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from http import HTTPStatus
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
TESTDATA = REPO / "e2e" / "testcases" / "testdata"
GO_TEST = "zz_legacy_parity_dump_test.go"
PARTIAL = "provider truncated its input"
CPU_THRESHOLDS = {"probability": 1e-3, "near_tie": 1e-3, "label": 1.0, "spans": 0.995}
ROCM_THRESHOLDS = {"probability": 0.02, "near_tie": 1e-3, "label": 0.995, "spans": 0.98}
# A reduced copy (max_speed) against exact: 99% agreement, near ties counted, |Δp| reported.
REDUCED_THRESHOLDS = {
    "probability": math.inf,
    "near_tie": 0.0,
    "label": 0.99,
    "spans": 0.99,
}

# job: (repo, mode, deployment max_tokens, overflow, window, (provider, device, graph head))
Job = tuple[str, str, int, str, tuple[int, int] | None, tuple[str, str, str]]
CANDLE = ("candle", "cpu", "")
MIGRAPHX = ("ort", "migraphx:0", "")
# The router's CPU defaults (candle).
CPU_JOBS: dict[str, Job] = {
    "domain": ("Domain", "sequence", 512, "truncate", None, CANDLE),
    "guard": ("Guard", "sequence_windows", 8192, "window", (512, 255), CANDLE),
    "safety": ("Safety", "sequence", 512, "truncate", None, CANDLE),
    "shield": ("Shield", "sequence", 512, "truncate", None, CANDLE),
    "factcheck": ("FactCheck", "sequence", 512, "truncate", None, CANDLE),
    "feedback": ("Feedback", "sequence", 512, "truncate", None, CANDLE),
    "modality": ("Modality", "sequence", 512, "truncate", None, CANDLE),
    "hazard": ("Hazard", "operating_point", 32768, "reject", None, CANDLE),
    "pii": ("PII", "token_windows", 32768, "window", (512, 255), CANDLE),
    "pii_truncate": ("PII", "tokens", 512, "truncate", None, CANDLE),
    "halu": ("Halu", "grounded", 8192, "truncate", None, CANDLE),
}
# config/recipes/vela-amd: ORT on MIGraphX (Guard on the ROCm EP with its 8K graph), fixed 8K sessions.
AMD_JOBS: dict[str, Job] = {
    "domain": ("Domain", "sequence", 8192, "reject", None, MIGRAPHX),
    "guard": (
        "Guard",
        "sequence",
        8192,
        "reject",
        None,
        ("ort", "rocm:0", "onnx/model_rocm_8k.onnx"),
    ),
    "safety": ("Safety", "sequence", 8192, "reject", None, MIGRAPHX),
    "factcheck": ("FactCheck", "sequence", 8192, "reject", None, MIGRAPHX),
    "feedback": ("Feedback", "sequence", 8192, "reject", None, MIGRAPHX),
    "modality": ("Modality", "sequence", 8192, "reject", None, MIGRAPHX),
    "hazard": ("Hazard", "operating_point", 32768, "reject", None, MIGRAPHX),
    "pii": ("PII", "tokens", 8192, "reject", None, MIGRAPHX),
    # Not in the recipe: the package's ONNX graph answers differently from its
    # checkpoint (which candle and the runtime serve), so this job times ORT only.
    "shield": ("Shield", "sequence", 8192, "reject", None, MIGRAPHX),
}
RECIPES = {"cpu": CPU_JOBS, "amd": AMD_JOBS}
REVISIONS = {
    "Domain": "f6354f54adcf38770f635ad903be2b00577f6c11",
    "Guard": "087f9e401012df839c83717b746967ac7aebfa3e",
    "Safety": "6e70e725a5f4d86da10f5be5e4dfd1da0358bb85",
    "Shield": "a981a99eeb05a2859b88b5cee9af4352897ec4ec",
    "FactCheck": "99ede1aba1563e59e416f744d25b3f6b7e9d8274",
    "Feedback": "47434a7fd7c245c0c7c17564a000b3c56ccfec41",
    "Modality": "5384b8997e3cbb79ca3a670e869577f4e4f4997e",
    "Hazard": "5dd25f2cc3c98f338e6a79b667662d60f936a28d",
    "PII": "6d3300c4bd7975f30a664503f6c725cf1fbbad48",
    "Halu": "ca87531211e414ac21c641b2faa8b8e21619de8f",
}
KIND = {
    "sequence": "sequence",
    "sequence_windows": "sequence",
    "operating_point": "scores",
    "tokens": "token",
    "token_windows": "token",
    "grounded": "grounded",
}


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def prompts() -> list[str]:
    """Every distinct user prompt in the repository's E2E test data, in file order."""
    seen: dict[str, None] = {}
    keys = (
        "question",
        "query",
        "original_question",
        "contradiction",
        "paraphrase",
        "context",
    )
    for path in sorted(TESTDATA.glob("*.json")):
        stack: list[Any] = [json.loads(path.read_text(encoding="utf-8"))]
        while stack:
            node = stack.pop(0)
            if isinstance(node, list):
                stack[:0] = node
            elif isinstance(node, dict):
                for key in keys:
                    if isinstance(node.get(key), str) and node[key].strip():
                        seen.setdefault(node[key].strip())
                stack[:0] = [v for k, v in node.items() if isinstance(v, list | dict)]
    return list(seen)


EDGE = [
    "Ünïcödé, 中文, العربية, हिन्दी, 🚀 and emoji 👩‍💻 in one line.",
    "Contact John Doe at john.doe@example.com or +1 (415) 555-0100; SSN 123-45-6789.",
    "<bos> literal special tokens <eos> stay content [SEP] [CLS]",
    "x",
    "   leading and trailing whitespace   ",
    "Meine Telefonnummer ist 030 1234567 und ich wohne in der Hauptstraße 5, 10115 Berlin.",
]


def corpus(seed: int = 0) -> list[dict[str, Any]]:
    """Text inputs (short prompts, edge cases, long documents) and grounded triples."""
    base = prompts()
    rng = random.Random(seed)
    items = [{"id": f"p{i:04d}", "text": text} for i, text in enumerate(base)]
    items += [{"id": f"e{i:02d}", "text": text} for i, text in enumerate(EDGE)]
    for index, count in enumerate([8, 16, 32, 48, 64, 96, 128, 192, 256, 384]):
        picked = rng.sample(base, k=min(count, len(base)))
        items.append({"id": f"l{index:02d}", "text": "\n".join(picked)})
    for index in range(60):
        context = " ".join(rng.sample(base, k=rng.randint(3, 40)))
        question = rng.choice(base)
        answer = " ".join(rng.sample(base, k=rng.randint(1, 4)))
        items.append(
            {
                "id": f"g{index:02d}",
                "context": context,
                "question": question,
                "answer": answer,
            }
        )
    for index in range(4):
        context = " ".join(rng.sample(base, k=min(400, len(base))))
        items.append(
            {
                "id": f"gl{index}",
                "context": context,
                "question": rng.choice(base),
                "answer": rng.choice(base),
            }
        )
    return items


def inputs_for(mode: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grounded = mode == "grounded"
    return [item for item in items if ("answer" in item) == grounded]


# ---------------------------------------------------------------------------
# Legacy (Go, in an exact mirror of the legacy commit)
# ---------------------------------------------------------------------------

GO_TEMPLATE = r"""//go:build !windows && cgo

package native

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type parityJob struct {
	Job, Path, Revision, Mode, Overflow string
	Provider, Device, Head, CacheDir    string
	MaxTokens, WindowSize, WindowOverlap  int
	Labels                                []string
	Inputs                                []map[string]string
	Repeats, Concurrency                  int
	Seconds                               float64
}

type parityLine struct {
	Job       string          `json:"job"`
	ID        string          `json:"id"`
	Result    json.RawMessage `json:"result,omitempty"`
	Error     string          `json:"error,omitempty"`
	LatencyNS []int64         `json:"latency_ns,omitempty"`
}

type parityThroughput struct {
	Job         string  `json:"job"`
	Concurrency int     `json:"concurrency"`
	Calls       int     `json:"calls"`
	Seconds     float64 `json:"seconds"`
	LatencyNS   []int64 `json:"latency_ns"`
}

func TestLegacyParityDump(t *testing.T) {
	raw, err := os.ReadFile(os.Getenv("LEGACY_PARITY_JOBS"))
	if err != nil {
		t.Fatal(err)
	}
	var jobs []parityJob
	if err := json.Unmarshal(raw, &jobs); err != nil {
		t.Fatal(err)
	}
	out, err := os.Create(os.Getenv("LEGACY_PARITY_OUT"))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	encoder := json.NewEncoder(out)
	runtime := New(nil)
	ctx := context.Background()
	for _, job := range jobs {
		call, closeTask := parityTask(t, runtime, job)
		for range 3 {
			_, _ = call(ctx, job.Inputs[0])
		}
		for _, input := range job.Inputs {
			line := parityLine{Job: job.Job, ID: input["id"]}
			for repeat := 0; repeat < max(job.Repeats, 1); repeat++ {
				start := time.Now()
				result, callErr := call(ctx, input)
				line.LatencyNS = append(line.LatencyNS, time.Since(start).Nanoseconds())
				if repeat > 0 {
					continue
				}
				if callErr != nil {
					line.Error = callErr.Error()
				}
				if callErr == nil || errors.Is(callErr, tasks.ErrTokenSpansTruncated) {
					if line.Result, err = json.Marshal(result); err != nil {
						t.Fatal(err)
					}
				}
			}
			if err := encoder.Encode(line); err != nil {
				t.Fatal(err)
			}
		}
		if job.Concurrency > 0 {
			if err := encoder.Encode(parityLoad(ctx, job, call)); err != nil {
				t.Fatal(err)
			}
		}
		closeTask()
	}
}

// TestLegacyParityServe answers stdin commands so a driver can alternate legacy and runtime
// work: "call<TAB>job<TAB>id" prints the call's latency in nanoseconds (-1 on error) and
// "load<TAB>job<TAB>concurrency<TAB>seconds" a load window's parityThroughput as JSON.
func TestLegacyParityServe(t *testing.T) {
	raw, err := os.ReadFile(os.Getenv("LEGACY_PARITY_JOBS"))
	if err != nil {
		t.Fatal(err)
	}
	var jobs []parityJob
	if err := json.Unmarshal(raw, &jobs); err != nil {
		t.Fatal(err)
	}
	runtime := New(nil)
	ctx := context.Background()
	calls := map[string]func(context.Context, map[string]string) (any, error){}
	byName := map[string]parityJob{}
	inputs := map[string]map[string]string{}
	for _, job := range jobs {
		call, closeTask := parityTask(t, runtime, job)
		defer closeTask()
		calls[job.Job] = call
		byName[job.Job] = job
		for _, input := range job.Inputs {
			inputs[job.Job+"\t"+input["id"]] = input
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
			report, _ := json.Marshal(parityLoad(ctx, job, calls[job.Job]))
			fmt.Println(string(report))
			continue
		}
		start := time.Now()
		_, callErr := calls[fields[1]](ctx, inputs[fields[1]+"\t"+fields[2]])
		latency := time.Since(start).Nanoseconds()
		if callErr != nil && !errors.Is(callErr, tasks.ErrTokenSpansTruncated) {
			latency = -1
		}
		fmt.Println(latency)
	}
}

func parityLoad(ctx context.Context, job parityJob, call func(context.Context, map[string]string) (any, error)) parityThroughput {
	deadline := time.Now().Add(time.Duration(job.Seconds * float64(time.Second)))
	var mutex sync.Mutex
	var wait sync.WaitGroup
	report := parityThroughput{Job: job.Job, Concurrency: job.Concurrency}
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

func paritySpec(job parityJob, contract string) config.ResolvedModelBinding {
	adapter := "modernbert"
	if job.Mode == "grounded" {
		adapter = "vela_halu"
	}
	return config.ResolvedModelBinding{
		Recipe: "parity", Name: job.Job,
		Binding: config.ModelBinding{Deployment: job.Job, Adapter: adapter, Contract: contract, Head: job.Head},
		Deployment: config.ModelDeployment{
			Artifact: job.Path, Revision: job.Revision, Provider: job.Provider, Device: job.Device, Precision: "native",
			CompilationCacheDir: job.CacheDir,
			Input:               config.ModelInputBudget{MaxTokens: job.MaxTokens, Overflow: job.Overflow},
		},
	}
}

func parityTask(t *testing.T, runtime *Runtime, job parityJob) (func(context.Context, map[string]string) (any, error), func()) {
	ctx := context.Background()
	window := tasks.TextWindowsRequest{Size: job.WindowSize, Overlap: job.WindowOverlap}
	switch job.Mode {
	case "sequence":
		task, err := runtime.Sequence(ctx, paritySpec(job, config.RemoteClassifierContractLabelDistribution))
		if err != nil {
			t.Fatal(err)
		}
		return func(ctx context.Context, in map[string]string) (any, error) { return task.Call(ctx, "parity", in["text"]) }, func() { _ = task.Close() }
	case "sequence_windows":
		task, err := runtime.SequenceWindows(ctx, paritySpec(job, config.RemoteClassifierContractLabelDistribution), window)
		if err != nil {
			t.Fatal(err)
		}
		return func(ctx context.Context, in map[string]string) (any, error) {
			request := window
			request.Text = in["text"]
			return task.Call(ctx, "parity", request)
		}, func() { _ = task.Close() }
	case "tokens":
		task, err := runtime.Tokens(ctx, paritySpec(job, config.RemoteClassifierContractTokenSpans))
		if err != nil {
			t.Fatal(err)
		}
		return func(ctx context.Context, in map[string]string) (any, error) { return task.Call(ctx, "parity", in["text"]) }, func() { _ = task.Close() }
	case "token_windows":
		task, err := runtime.TokenWindows(ctx, paritySpec(job, config.RemoteClassifierContractTokenSpans), window)
		if err != nil {
			t.Fatal(err)
		}
		return func(ctx context.Context, in map[string]string) (any, error) {
			request := window
			request.Text = in["text"]
			return task.Call(ctx, "parity", request)
		}, func() { _ = task.Close() }
	case "operating_point":
		spec := paritySpec(job, config.RemoteClassifierContractLabelScores)
		data, err := os.ReadFile(filepath.Join(job.Path, "operating_point.json"))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		spec.Binding.OperatingPoint = &config.OperatingPointReference{Path: "operating_point.json", SHA256: hex.EncodeToString(sum[:])}
		task, err := runtime.OperatingPoint(ctx, spec, job.Labels)
		if err != nil {
			t.Fatal(err)
		}
		return func(ctx context.Context, in map[string]string) (any, error) { return task.Score(ctx, "parity", in["text"]) }, func() { _ = task.Close() }
	case "grounded":
		task, err := runtime.Grounded(ctx, paritySpec(job, config.RemoteClassifierContractTokenSpans), 0.5)
		if err != nil {
			t.Fatal(err)
		}
		return func(ctx context.Context, in map[string]string) (any, error) {
			return task.Call(ctx, "parity", tasks.GroundedTextRequest{Context: in["context"], Question: in["question"], Answer: in["answer"]})
		}, func() { _ = task.Close() }
	}
	t.Fatalf("unknown mode %q", job.Mode)
	return nil, nil
}
"""


def snapshot(cache: Path, name: str) -> Path:
    path = (
        cache
        / f"models--vllm-sr--Vela-1.0-Encoder-307M-{name}"
        / "snapshots"
        / REVISIONS[name]
    )
    if not (path / "config.json").is_file():
        raise SystemExit(f"{name}@{REVISIONS[name]} is not in {cache}")
    return path


def flat_copy(source: Path, target: Path) -> Path:
    """The snapshot as plain files (hard links to its blobs), as the router downloads models."""
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        destination = target / path.relative_to(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path.resolve(), destination)
    return target


def job_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Each job's spec; with ``--inputs`` (an earlier run's jobs file) its inputs exactly.

    The corpus reads the tree's e2e testdata, which changes over time, so a
    runtime run compared with an earlier legacy run takes that run's inputs,
    and without ``--jobs`` the recipe's jobs that run had.
    """
    items = corpus(args.seed)
    recorded = {}
    if getattr(args, "inputs", None):
        recorded = {
            spec["Job"]: spec["Inputs"]
            for spec in json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        }
    table = RECIPES[args.recipe]
    if args.jobs:
        jobs = args.jobs.split(",")
    else:
        jobs = [job for job in table if not recorded or job in recorded]
    specs = []
    for job in jobs:
        name, mode, max_tokens, overflow, window, execution = table[job]
        path = snapshot(Path(args.cache), name)
        if getattr(args, "flat", None):
            path = flat_copy(path, Path(args.flat) / f"{name}-{REVISIONS[name][:12]}")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
        if recorded and job not in recorded:
            raise SystemExit(f"{args.inputs} has no job {job!r}")
        selected = recorded[job] if recorded else inputs_for(mode, items)
        if args.limit:
            selected = selected[: args.limit]
        specs.append(
            {
                "Job": job,
                "Repo": f"vllm-sr/Vela-1.0-Encoder-307M-{name}",
                "Path": str(path),
                "Revision": REVISIONS[name],
                "Mode": mode,
                "Provider": execution[0],
                "Device": execution[1],
                "Head": execution[2],
                "CacheDir": (
                    (getattr(args, "compile_cache", "") or "")
                    if execution[1].startswith("migraphx:")
                    else ""
                ),
                "Overflow": overflow,
                "MaxTokens": max_tokens,
                "WindowSize": window[0] if window else 0,
                "WindowOverlap": window[1] if window else 0,
                "Labels": labels,
                "Inputs": [{k: str(v) for k, v in item.items()} for item in selected],
                "Repeats": args.repeats,
                "Concurrency": args.concurrency,
                "Seconds": args.seconds,
            }
        )
    return specs


def legacy_env(args: argparse.Namespace, jobs: Path) -> dict[str, str]:
    """The cgo and loader environment of the legacy tree's bindings, plus the jobs file."""
    tree = Path(args.tree)
    libraries = [Path(p) for p in args.libs.split(":")] if args.libs else []
    libraries += [
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
        "LEGACY_PARITY_JOBS": str(jobs),
    }


def write_legacy_test(args: argparse.Namespace, jobs: Path) -> Path:
    """The dump test in the legacy tree and the jobs file; returns the router module."""
    tree = Path(args.tree)
    package = tree / "src" / "semantic-router" / "pkg" / "modelruntime" / "native"
    (package / GO_TEST).write_text(GO_TEMPLATE, encoding="utf-8")
    jobs.write_text(json.dumps(job_specs(args)), encoding="utf-8")
    return tree / "src" / "semantic-router"


def run_legacy(args: argparse.Namespace) -> None:
    """Run every job's inputs on the legacy facade and write results and latencies."""
    jobs = Path(args.out).with_suffix(".jobs.json")
    module = write_legacy_test(args, jobs)
    env = {**legacy_env(args, jobs), "LEGACY_PARITY_OUT": str(Path(args.out).resolve())}
    command = [
        "go",
        "test",
        "-count=1",
        "-timeout",
        "6h",
        "-run",
        "^TestLegacyParityDump$",
    ]
    subprocess.run(
        [*command, "./pkg/modelruntime/native/"], cwd=module, env=env, check=True
    )


def build_legacy(args: argparse.Namespace) -> None:
    """Compile the legacy facade's test binary (serve mode for ``ab``) next to its jobs file."""
    jobs = Path(args.out).with_suffix(".jobs.json")
    module = write_legacy_test(args, jobs)
    command = [
        "go",
        "test",
        "-c",
        "-o",
        str(Path(args.out).resolve()),
        "./pkg/modelruntime/native/",
    ]
    subprocess.run(command, cwd=module, env=legacy_env(args, jobs), check=True)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def runtime_body(spec: dict[str, Any], item: dict[str, str]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "max_tokens": spec["MaxTokens"],
        "overflow": spec["Overflow"],
    }
    if spec["WindowSize"]:
        options["window"] = {
            "tokens": spec["WindowSize"],
            "overlap": spec["WindowOverlap"],
        }
    if spec["Mode"] == "operating_point":
        options = {}
    if spec["Mode"] == "grounded":
        value: Any = {k: item[k] for k in ("context", "question", "answer")}
    else:
        value = item["text"]
    return {"model": spec["Job"], "input": [value], "options": options}


async def runtime_call(runtime: Any, body: dict[str, Any]) -> tuple[int, Any]:
    """One request as the HTTP server makes it: with its encoded size (small ones plan inline)."""
    return await runtime.call("classify", body, len(json.dumps(body)))


def with_profile(body: dict[str, Any], profile: str) -> dict[str, Any]:
    return {**body, "options": {**body["options"], "profile": profile}}


def serve_runtime(
    args: argparse.Namespace, specs: list[dict[str, Any]], device: str
) -> Any:
    """A started in-process runtime serving every job's model, result cache off."""
    sys.path.insert(0, str(REPO / "src" / "model-runtime"))
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    models = tuple(
        ModelConfig(
            model=spec["Repo"], name=spec["Job"], device=device, profile=args.profile
        )
        for spec in specs
    )
    runtime = Runtime(
        ServeConfig(
            models=models,
            threads=args.threads,
            result_cache_entries=0,
            cache_dir=args.cache,
            offline=True,
        )
    )
    runtime.start(background=False)
    return runtime


def run_runtime(args: argparse.Namespace) -> None:
    specs = job_specs(args)
    runtime = serve_runtime(args, specs, args.device)

    async def call(body: dict[str, Any]) -> dict[str, Any]:
        status, out = await runtime_call(runtime, body)
        if status != HTTPStatus.OK:
            raise RuntimeError(json.dumps(out))
        return out

    async def measure(stream: Any) -> None:
        for spec in specs:
            for _ in range(3):
                await call(runtime_body(spec, spec["Inputs"][0]))
            for item in spec["Inputs"]:
                line: dict[str, Any] = {
                    "job": spec["Job"],
                    "id": item["id"],
                    "latency_ns": [],
                }
                for repeat in range(max(args.repeats, 1)):
                    start = time.perf_counter_ns()
                    out = await call(runtime_body(spec, item))
                    line["latency_ns"].append(time.perf_counter_ns() - start)
                    if repeat == 0:
                        line["result"] = out["results"][0]
                stream.write(json.dumps(line) + "\n")
            if args.concurrency:
                load = await runtime_load(runtime, spec, args)
                stream.write(json.dumps(load) + "\n")

    with open(args.out, "w", encoding="utf-8") as stream:
        asyncio.run(measure(stream))
    runtime.stop()


async def runtime_load(
    runtime: Any,
    spec: dict[str, Any],
    args: argparse.Namespace,
    profile: str | None = None,
) -> dict[str, Any]:
    """``args.concurrency`` callers for ``args.seconds``: concurrent requests on one event loop."""
    deadline = time.monotonic() + args.seconds
    latencies: list[int] = []

    async def caller(index: int) -> None:
        position = index
        while time.monotonic() < deadline:
            body = runtime_body(spec, spec["Inputs"][position % len(spec["Inputs"])])
            start = time.perf_counter_ns()
            await runtime_call(
                runtime, with_profile(body, profile) if profile else body
            )
            latencies.append(time.perf_counter_ns() - start)
            position += args.concurrency

    began = time.monotonic()
    await asyncio.gather(*(caller(index) for index in range(args.concurrency)))
    return {
        "job": spec["Job"],
        "concurrency": args.concurrency,
        "calls": len(latencies),
        "seconds": time.monotonic() - began,
        "latency_ns": latencies,
    }


def bootstrap_ci(
    pairs: list[tuple[int, int]],
    statistic: Callable[[list[int], list[int]], float],
    resamples: int = 2000,
    seed: int = 0,
) -> list[float]:
    """The 95% interval of ``statistic(legacy, runtime)`` over pairs resampled with replacement."""
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        values.append(statistic([a for a, _ in sample], [b for _, b in sample]))
    values.sort()
    return [values[int(0.025 * resamples)], values[int(0.975 * resamples) - 1]]


def ab_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Each job's latencies, paired speedups with 95% intervals, and load-window rates.

    The intervals resample the interleaved pairs (each input's legacy and
    runtime call), so they hold the inputs fixed across both sides.
    """
    summary = {}
    for job, values in report.items():
        pairs = list(zip(values["legacy"], values["runtime"], strict=True))
        windows = values.get("throughput", {})
        row: dict[str, Any] = {
            "pairs": len(pairs),
            "legacy_ms": {
                "p50": percentile(values["legacy"], 0.5),
                "p95": percentile(values["legacy"], 0.95),
            },
            "runtime_ms": {
                "p50": percentile(values["runtime"], 0.5),
                "p95": percentile(values["runtime"], 0.95),
            },
            "median_speedup": (
                statistics.median(values["ratio"]) if values["ratio"] else math.nan
            ),
            "throughput_per_s": {
                side: statistics.median(rates) for side, rates in windows.items()
            },
            "throughput_windows_per_s": windows,
        }
        if pairs:
            row["ci95"] = {
                "median_speedup": bootstrap_ci(
                    pairs,
                    lambda a, b: statistics.median(
                        x / y for x, y in zip(a, b, strict=True)
                    ),
                ),
                "p50_ratio": bootstrap_ci(
                    pairs, lambda a, b: percentile(a, 0.5) / percentile(b, 0.5)
                ),
                "p95_ratio": bootstrap_ci(
                    pairs, lambda a, b: percentile(a, 0.95) / percentile(b, 0.95)
                ),
            }
        summary[job] = row
    return summary


async def ab_rounds(
    state: dict[str, Any],
    specs: list[dict[str, Any]],
    rounds: int,
    pair: Callable[[dict[str, Any], dict[str, str], bool], Awaitable[tuple[int, int]]],
    sides: list[str],
    window: Callable[[str, dict[str, Any]], Awaitable[float]] | None,
    save: Callable[[], None],
) -> None:
    """Run the rounds up to ``rounds`` that ``state`` has not finished, saving it after each.

    ``state`` holds the finished pair and window rounds and their report, so an
    interrupted run resumes where it stopped and a larger ``rounds`` adds
    rounds to an earlier run, continuing its rotation. A pair round calls
    ``pair(spec, item, runtime_first)`` for every input, with the order
    flipping each round; a window round rotates ``sides`` by one each round
    and records ``window(side, spec)``'s calls per second. Without ``window``
    only pair rounds run.
    """
    report = state["report"]
    for round_index in range(state["pair_rounds"], rounds):
        for spec in specs:
            entry = report.setdefault(
                spec["Job"], {"legacy": [], "runtime": [], "ratio": []}
            )
            for item in spec["Inputs"]:
                legacy_ns, runtime_ns = await pair(spec, item, bool(round_index % 2))
                if legacy_ns < 0 or runtime_ns < 0:
                    continue
                entry["legacy"].append(legacy_ns)
                entry["runtime"].append(runtime_ns)
                entry["ratio"].append(legacy_ns / runtime_ns)
        state["pair_rounds"] = round_index + 1
        save()
    for round_index in range(state["window_rounds"], rounds if window else 0):
        shift = round_index % len(sides)
        for spec in specs:
            rates = report[spec["Job"]].setdefault("throughput", {})
            for side in sides[shift:] + sides[:shift]:
                rates.setdefault(side, []).append(await window(side, spec))
        state["window_rounds"] = round_index + 1
        save()


def run_ab(args: argparse.Namespace) -> None:
    """Alternate legacy and runtime work on the same cores, round after round.

    Both sides then see the same moment's contention on a shared host. Per
    input, one legacy call and one runtime ``exact`` call alternate (the order
    flips every round); with ``--concurrency``, load windows of the legacy
    facade, the runtime's ``exact`` profile and the models' profile rotate.
    The legacy side is the ``build-legacy`` binary in serve mode, the runtime
    side this process. Every finished round goes to ``<out>.raw.json``, which
    ``--resume`` continues.
    """
    raw = Path(args.out).with_suffix(".raw.json")
    state = (
        json.loads(Path(args.resume).read_text(encoding="utf-8"))
        if args.resume
        else {"pair_rounds": 0, "window_rounds": 0, "report": {}}
    )
    jobs = Path(args.binary).with_suffix(".jobs.json")
    specs = json.loads(jobs.read_text(encoding="utf-8"))
    legacy = subprocess.Popen(
        [args.binary, "-test.run", "^TestLegacyParityServe$", "-test.timeout", "12h"],
        env=legacy_env(args, jobs),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert legacy.stdin is not None and legacy.stdout is not None
    while legacy.stdout.readline().strip() != "READY":
        if legacy.poll() is not None:
            raise SystemExit("the legacy binary exited before it was ready")
    runtime = serve_runtime(args, specs, "cpu")
    sides = ["legacy", "exact", *([args.profile] if args.profile != "exact" else [])]

    def call_legacy(spec: dict[str, Any], item: dict[str, str]) -> int:
        legacy.stdin.write(f"call\t{spec['Job']}\t{item['id']}\n")
        return int(legacy.stdout.readline())

    async def call_runtime(spec: dict[str, Any], item: dict[str, str]) -> int:
        body = with_profile(runtime_body(spec, item), "exact")
        start = time.perf_counter_ns()
        status, _ = await runtime_call(runtime, body)
        elapsed = time.perf_counter_ns() - start
        return elapsed if status == HTTPStatus.OK else -1

    async def pair(
        spec: dict[str, Any], item: dict[str, str], runtime_first: bool
    ) -> tuple[int, int]:
        if runtime_first:
            runtime_ns = await call_runtime(spec, item)
            return call_legacy(spec, item), runtime_ns
        legacy_ns = call_legacy(spec, item)
        return legacy_ns, await call_runtime(spec, item)

    async def window(side: str, spec: dict[str, Any]) -> float:
        if side != "legacy":
            load = await runtime_load(runtime, spec, args, side)
        else:
            legacy.stdin.write(
                f"load\t{spec['Job']}\t{args.concurrency}\t{args.seconds}\n"
            )
            load = json.loads(legacy.stdout.readline())
        return load["calls"] / load["seconds"]

    def save() -> None:
        raw.write_text(json.dumps(state) + "\n", encoding="utf-8")

    async def measure() -> None:
        for spec in specs:
            for _ in range(3):
                await call_runtime(spec, spec["Inputs"][0])
        await ab_rounds(
            state,
            specs,
            args.rounds,
            pair,
            sides,
            window if args.concurrency else None,
            save,
        )

    asyncio.run(measure())
    legacy.stdin.close()
    legacy.wait()
    runtime.stop()
    summary = ab_summary(state["report"])
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for job, row in summary.items():
        rates = " ".join(
            f"{side} {rate:.1f}/s" for side, rate in row["throughput_per_s"].items()
        )
        print(
            f"{job:13s} pairs={row['pairs']:4d} p50 {row['legacy_ms']['p50']:.2f}->{row['runtime_ms']['p50']:.2f} ms "
            f"p95 {row['legacy_ms']['p95']:.2f}->{row['runtime_ms']['p95']:.2f} ms speedup x{row['median_speedup']:.2f} {rates}"
        )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def code_points(text: str, offset: int) -> int:
    return len(text.encode("utf-8")[:offset].decode("utf-8", errors="strict"))


def legacy_values(
    spec: dict[str, Any], item: dict[str, str], result: dict[str, Any]
) -> dict[str, Any]:
    """The legacy result in the runtime's terms: values, label and code-point spans."""
    mode = spec["Mode"]
    if mode == "sequence":
        return {"values": result["Probabilities"]}
    if mode == "sequence_windows":
        windows = result["Windows"]
        return {
            "values": [
                max(w["Probabilities"][i] for w in windows)
                for i in range(len(windows[0]["Probabilities"]))
            ],
            "windows": [[w["Start"], w["End"]] for w in windows],
        }
    if mode == "operating_point":
        return {"values": result["Scores"], "windows": result.get("Windows")}
    if mode == "token_windows":
        entities = result["Result"]["Entities"] or []
    else:
        entities = result["Entities"] or []
    text = item["answer"] if mode == "grounded" else item["text"]
    spans = [
        (
            e["EntityType"].upper() if mode != "grounded" else "HALLUCINATED",
            code_points(text, e["Start"]),
            code_points(text, e["End"]),
            e["Confidence"],
        )
        for e in entities
    ]
    return {"spans": spans}


def runtime_values(spec: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    mode = spec["Mode"]
    if mode in ("sequence", "sequence_windows"):
        out = {"values": result["probabilities"]}
        if "windows" in result:
            out["windows"] = [[w["start"], w["end"]] for w in result["windows"]]
        return out
    if mode == "operating_point":
        return {
            "values": result["scores"],
            "windows": [[w["start"], w["end"]] for w in result.get("windows", [])],
        }
    label = (
        (lambda s: "HALLUCINATED")
        if mode == "grounded"
        else (lambda s: s["label"].upper())
    )
    return {
        "spans": [
            (label(s), s["start"], s["end"], s["probability"]) for s in result["spans"]
        ]
    }


def read_lines(
    path: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    results, loads = {}, {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if "concurrency" in value:
                loads[value["job"]] = value
            else:
                results[(value["job"], value["id"])] = value
    return results, loads


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))] / 1e6


def compare_job(
    spec: dict[str, Any],
    legacy: dict,
    runtime: dict,
    thresholds: dict[str, float],
    baseline: Callable[..., dict[str, Any]] = legacy_values,
) -> dict[str, Any]:
    """One job's agreement and latency; ``baseline`` reads the ``legacy`` side's results."""
    inputs = {item["id"]: item for item in spec["Inputs"]}
    kind = KIND[spec["Mode"]]
    disagreements: list[dict[str, Any]] = []
    max_delta, compared, agreed, near_ties, both_errors = 0.0, 0, 0, 0, 0
    latency_legacy, latency_runtime = [], []
    for item_id, item in inputs.items():
        left, right = legacy.get((spec["Job"], item_id)), runtime.get(
            (spec["Job"], item_id)
        )
        if left is None or right is None:
            disagreements.append({"id": item_id, "reason": "missing"})
            continue
        latency_legacy += left.get("latency_ns", [])[1:] or left.get("latency_ns", [])
        latency_runtime += right.get("latency_ns", [])[1:] or right.get(
            "latency_ns", []
        )
        legacy_error = left.get("error") or (left.get("result") or {}).get("error")
        runtime_error = right.get("result", {}).get("error")
        if legacy_error and PARTIAL in legacy_error and left.get("result"):
            # The facade reports a truncated token scan as partial, with its spans.
            truncated = right.get("result", {}).get("input", {}).get("truncated")
            legacy_error = None if truncated else legacy_error
        if legacy_error or runtime_error:
            if legacy_error and runtime_error:
                both_errors += 1
            else:
                disagreements.append(
                    {
                        "id": item_id,
                        "reason": "error",
                        "legacy": legacy_error,
                        "runtime": runtime_error,
                    }
                )
            continue
        compared += 1
        a = baseline(spec, item, left["result"])
        b = runtime_values(spec, right["result"])
        if (
            "windows" in a
            and a.get("windows") is not None
            and a["windows"] != b.get("windows")
        ):
            disagreements.append(
                {
                    "id": item_id,
                    "reason": "windows",
                    "legacy": a["windows"],
                    "runtime": b.get("windows"),
                }
            )
        if kind in ("sequence", "scores"):
            delta = max(
                abs(x - y) for x, y in zip(a["values"], b["values"], strict=True)
            )
            max_delta = max(max_delta, delta)
            if kind == "sequence":
                ordered = sorted(a["values"], reverse=True)
                tie = (
                    len(ordered) > 1
                    and ordered[0] - ordered[1] < thresholds["near_tie"]
                )
                near_ties += tie
                same = a["values"].index(max(a["values"])) == b["values"].index(
                    max(b["values"])
                )
                agreed += same or tie
                if not same:
                    disagreements.append(
                        {
                            "id": item_id,
                            "reason": "label",
                            "near_tie": tie,
                            "delta": delta,
                        }
                    )
            else:
                agreed += 1
            if delta > thresholds["probability"]:
                disagreements.append(
                    {"id": item_id, "reason": "probability", "delta": delta}
                )
        else:
            left_spans = {(s[0], s[1], s[2]) for s in a["spans"]}
            right_spans = {(s[0], s[1], s[2]) for s in b["spans"]}
            same = left_spans == right_spans
            agreed += same
            if same:
                by_span = {(s[0], s[1], s[2]): s[3] for s in a["spans"]}
                for s in b["spans"]:
                    max_delta = max(max_delta, abs(by_span[(s[0], s[1], s[2])] - s[3]))
            else:
                disagreements.append(
                    {
                        "id": item_id,
                        "reason": "spans",
                        "legacy_only": sorted(left_spans - right_spans),
                        "runtime_only": sorted(right_spans - left_spans),
                    }
                )
    rate = agreed / compared if compared else 0.0
    bar = thresholds["label"] if kind in ("sequence", "scores") else thresholds["spans"]
    passed = (
        rate >= bar
        and max_delta <= thresholds["probability"]
        and not any(
            d["reason"] in ("missing", "error", "windows") for d in disagreements
        )
    )
    return {
        "job": spec["Job"],
        "model": spec["Repo"],
        "revision": spec["Revision"],
        "mode": spec["Mode"],
        "options": {
            "max_tokens": spec["MaxTokens"],
            "overflow": spec["Overflow"],
            "window": (
                [spec["WindowSize"], spec["WindowOverlap"]]
                if spec["WindowSize"]
                else None
            ),
        },
        "inputs": len(inputs),
        "compared": compared,
        "both_rejected": both_errors,
        "agreement": rate,
        "near_ties": near_ties,
        "max_abs_delta": max_delta,
        "passed": passed,
        "disagreements": disagreements,
        "latency_ms": {
            "legacy": {
                "p50": percentile(latency_legacy, 0.5),
                "p95": percentile(latency_legacy, 0.95),
            },
            "runtime": {
                "p50": percentile(latency_runtime, 0.5),
                "p95": percentile(latency_runtime, 0.95),
            },
        },
    }


def throughput(load: dict[str, Any] | None) -> dict[str, float] | None:
    if not load:
        return None
    return {
        "concurrency": load["concurrency"],
        "calls_per_s": load["calls"] / load["seconds"],
        "p50_ms": percentile(load["latency_ns"], 0.5),
        "p95_ms": percentile(load["latency_ns"], 0.95),
    }


def run_compare(args: argparse.Namespace) -> None:
    legacy, legacy_loads = read_lines(args.legacy)
    runtime, runtime_loads = read_lines(args.runtime)
    jobs_file = args.jobs or Path(args.legacy).with_suffix(".jobs.json")
    specs = json.loads(Path(jobs_file).read_text(encoding="utf-8"))
    if args.baseline == "runtime":
        thresholds = REDUCED_THRESHOLDS

        def baseline(spec: dict[str, Any], _: Any, result: Any) -> dict[str, Any]:
            return runtime_values(spec, result)

    else:
        thresholds = ROCM_THRESHOLDS if args.device_class == "rocm" else CPU_THRESHOLDS
        baseline = legacy_values
    jobs = []
    for spec in specs:
        report = compare_job(spec, legacy, runtime, thresholds, baseline)
        report["throughput"] = {
            "legacy": throughput(legacy_loads.get(spec["Job"])),
            "runtime": throughput(runtime_loads.get(spec["Job"])),
        }
        jobs.append(report)
    record = {
        "format": "vela1-legacy-parity/1",
        "baseline": args.baseline,
        "device_class": args.device_class,
        "thresholds": thresholds,
        "inputs_sha256": hashlib.sha256(
            json.dumps([spec["Inputs"] for spec in specs], sort_keys=True).encode()
        ).hexdigest(),
        "context": json.loads(args.context) if args.context else {},
        "jobs": jobs,
        "passed": all(job["passed"] for job in jobs),
    }
    Path(args.record).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for job in jobs:
        lat, thr = job["latency_ms"], job["throughput"]
        print(
            f"{job['job']:13s} {'PASS' if job['passed'] else 'FAIL'} agree={job['agreement']:.4f} "
            f"max|Δ|={job['max_abs_delta']:.2e} p50 {lat['legacy']['p50']:.2f}→{lat['runtime']['p50']:.2f} ms "
            f"p95 {lat['legacy']['p95']:.2f}→{lat['runtime']['p95']:.2f} ms "
            + (
                f"tput {thr['legacy']['calls_per_s']:.1f}→{thr['runtime']['calls_per_s']:.1f}/s"
                if thr["legacy"] and thr["runtime"]
                else ""
            )
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    corpus_parser = commands.add_parser("corpus")
    corpus_parser.add_argument("--out", required=True)
    for name in ("legacy", "runtime"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--cache", required=True, help="HF cache holding the pinned snapshots"
        )
        sub.add_argument("--recipe", choices=sorted(RECIPES), default="cpu")
        sub.add_argument(
            "--jobs",
            default="",
            help="comma-separated (default: every job of the recipe)",
        )
        sub.add_argument("--out", required=True)
        sub.add_argument("--repeats", type=int, default=1)
        sub.add_argument("--concurrency", type=int, default=0)
        sub.add_argument("--seconds", type=float, default=20.0)
        sub.add_argument("--limit", type=int, default=0)
        sub.add_argument(
            "--inputs", help="an earlier run's .jobs.json: answer exactly its inputs"
        )
        if name == "legacy":
            sub.add_argument(
                "--tree",
                required=True,
                help="a copy of the legacy commit's mirror with built bindings",
            )
            sub.add_argument(
                "--flat",
                required=True,
                help="where to link the snapshots as plain model directories",
            )
            sub.add_argument(
                "--libs",
                help="binding library directories to link first (a GPU image's)",
            )
            sub.add_argument("--compile-cache", help="ORT compilation cache directory")
        else:
            sub.add_argument("--device", default="cpu")
            sub.add_argument("--profile", default="exact")
            sub.add_argument("--threads", type=int)
    build = commands.add_parser("build-legacy")
    build.add_argument("--cache", required=True)
    build.add_argument("--recipe", choices=sorted(RECIPES), default="cpu")
    build.add_argument("--jobs", default="")
    build.add_argument("--out", required=True, help="the test binary to write")
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--tree", required=True)
    build.add_argument("--flat", required=True)
    build.add_argument("--libs")
    build.add_argument("--compile-cache")
    for name, default in (("repeats", 1), ("concurrency", 0)):
        build.add_argument(f"--{name}", type=int, default=default)
    build.add_argument("--seconds", type=float, default=0.0)
    ab_parser = commands.add_parser("ab")
    ab_parser.add_argument("--binary", required=True, help="a build-legacy test binary")
    ab_parser.add_argument("--tree", required=True)
    ab_parser.add_argument("--libs")
    ab_parser.add_argument("--cache", required=True)
    ab_parser.add_argument("--threads", type=int)
    ab_parser.add_argument(
        "--profile",
        default="batching",
        help="the models' profile (exact is always measured)",
    )
    ab_parser.add_argument("--rounds", type=int, default=2)
    ab_parser.add_argument(
        "--concurrency", type=int, default=0, help="callers per load window (0: none)"
    )
    ab_parser.add_argument("--seconds", type=float, default=20.0)
    ab_parser.add_argument("--out", required=True)
    ab_parser.add_argument(
        "--resume",
        help="an earlier run's .raw.json: keep its rounds, run the rest up to --rounds",
    )
    compare = commands.add_parser("compare")
    compare.add_argument("--legacy", required=True, help="the baseline's results")
    compare.add_argument(
        "--baseline",
        choices=("legacy", "runtime"),
        default="legacy",
        help="runtime: --legacy is another runtime run (a reduced copy against exact)",
    )
    compare.add_argument(
        "--jobs", help="job specs (default: the baseline's .jobs.json)"
    )
    compare.add_argument("--runtime", required=True)
    compare.add_argument("--record", required=True)
    compare.add_argument("--device-class", choices=("cpu", "rocm"), default="cpu")
    compare.add_argument(
        "--context", help="JSON with commit, image, device and threads"
    )
    for sub in (corpus_parser, *commands.choices.values()):
        if not any(action.dest == "seed" for action in sub._actions):
            sub.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "corpus":
        Path(args.out).write_text(
            "".join(json.dumps(i) + "\n" for i in corpus(args.seed)), encoding="utf-8"
        )
    elif args.command == "legacy":
        run_legacy(args)
    elif args.command == "build-legacy":
        build_legacy(args)
    elif args.command == "ab":
        run_ab(args)
    elif args.command == "runtime":
        run_runtime(args)
    else:
        run_compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
