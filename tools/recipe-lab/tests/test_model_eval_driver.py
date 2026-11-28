"""Tests for the model_eval verify driver (C7)."""

import hashlib
import json
from pathlib import Path

import pytest
from recipe_lab.adapters.router_replay import (
    ReplayFetchError,
    routing_shares_from_replays,
)
from recipe_lab.contract import API_VERSION, ContractError, ExperimentSpec
from recipe_lab.live import paired_stats_from_comparison
from recipe_lab.model_eval_driver import (
    arm_quality_rates,
    paired_stats_from_judged,
    random_mix_paired_stats,
    run_arm,
    verify_live_model_eval,
)
from recipe_lab.verdict import compute_paired_stats

BASE_ENDPOINT = "http://127.0.0.1:9001/v1"
CAND_ENDPOINT = "http://127.0.0.1:9002/v1"
ROUTER_MODEL = "vllm-sr/auto"
STRONG = "Qwen/Qwen3.5-4B"
WEAK = "plainmodel"
N_TASKS = 30  # radius sqrt(2*ln(40)/30) < 0.5, so a 0.5 delta passes a 0pp bar


@pytest.fixture(autouse=True)
def _candidate_config(tmp_path, monkeypatch):
    """Y4: a candidate config the router's management API acknowledges.

    Every verify call binds `candidate_config` to this file, and the
    patched `fetch_config_hash` reports it active with the file's source
    hash. Negative tests override the patch.
    """
    path = tmp_path / "candidate-config.yaml"
    path.write_text("api_version: v1alpha1\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "recipe_lab.model_eval_driver.fetch_config_hash",
        lambda url, **kwargs: {
            "activation_status": "active",
            "source_config_hash": digest,
        },
    )
    return path


def _spec_dict(**overrides):
    spec = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "accuracy",
            "direction": "maximize",
            "min_cases": 10,
            "min_improvement_pp": 0.0,
        },
        # default min_label_cases is 20; the fixtures use 15 cases per label
        "profile": {"min_label_cases": 2},
        "workloads": [
            {
                "name": "cold",
                "profile": "quick",
                "driver": "model_eval",
                "splits": {
                    "dev": {
                        "manifest": "dev.jsonl",
                        "content_hash": "sha256:" + "0" * 64,
                    }
                },
            }
        ],
        "attribution": {"oracle_policy": {"math": STRONG}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 5},
        "models": [
            {"name": STRONG, "endpoint": "http://127.0.0.1:8001/v1"},
            {
                "name": WEAK,
                "endpoint": "http://127.0.0.1:8002/v1",
                "served_name": "plain-served",
            },
        ],
        "generation": {"temperature": 0.0, "max_tokens": 512},
        "references": {"strong": STRONG, "weak": WEAK},
        "cost_basis": {
            "source": "provider_pricing",
            "assumed_pricing": {
                STRONG: {"prompt_per_1m": 0.5, "completion_per_1m": 1.0}
            },
        },
    }
    spec.update(overrides)
    return spec


def _write_tasks(path: Path, n: int = N_TASKS) -> None:
    rows = [
        {
            "task_id": f"t{i}",
            "prompt_user": f"question {i}",
            "category": "math" if i % 2 == 0 else "law",
            "answer": "A",
        }
        for i in range(n)
    ]
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )


def _write_labels(path: Path, n: int = N_TASKS) -> None:
    """C2-shaped labels matching the tasks' own categories (math/law)."""
    rows = [
        {"case_id": f"t{i}", "router_label": "math" if i % 2 == 0 else "law"}
        for i in range(n)
    ]
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )


def _judged_rows(tasks: list[dict], correct_ids: set[str]) -> list[dict]:
    return [
        {
            "task_id": task["task_id"],
            "category": task["category"],
            "is_correct": task["task_id"] in correct_ids,
            "no_answer": False,
            "finish_reason": "stop",
            "completion_tokens": 10,
            "latency_s": 0.5,
        }
        for task in tasks
    ]


def _fake_run_arm(correct_by_arm: dict[str, set[str]]):
    """A run_arm stand-in: writes judged+records driven by the plan.

    `correct_by_arm` keys match the endpoint for router arms and the
    served model name for reference arms. The records carry no responding
    model (the generation script's shape): the routing distribution is
    replay evidence, not a record field. Returns (fake, calls).
    """
    calls: list[dict] = []

    def fake(
        *,
        python_bin,
        model_eval_dir,
        tasks_path,
        out_dir,
        endpoint,
        model,
        params_path,
        concurrency,
        request_timeout,
        timeout_s,
        force=False,
    ):
        calls.append({"endpoint": endpoint, "model": model})
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tasks = [
            json.loads(line)
            for line in Path(tasks_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        correct = correct_by_arm.get(endpoint, correct_by_arm.get(model, set()))
        judged = _judged_rows(tasks, correct)
        (out / "judged.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in judged),
            encoding="utf-8",
        )
        records = [
            {
                "task_id": task["task_id"],
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "latency_s": 0.5,
                "error": None,
            }
            for task in tasks
        ]
        (out / "records.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
            encoding="utf-8",
        )
        return out / "judged.jsonl"

    return fake, calls


def _golden_plan():
    tasks = [
        {"task_id": f"t{i}", "category": "math" if i % 2 == 0 else "law"}
        for i in range(N_TASKS)
    ]
    return tasks, {
        # baseline: correct on every even case; candidate: always correct
        BASE_ENDPOINT: {t["task_id"] for t in tasks if int(t["task_id"][1:]) % 2 == 0},
        CAND_ENDPOINT: {t["task_id"] for t in tasks},
        STRONG: {t["task_id"] for t in tasks},
        "plain-served": set(),
    }


def test_paired_stats_from_judged_matches_compute_paired_stats():
    base = [
        {"task_id": "a", "is_correct": True},
        {"task_id": "b", "is_correct": False},
        {"task_id": "c", "is_correct": True},
        {"task_id": "d", "is_correct": True},
    ]
    cand = [
        {"task_id": "a", "is_correct": True},
        {"task_id": "b", "is_correct": True},
        {"task_id": "c", "is_correct": False},
        {"task_id": "d", "is_correct": True},
    ]
    stats = paired_stats_from_judged(base, cand)
    expected = compute_paired_stats([0, 1, -1, 0])  # cand - base per case
    assert stats == expected
    assert (stats.wins, stats.losses, stats.ties) == (1, 1, 2)
    assert stats.paired_cases == 4


def test_paired_stats_from_judged_matches_srbench_compare_shape():
    # C7 completion criterion (unit form): on the same per-case results the
    # driver's stats carry the same fields an sr-bench compare payload would.
    base = [
        {"task_id": "a", "is_correct": True},
        {"task_id": "b", "is_correct": False},
        {"task_id": "c", "is_correct": True},
        {"task_id": "d", "is_correct": True},
    ]
    cand = [
        {"task_id": "a", "is_correct": True},
        {"task_id": "b", "is_correct": True},
        {"task_id": "c", "is_correct": True},
        {"task_id": "d", "is_correct": True},
    ]
    stats = paired_stats_from_judged(base, cand)
    comparison = {
        "comparisons": [
            {
                "paired_cases": 4,
                "quality_delta": 0.25,
                "quality_delta_ci95": [0.0, 0.5],
                "wins": 1,
                "losses": 0,
                "ties": 3,
            }
        ]
    }
    from_compare = paired_stats_from_comparison(comparison)
    assert from_compare is not None
    assert stats.paired_cases == from_compare.paired_cases
    assert stats.quality_delta == from_compare.quality_delta
    assert (stats.wins, stats.losses, stats.ties) == (
        from_compare.wins,
        from_compare.losses,
        from_compare.ties,
    )


def test_paired_stats_from_judged_partial_overlap_and_disjoint():
    base = [{"task_id": "a", "is_correct": True}, {"task_id": "b", "is_correct": True}]
    cand = [{"task_id": "b", "is_correct": False}, {"task_id": "c", "is_correct": True}]
    stats = paired_stats_from_judged(base, cand)
    assert stats is not None
    assert stats.paired_cases == 1  # only "b" is common
    assert (stats.wins, stats.losses, stats.ties) == (0, 1, 0)
    assert (
        paired_stats_from_judged(base, [{"task_id": "z", "is_correct": True}]) is None
    )


def test_arm_quality_rates_rates_and_missing():
    rows = [
        {"finish_reason": "stop", "no_answer": False},
        {"finish_reason": "length", "no_answer": True},
        {"finish_reason": "stop", "no_answer": False},
        {"finish_reason": "stop", "no_answer": False},
    ]
    rates = arm_quality_rates(rows, planned_cases=5)
    assert rates == {
        "truncation_rate": 0.25,
        "missing_result_rate": 0.2,
        "no_answer_rate": 0.25,
        "finish_reason_length_rate": 0.25,
    }
    # no judged rows at all: the arm is entirely missing, the per-case rates
    # stay unknown (never guessed as zero)
    assert arm_quality_rates([], planned_cases=5) == {
        "truncation_rate": None,
        "missing_result_rate": 1.0,
        "no_answer_rate": None,
        "finish_reason_length_rate": None,
    }
    assert arm_quality_rates(rows, planned_cases=0) is None


def test_routing_shares_from_replays():
    # the routing distribution joins the candidate router's replay records
    # by prompt; failed records and model-less records are no routing
    # decision, and prompts outside the task set are not this arm's traffic
    prompts = {"question 0", "question 1"}
    records = [
        {"prompt": "question 0", "selected_model": "m-a"},
        {"prompt": "question 0", "SelectedModel": "m-b"},
        {"prompt": "question 1", "selected_model": "m-a"},
        {"prompt": "question 1", "lifecycle_state": "failed", "selected_model": "m-a"},
        {"prompt": "question 1", "selected_model": "  "},
        {"prompt": "other", "selected_model": "m-z"},
        {"selected_model": "m-a"},  # no prompt: unmatched
    ]
    shares, stats = routing_shares_from_replays(records, prompts)
    assert shares == {"m-a": 2 / 3, "m-b": 1 / 3}
    assert stats == {"matched": 5, "failed": 1, "no_model": 1, "total": 7}
    # nothing countable -> unknown, never guessed
    assert routing_shares_from_replays([], prompts) == (
        None,
        {"matched": 0, "failed": 0, "no_model": 0, "total": 0},
    )
    only_failed = [{"prompt": "question 0", "lifecycle_state": "failed"}]
    failed_shares, failed_stats = routing_shares_from_replays(only_failed, prompts)
    assert failed_shares is None
    assert failed_stats["matched"] == 1
    assert failed_stats["failed"] == 1


def test_random_mix_paired_stats_matches_analytic():
    # C8 test: two models, synthetic per-task data; the bootstrap mean
    # matches the analytic delta mean(c) - sum(p_m * mean(ref_m)) and the
    # interval brackets it.
    n = 40
    cand = [{"task_id": f"t{i}", "is_correct": i < 30} for i in range(n)]
    strong = [{"task_id": f"t{i}", "is_correct": True} for i in range(n)]
    weak = [{"task_id": f"t{i}", "is_correct": i < 10} for i in range(n)]
    shares = {STRONG: 0.5, WEAK: 0.5}
    stats = random_mix_paired_stats(cand, {STRONG: strong, WEAK: weak}, shares)
    assert stats is not None
    assert stats["paired_cases"] == n
    analytic = 0.75 - (0.5 * 1.0 + 0.5 * 0.25)
    assert stats["delta"] == pytest.approx(analytic)
    assert stats["bootstrap_mean"] == pytest.approx(analytic, abs=0.01)
    low, high = stats["bootstrap_ci95"]
    assert low <= analytic <= high


def test_random_mix_paired_stats_edges():
    n = 10
    cand = [{"task_id": f"t{i}", "is_correct": i % 2 == 0} for i in range(n)]
    strong = [{"task_id": f"t{i}", "is_correct": True} for i in range(n)]
    weak = [{"task_id": f"t{i}", "is_correct": False} for i in range(n)]
    # a single-model share collapses R to that reference arm exactly
    stats = random_mix_paired_stats(
        cand, {STRONG: strong, WEAK: weak}, {STRONG: 1.0, WEAK: 0.0}
    )
    assert stats is not None
    assert stats["delta"] == pytest.approx(0.5 - 1.0)
    assert stats["bootstrap_mean"] == pytest.approx(0.5 - 1.0, abs=0.01)
    # no common cases between the arms -> no paired evidence
    stranger = [{"task_id": "zz", "is_correct": True}]
    assert random_mix_paired_stats(stranger, {STRONG: strong}, {STRONG: 1.0}) is None
    # no positively-weighted model -> nothing to mix
    assert random_mix_paired_stats(cand, {STRONG: strong}, {STRONG: 0.0}) is None


def test_run_arm_argv_outputs_and_missing_script(tmp_path, monkeypatch):
    model_eval_dir = tmp_path / "scripts"
    model_eval_dir.mkdir()
    (model_eval_dir / "multi_benchmark_eval.py").write_text("# gen\n")
    (model_eval_dir / "mcq_judge.py").write_text("# judge\n")
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, n=2)
    captured: list[list[str]] = []

    def fake_run(argv, *, timeout_s, what):
        captured.append(argv)
        # emulate the judge writing its output
        out = Path(argv[argv.index("--out") + 1])
        out.write_text("", encoding="utf-8")

    monkeypatch.setattr("recipe_lab.model_eval_driver._run", fake_run)
    judged = run_arm(
        python_bin="python",
        model_eval_dir=model_eval_dir,
        tasks_path=tasks,
        out_dir=tmp_path / "arm",
        endpoint="http://ep/v1",
        model="served",
        params_path=tmp_path / "generation.json",
        concurrency=4,
        request_timeout=30.0,
        timeout_s=60.0,
    )
    assert judged == tmp_path / "arm" / "judged.jsonl"
    gen_argv, judge_argv = captured
    assert gen_argv[:2] == ["python", str(model_eval_dir / "multi_benchmark_eval.py")]
    assert gen_argv[2] == "--split"
    assert gen_argv[3] == str(tasks)
    assert gen_argv[gen_argv.index("--endpoint") + 1] == "http://ep/v1"
    assert gen_argv[gen_argv.index("--model") + 1] == "served"
    assert "--force" not in gen_argv
    assert judge_argv[1] == str(model_eval_dir / "mcq_judge.py")
    assert judge_argv[judge_argv.index("--tasks") + 1] == str(tasks)
    # --analysis is a profile-only output; the driver skips it
    assert "--analysis" not in judge_argv

    forced = run_arm(
        python_bin="python",
        model_eval_dir=model_eval_dir,
        tasks_path=tasks,
        out_dir=tmp_path / "arm2",
        endpoint="http://ep/v1",
        model="served",
        params_path=tmp_path / "generation.json",
        concurrency=4,
        request_timeout=30.0,
        timeout_s=60.0,
        force=True,
    )
    assert forced.is_file()
    assert captured[-2][-1] == "--force"

    # a missing script is an operational error, not a verdict input
    (model_eval_dir / "mcq_judge.py").unlink()
    with pytest.raises(ContractError, match="model_eval script not found"):
        run_arm(
            python_bin="python",
            model_eval_dir=model_eval_dir,
            tasks_path=tasks,
            out_dir=tmp_path / "arm3",
            endpoint="http://ep/v1",
            model="served",
            params_path=tmp_path / "generation.json",
            concurrency=4,
            request_timeout=30.0,
            timeout_s=60.0,
        )


def test_verify_live_model_eval_end_to_end(tmp_path, monkeypatch):
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path)
    _, plan = _golden_plan()
    fake, calls = _fake_run_arm(plan)
    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)
    spec = ExperimentSpec.from_dict(_spec_dict())
    run_cache = tmp_path / "run-cache.json"
    # an empty replay store: R stays unevaluable with the gap named
    monkeypatch.setattr("recipe_lab.model_eval_driver.fetch_replays", lambda url: [])

    result = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        labels_path=labels_path,
        candidate_config=tmp_path / "candidate-config.yaml",
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        management_endpoint="http://127.0.0.1:8080",
        work_dir=tmp_path / "work",
        run_cache=run_cache,
    )

    # router arms ran in AB order; references ran once each, direct
    assert [(c["endpoint"], c["model"]) for c in calls] == [
        (BASE_ENDPOINT, ROUTER_MODEL),
        (CAND_ENDPOINT, ROUTER_MODEL),
        ("http://127.0.0.1:8001/v1", STRONG),
        ("http://127.0.0.1:8002/v1", "plain-served"),
    ]
    assert result["verdict"] == "PASS"
    assert result["metrics"] == {"baseline": 0.5, "candidate": 1.0}
    assert result["driver"] == "model_eval"
    assert result["planned_cases"] == N_TASKS
    # Y4: the config the management API acknowledged is the fixture's file
    assert (
        result["config_sha256"]
        == "sha256:"
        + hashlib.sha256((tmp_path / "candidate-config.yaml").read_bytes()).hexdigest()
    )
    assert result["tasks_hash"].startswith("sha256:")
    assert result["labels_sha256"].startswith("sha256:")
    assert result["labeled_tasks_sha256"].startswith("sha256:")
    # labels matching the tasks' own categories rewrite byte-identically
    assert result["labeled_tasks_sha256"] == result["tasks_hash"]
    assert result["label_merges"] == {}
    assert result["uncovered_cases"] == 0
    assert result["preview"] is None
    assert result["baseline_run_id"] is None
    # M2: merged single-pair stats in the sr-bench compare shape
    paired = result["paired"]
    assert paired["paired_cases"] == N_TASKS
    assert paired["quality_delta"] == pytest.approx(0.5)
    assert paired["wins"] == 15
    assert paired["losses"] == 0
    assert paired["ties"] == 15
    assert paired["repetitions"] == 1
    assert paired["paired_observations"] == N_TASKS
    # M3: every rate measured and clean
    assert result["evidence_quality"]["candidate"]["truncation_rate"] == 0.0
    assert result["evidence_quality"]["candidate"]["missing_result_rate"] == 0.0
    # M4: references anchored, strong cost assumed from token counts
    assert result["references"]["strong"]["metric"] == 1.0
    assert result["references"]["strong"]["cached"] is False
    assert result["references"]["strong"]["cost_usd"] == pytest.approx(
        N_TASKS * (100 * 0.5 + 10 * 1.0) / 1e6
    )
    assert result["references"]["strong"]["cost_note"] == "assumed"
    # no assumed pricing for the weak model: cost unknown, never zero
    assert result["references"]["weak"]["metric"] == 0.0
    assert result["references"]["weak"]["cost_usd"] is None
    assert result["references"]["weak"]["cost_note"] is None
    pareto = result["pareto"]
    assert [row["id"] for row in pareto] == [
        f"router:{BASE_ENDPOINT}",
        f"router:{CAND_ENDPOINT}",
        STRONG,
        WEAK,
    ]
    assert pareto[0]["cost_usd"] is None  # router hides the served model
    assert pareto[2]["role"] == "strong"
    bench_refs = result["bench_refs"]
    assert bench_refs[0].startswith("meval:tasks:sha256:")
    assert bench_refs[1].startswith("meval:labeled-tasks:sha256:")
    assert bench_refs[2].startswith("meval:labels:sha256:")
    assert bench_refs[3:] == [
        "meval:baseline-rep0",
        "meval:candidate-rep0",
        "meval:reference:strong",
        "meval:reference:weak",
    ]
    cached_keys = json.loads(run_cache.read_text(encoding="utf-8"))
    assert len(cached_keys) == 2
    assert all(key.startswith(result["labeled_tasks_sha256"]) for key in cached_keys)
    # an empty replay store: the routing distribution is unknown, so the
    # random mix is unevaluable with a reason - never guessed
    assert result["random_mix"]["evaluable"] is False
    assert (
        "no replay record with a selected model matches"
        in result["random_mix"]["reason"]
    )
    assert result["random_mix"]["expected_accuracy"] is None
    assert result["distribution"] is None

    # a second verify reuses the reference arms (same rewritten tasks hash
    # + alias)
    result2 = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        labels_path=labels_path,
        candidate_config=tmp_path / "candidate-config.yaml",
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        management_endpoint="http://127.0.0.1:8080",
        work_dir=tmp_path / "work2",
        run_cache=run_cache,
    )
    assert result2["references"]["strong"]["cached"] is True
    assert result2["references"]["weak"]["cached"] is True
    assert result2["verdict"] == "PASS"
    ref_calls = [c for c in calls if c["model"] != ROUTER_MODEL]
    assert len(ref_calls) == 2  # not re-run


def test_verify_live_model_eval_random_mix(tmp_path, monkeypatch):
    # C8: the candidate router routes 3 of every 5 cases to STRONG (served
    # name defaults to the alias) and the rest to the weak model's served
    # name; the routing distribution comes from the candidate router's
    # replay store, fetched from the management endpoint. The mix resolves
    # against the reference arms and anchors a `ref:random_mix` constraint.
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path)
    _, plan = _golden_plan()
    fake, _ = _fake_run_arm(plan)
    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)
    fetch_urls: list[str] = []

    def fake_fetch(url):
        fetch_urls.append(url)
        return [
            {
                "prompt": f"question {i}",
                "selected_model": STRONG if i % 5 < 3 else "plain-served",
            }
            for i in range(N_TASKS)
        ]

    monkeypatch.setattr("recipe_lab.model_eval_driver.fetch_replays", fake_fetch)
    spec_dict = _spec_dict()
    # price the weak model too, so R's expected cost is computable
    spec_dict["cost_basis"]["assumed_pricing"][WEAK] = {
        "prompt_per_1m": 0.2,
        "completion_per_1m": 0.4,
    }
    spec_dict["objective"]["constraints"] = [
        {
            "name": "beat-mix",
            "metric": "accuracy",
            "op": "gte",
            "bound": "ref:random_mix",
        }
    ]
    spec = ExperimentSpec.from_dict(spec_dict)
    result = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        labels_path=labels_path,
        candidate_config=tmp_path / "candidate-config.yaml",
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        management_endpoint="http://127.0.0.1:8080",
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        work_dir=tmp_path / "work",
    )

    assert fetch_urls == ["http://127.0.0.1:8080"]
    mix = result["random_mix"]
    assert mix["evaluable"] is True
    assert mix["reason"] is None
    assert mix["shares"] == {STRONG: 0.6, WEAK: 0.4}
    assert result["distribution"] == {STRONG: 0.6, WEAK: 0.4}
    assert mix["expected_accuracy"] == pytest.approx(0.6)  # 0.6*1.0 + 0.4*0.0
    strong_case = (100 * 0.5 + 10 * 1.0) / 1e6
    weak_case = (100 * 0.2 + 10 * 0.4) / 1e6
    assert mix["expected_cost"] == pytest.approx(
        N_TASKS * (0.6 * strong_case + 0.4 * weak_case)
    )
    assert mix["cost_note"] == "assumed"
    paired = mix["paired"]
    assert paired["paired_cases"] == N_TASKS
    assert paired["delta"] == pytest.approx(0.4)  # 1.0 - 0.6
    assert paired["bootstrap_mean"] == pytest.approx(0.4, abs=0.01)
    low, high = paired["bootstrap_ci95"]
    assert low <= 0.4 <= high
    # the constraint resolves against the mix and passes
    constraint = result["constraints"][0]
    assert constraint["status"] == "pass"
    assert constraint["bound_value"] == pytest.approx(0.6)
    assert result["verdict"] == "PASS"
    # R's Pareto row trails the reference arms
    assert [row["id"] for row in result["pareto"]] == [
        f"router:{BASE_ENDPOINT}",
        f"router:{CAND_ENDPOINT}",
        STRONG,
        WEAK,
        "ref:random_mix",
    ]
    assert result["pareto"][-1]["accuracy"] == pytest.approx(0.6)
    assert result["pareto"][-1]["accuracy_ci95"] is None
    assert "meval:reference:random_mix" in result["bench_refs"]


def test_verify_live_model_eval_random_mix_unknown_model(tmp_path, monkeypatch):
    # a routed model outside spec.models keeps the mix unevaluable: a
    # ref:random_mix constraint is then not evaluated (INCONCLUSIVE)
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path)
    _, plan = _golden_plan()
    fake, _ = _fake_run_arm(plan)
    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)

    def fake_fetch(url):
        return [
            {"prompt": f"question {i}", "selected_model": "mystery-served"}
            for i in range(N_TASKS)
        ]

    monkeypatch.setattr("recipe_lab.model_eval_driver.fetch_replays", fake_fetch)
    spec_dict = _spec_dict()
    spec_dict["objective"]["constraints"] = [
        {
            "name": "beat-mix",
            "metric": "accuracy",
            "op": "gte",
            "bound": "ref:random_mix",
        }
    ]
    spec = ExperimentSpec.from_dict(spec_dict)
    result = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        labels_path=labels_path,
        candidate_config=tmp_path / "candidate-config.yaml",
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        management_endpoint="http://127.0.0.1:8080",
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        work_dir=tmp_path / "work",
    )
    mix = result["random_mix"]
    assert mix["evaluable"] is False
    assert "outside spec.models" in mix["reason"]
    assert result["distribution"] is None
    assert mix["expected_accuracy"] is None
    assert result["constraints"][0]["status"] == "not_evaluated"
    assert result["verdict"] == "INCONCLUSIVE"
    assert not any(row["id"] == "ref:random_mix" for row in result["pareto"])
    assert "meval:reference:random_mix" not in result["bench_refs"]


def test_verify_live_model_eval_random_mix_replay_gaps(tmp_path, monkeypatch):
    # the distribution evidence cannot be faked: a failed fetch or no
    # matching replay each keep R unevaluable with the gap named, the
    # ref:random_mix constraint unevaluated and the verdict INCONCLUSIVE
    # (a missing management endpoint is a hard contract error - Y4)
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path)
    _, plan = _golden_plan()
    fake, _ = _fake_run_arm(plan)
    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)
    spec_dict = _spec_dict()
    spec_dict["objective"]["constraints"] = [
        {
            "name": "beat-mix",
            "metric": "accuracy",
            "op": "gte",
            "bound": "ref:random_mix",
        }
    ]
    spec = ExperimentSpec.from_dict(spec_dict)
    common = {
        "spec": spec,
        "tasks_path": tasks_path,
        "labels_path": labels_path,
        "candidate_config": tmp_path / "candidate-config.yaml",
        "baseline_endpoint": BASE_ENDPOINT,
        "candidate_endpoint": CAND_ENDPOINT,
        "router_model": ROUTER_MODEL,
        "repetitions": 1,
        "measurements": [],
        "model_eval_dir": tmp_path / "scripts",
        "python_bin": "python",
        "work_dir": tmp_path / "work",
    }

    def broken_fetch(url):
        raise ReplayFetchError("management API returned HTTP 500")

    monkeypatch.setattr("recipe_lab.model_eval_driver.fetch_replays", broken_fetch)
    fetch_failed = verify_live_model_eval(
        **common, management_endpoint="http://127.0.0.1:8080"
    )

    monkeypatch.setattr(
        "recipe_lab.model_eval_driver.fetch_replays",
        lambda url: [{"prompt": "unrelated", "selected_model": "m-a"}],
    )
    no_match = verify_live_model_eval(
        **common, management_endpoint="http://127.0.0.1:8080"
    )

    assert "replay fetch failed" in fetch_failed["random_mix"]["reason"]
    assert (
        "no replay record with a selected model matches"
        in no_match["random_mix"]["reason"]
    )
    for result in (fetch_failed, no_match):
        assert result["random_mix"]["evaluable"] is False
        assert result["random_mix"]["expected_accuracy"] is None
        assert result["distribution"] is None
        assert result["constraints"][0]["status"] == "not_evaluated"
        assert result["verdict"] == "INCONCLUSIVE"
        assert not any(row["id"] == "ref:random_mix" for row in result["pareto"])
        assert "meval:reference:random_mix" not in result["bench_refs"]


def test_verify_live_model_eval_input_guards(tmp_path):
    # every guard below fires before the labels file is touched, so an
    # absent labels path is fine here
    labels_path = tmp_path / "absent-labels.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text("", encoding="utf-8")
    spec = ExperimentSpec.from_dict(_spec_dict())
    with pytest.raises(ContractError, match="empty"):
        verify_live_model_eval(
            spec=spec,
            tasks_path=tasks_path,
            baseline_endpoint=BASE_ENDPOINT,
            candidate_endpoint=CAND_ENDPOINT,
            router_model=ROUTER_MODEL,
            repetitions=1,
            measurements=[],
            model_eval_dir=tmp_path,
            labels_path=labels_path,
            candidate_config=tmp_path / "candidate-config.yaml",
            python_bin="python",
            work_dir=tmp_path / "w",
        )
    with pytest.raises(ContractError, match="repetition"):
        verify_live_model_eval(
            spec=spec,
            tasks_path=tasks_path,
            baseline_endpoint=BASE_ENDPOINT,
            candidate_endpoint=CAND_ENDPOINT,
            router_model=ROUTER_MODEL,
            repetitions=0,
            measurements=[],
            model_eval_dir=tmp_path,
            labels_path=labels_path,
            candidate_config=tmp_path / "candidate-config.yaml",
            python_bin="python",
            work_dir=tmp_path / "w",
        )
    _write_tasks(tasks_path)
    no_gen = _spec_dict()
    del no_gen["generation"]
    with pytest.raises(ContractError, match="generation"):
        verify_live_model_eval(
            spec=ExperimentSpec.from_dict(no_gen),
            tasks_path=tasks_path,
            baseline_endpoint=BASE_ENDPOINT,
            candidate_endpoint=CAND_ENDPOINT,
            router_model=ROUTER_MODEL,
            repetitions=1,
            measurements=[],
            model_eval_dir=tmp_path,
            labels_path=labels_path,
            candidate_config=tmp_path / "candidate-config.yaml",
            python_bin="python",
            work_dir=tmp_path / "w",
        )


def test_verify_live_model_eval_rewrites_categories_from_labels(tmp_path, monkeypatch):
    # Y3: the driver rewrites the bound tasks' categories from the C2
    # labels file itself - the arms answer the rewritten file written into
    # the work dir, never an externally produced one, and the rewrite
    # inputs are hashed into the result (uncovered cases join `other`).
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    labels_path = tmp_path / "labels.jsonl"
    # labels cover only every other case; the rest is uncovered
    rows = [{"case_id": f"t{i}", "router_label": "code"} for i in range(0, N_TASKS, 2)]
    labels_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    seen_tasks: list[Path] = []

    def fake(**kwargs):
        seen_tasks.append(Path(kwargs["tasks_path"]))
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        tasks = [
            json.loads(line)
            for line in Path(kwargs["tasks_path"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        judged_rows = _judged_rows(tasks, {t["task_id"] for t in tasks})
        judged = out / "judged.jsonl"
        judged.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in judged_rows),
            encoding="utf-8",
        )
        return judged

    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)
    spec = ExperimentSpec.from_dict(_spec_dict())
    result = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        labels_path=labels_path,
        candidate_config=tmp_path / "candidate-config.yaml",
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        management_endpoint="http://127.0.0.1:8080",
        work_dir=tmp_path / "work",
    )
    # every arm answered the same rewritten file, not the bound one
    assert seen_tasks
    assert all(path == seen_tasks[0] for path in seen_tasks)
    assert seen_tasks[0] != tasks_path
    labeled = [
        json.loads(line)
        for line in seen_tasks[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["category"] for row in labeled] == [
        "code" if i % 2 == 0 else "other" for i in range(N_TASKS)
    ]
    assert result["uncovered_cases"] == N_TASKS // 2
    assert result["label_merges"] == {}
    assert result["labels_sha256"].startswith("sha256:")
    assert result["labeled_tasks_sha256"].startswith("sha256:")
    assert result["labeled_tasks_sha256"] != result["tasks_hash"]


def test_verify_live_model_eval_missing_labels_is_actionable(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    spec = ExperimentSpec.from_dict(_spec_dict())
    with pytest.raises(ContractError, match="labels file not found"):
        verify_live_model_eval(
            spec=spec,
            tasks_path=tasks_path,
            labels_path=tmp_path / "absent-labels.jsonl",
            candidate_config=tmp_path / "candidate-config.yaml",
            baseline_endpoint=BASE_ENDPOINT,
            candidate_endpoint=CAND_ENDPOINT,
            router_model=ROUTER_MODEL,
            repetitions=1,
            measurements=[],
            model_eval_dir=tmp_path / "scripts",
            python_bin="python",
            work_dir=tmp_path / "work",
        )


def test_verify_live_model_eval_candidate_config_guards(tmp_path, monkeypatch):
    # Y4: the candidate config is bound before any arm is dispatched - the
    # file must exist, the management endpoint is mandatory, and the router
    # must acknowledge an ACTIVE config whose source hash is the file's.
    # Every guard fires before run_arm, so no arm ever answers an
    # unverified router.
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path)
    spec = ExperimentSpec.from_dict(_spec_dict())
    config_path = tmp_path / "candidate-config.yaml"  # written by the fixture
    common = {
        "spec": spec,
        "tasks_path": tasks_path,
        "labels_path": labels_path,
        "candidate_config": config_path,
        "baseline_endpoint": BASE_ENDPOINT,
        "candidate_endpoint": CAND_ENDPOINT,
        "router_model": ROUTER_MODEL,
        "repetitions": 1,
        "measurements": [],
        "model_eval_dir": tmp_path / "scripts",
        "python_bin": "python",
        "work_dir": tmp_path / "work",
    }
    with pytest.raises(ContractError, match="candidate config file not found"):
        verify_live_model_eval(
            **{**common, "candidate_config": tmp_path / "absent.yaml"}
        )
    with pytest.raises(
        ContractError, match="--candidate-management-endpoint is required"
    ):
        verify_live_model_eval(**common)

    def broken(url, **kwargs):
        raise ReplayFetchError("management API returned HTTP 503")

    monkeypatch.setattr("recipe_lab.model_eval_driver.fetch_config_hash", broken)
    with pytest.raises(
        ContractError, match="cannot check the candidate router's config"
    ):
        verify_live_model_eval(**common, management_endpoint="http://127.0.0.1:8080")
    # the pending test uses the file's own hash: the failure is about the
    # activation status, not a mismatch
    monkeypatch.setattr(
        "recipe_lab.model_eval_driver.fetch_config_hash",
        lambda url, **kwargs: {
            "activation_status": "pending",
            "source_config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
    )
    with pytest.raises(ContractError, match="not active"):
        verify_live_model_eval(**common, management_endpoint="http://127.0.0.1:8080")
    monkeypatch.setattr(
        "recipe_lab.model_eval_driver.fetch_config_hash",
        lambda url, **kwargs: {
            "activation_status": "active",
            "source_config_hash": "f" * 64,
        },
    )
    with pytest.raises(ContractError, match="hashes to"):
        verify_live_model_eval(**common, management_endpoint="http://127.0.0.1:8080")


def test_verify_live_model_eval_empty_arm_is_inconclusive(tmp_path, monkeypatch):
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path)

    def fake(**kwargs):
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        judged = out / "judged.jsonl"
        if kwargs["endpoint"] == CAND_ENDPOINT:
            judged.write_text("", encoding="utf-8")  # zero judged rows
        else:
            tasks = [
                json.loads(line)
                for line in Path(kwargs["tasks_path"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            rows = _judged_rows(tasks, {t["task_id"] for t in tasks})
            judged.write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                encoding="utf-8",
            )
        return judged

    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)
    spec = ExperimentSpec.from_dict(_spec_dict())
    result = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        labels_path=labels_path,
        candidate_config=tmp_path / "candidate-config.yaml",
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        management_endpoint="http://127.0.0.1:8080",
        work_dir=tmp_path / "work",
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("no judged results" in reason for reason in result["reasons"])
    assert result["paired"] is None
    assert result["metrics"]["candidate"] is None


def test_verify_live_model_eval_constraint_not_measurable(tmp_path, monkeypatch):
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path)
    _, plan = _golden_plan()
    fake, _ = _fake_run_arm(plan)
    monkeypatch.setattr("recipe_lab.model_eval_driver.run_arm", fake)
    spec_dict = _spec_dict()
    spec_dict["objective"]["constraints"] = [
        {"name": "cheap", "metric": "routed_cost_index", "op": "lte", "bound": 1.0}
    ]
    spec = ExperimentSpec.from_dict(spec_dict)
    result = verify_live_model_eval(
        spec=spec,
        tasks_path=tasks_path,
        labels_path=labels_path,
        candidate_config=tmp_path / "candidate-config.yaml",
        baseline_endpoint=BASE_ENDPOINT,
        candidate_endpoint=CAND_ENDPOINT,
        router_model=ROUTER_MODEL,
        repetitions=1,
        measurements=[],
        model_eval_dir=tmp_path / "scripts",
        python_bin="python",
        management_endpoint="http://127.0.0.1:8080",
        work_dir=tmp_path / "work",
    )
    assert result["verdict"] == "INCONCLUSIVE"
    constraint = result["constraints"][0]
    assert constraint["status"] == "not_evaluated"
    assert any("not measurable" in reason for reason in result["reasons"])
