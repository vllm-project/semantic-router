"""Decision 3 quantized qualification records bind evidence to exact packages."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).parents[1] / "tools" / "decision3_quantized_qualification.py"


@pytest.fixture(scope="module")
def qualification():
    spec = importlib.util.spec_from_file_location(
        "decision3_quantized_qualification", TOOL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package(tmp_path, name, model_sha256, published=True):
    manifest = {
        "schema": "d3-package-manifest/1",
        "identity": {"model_sha256": model_sha256},
    }
    path = tmp_path / f"{name}-MODEL_MANIFEST.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    ref = {
        "model_sha256": model_sha256,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if published:
        ref.update(
            {
                "repo": f"vllm-sr/{name}",
                "revision": ("a" if name == "d3" else "b") * 40,
            }
        )
    return path, ref


def dataset(split, rows, digit):
    return {
        "repo": "tasksource/tasksource-jev-typed-decisions",
        "revision": "c" * 40,
        "split": split,
        "rows": rows,
        "digest": "sha256:" + digit * 64,
    }


def evidence(tmp_path, scheme="int4", symmetric=False):
    baseline_path, baseline = package(tmp_path, "d3", "1" * 64)
    candidate_path, candidate = package(tmp_path, "d3-awq", "2" * 64, published=False)
    record = {
        "format": "decision3-quantized-qualification/1",
        "policy": {
            "issue": "https://github.com/vllm-project/semantic-router/issues/9999"
        },
        "baseline": baseline,
        "candidate": candidate,
        "quantization": {
            "source": dict(baseline),
            "scheme": scheme,
            "method": "max-per-channel" if scheme == "float8" else "awq",
            "group_size": 0 if scheme == "float8" else 128,
            "symmetric": True if scheme == "float8" else symmetric,
            "calibration": None if scheme == "float8" else dataset("train", 128, "3"),
            "calibration_evaluation_overlap_rows": None if scheme == "float8" else 0,
        },
        "evaluation": {
            "dataset": dataset("test", 14_031, "4"),
            "harness": {
                "code_revision": "d" * 40,
                "command": ["python3", "bench/d3-quant-eval/evaluate.py"],
                "device": {
                    "kind": "rocm",
                    "name": "AMD Instinct MI300X",
                    "architecture": "gfx942",
                },
                "dependencies": {
                    "torch": "2.12.0",
                    "triton": "3.7.1",
                    "fla": "0.5.2",
                },
            },
            "quality": {
                "baseline_correct": 11_681,
                "candidate_correct": 11_658,
                "changed_answers": 314,
                "by_kind": {
                    "choice": {
                        "rows": 6_000,
                        "baseline_correct": 5_000,
                        "candidate_correct": 4_990,
                        "changed_answers": 120,
                    },
                    "noul": {
                        "rows": 4_000,
                        "baseline_correct": 3_300,
                        "candidate_correct": 3_290,
                        "changed_answers": 90,
                    },
                    "score": {
                        "rows": 4_031,
                        "baseline_correct": 3_381,
                        "candidate_correct": 3_378,
                        "changed_answers": 104,
                    },
                },
            },
            "reliability": {
                "baseline_errors": 0,
                "candidate_errors": 0,
                "baseline_nan_results": 0,
                "candidate_nan_results": 0,
                "baseline_oom_events": 0,
                "candidate_oom_events": 0,
                "baseline_missing_results": 0,
                "candidate_missing_results": 0,
            },
            "memory": {
                "baseline_package_bytes": 52_200_000_000,
                "candidate_package_bytes": 16_100_000_000,
                "baseline_loaded_device_bytes": 52_200_000_000,
                "candidate_loaded_device_bytes": 16_100_000_000,
                "baseline_peak_bytes": 52_900_000_000,
                "candidate_peak_bytes": 17_500_000_000,
            },
            "performance": {
                "requests": 500,
                "baseline_load_seconds": 38.2,
                "candidate_load_seconds": 17.5,
                "latency_ms": {
                    "baseline": {"median": 45.1, "p95": 78.4},
                    "candidate": {"median": 60.2, "p95": 92.1},
                },
                "throughput_rps": {
                    "1": {"baseline": 20.1, "candidate": 15.2},
                    "4": {"baseline": 52.4, "candidate": 39.7},
                    "16": {"baseline": 71.8, "candidate": 58.3},
                },
            },
        },
        "gates": {
            "min_decision_agreement": 0.97,
            "max_accuracy_drop": 0.005,
            "max_memory_ratio": 0.5,
            "max_latency_ratio": 1.5,
        },
    }
    return record, baseline_path, candidate_path


def test_finalize_binds_a_pass_to_exact_package_manifests(tmp_path, qualification):
    record, baseline_path, candidate_path = evidence(tmp_path)

    finalized = qualification.finalize_record(record)
    qualification.verify_package_manifest(
        finalized["baseline"], baseline_path, "baseline"
    )
    qualification.verify_package_manifest(
        finalized["candidate"], candidate_path, "candidate"
    )
    qualification.check_record(finalized)

    assert finalized["outcome"] == {
        "passed": True,
        "checks": {
            "decision_agreement": {
                "observed": 0.97762098,
                "minimum": 0.97,
                "passed": True,
            },
            "accuracy_drop": {
                "observed": 0.00163923,
                "maximum": 0.005,
                "passed": True,
            },
            "memory_ratio": {
                "observed": 0.33081285,
                "maximum": 0.5,
                "passed": True,
            },
            "latency_ratio": {
                "observed": 1.33481153,
                "maximum": 1.5,
                "passed": True,
            },
            "reliability_failures": {
                "observed": 0,
                "maximum": 0,
                "passed": True,
            },
            "calibration_overlap_rows": {
                "observed": 0,
                "maximum": 0,
                "passed": True,
            },
        },
    }


def test_failed_gate_produces_an_explicit_fail_record(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["gates"]["min_decision_agreement"] = 0.99

    finalized = qualification.finalize_record(record)

    assert finalized["outcome"]["passed"] is False
    assert finalized["outcome"]["checks"]["decision_agreement"]["passed"] is False


def test_stale_or_hand_written_verdict_is_rejected(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    finalized = qualification.finalize_record(record)
    finalized["outcome"]["passed"] = False

    with pytest.raises(qualification.QualificationError, match="differs"):
        qualification.check_record(finalized)


def test_manifest_bytes_must_match_the_record(tmp_path, qualification):
    record, _, candidate_path = evidence(tmp_path)
    finalized = qualification.finalize_record(record)
    candidate_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(qualification.QualificationError, match="SHA-256"):
        qualification.verify_package_manifest(
            finalized["candidate"], candidate_path, "candidate"
        )


def test_non_utf8_manifest_is_a_validation_error(tmp_path, qualification):
    record, _, candidate_path = evidence(tmp_path)
    finalized = qualification.finalize_record(record)
    candidate_path.write_bytes(b"\xff")
    finalized["candidate"]["manifest_sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()

    with pytest.raises(qualification.QualificationError, match="cannot be read"):
        qualification.verify_package_manifest(
            finalized["candidate"], candidate_path, "candidate"
        )


def test_quantized_model_identity_must_match_the_record(tmp_path, qualification):
    record, _, candidate_path = evidence(tmp_path)
    finalized = qualification.finalize_record(record)
    manifest = json.loads(candidate_path.read_text(encoding="utf-8"))
    manifest["identity"]["model_sha256"] = "9" * 64
    candidate_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    finalized["candidate"]["manifest_sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()

    with pytest.raises(qualification.QualificationError, match="model identity"):
        qualification.verify_package_manifest(
            finalized["candidate"], candidate_path, "candidate"
        )


def test_int4_record_requires_calibration_provenance(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["quantization"]["calibration"] = None

    with pytest.raises(qualification.QualificationError, match="must be an object"):
        qualification.finalize_record(record)


def test_calibration_rows_cannot_be_reused_as_evaluation_rows(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    reused = dict(record["evaluation"]["dataset"])
    reused.update(split="test[:128]", rows=128, digest="sha256:" + "5" * 64)
    record["quantization"]["calibration"] = reused

    with pytest.raises(qualification.QualificationError, match="same rows"):
        qualification.finalize_record(record)


def test_concatenated_calibration_split_cannot_include_evaluation(
    tmp_path, qualification
):
    record, _, _ = evidence(tmp_path)
    record["quantization"]["calibration"] = dataset("train+test", 128, "5")

    with pytest.raises(qualification.QualificationError, match="same rows"):
        qualification.finalize_record(record)


def test_correct_count_delta_cannot_exceed_changed_answers(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["evaluation"]["quality"].update(candidate_correct=10_000, changed_answers=1)

    with pytest.raises(qualification.QualificationError, match="correct-answer delta"):
        qualification.finalize_record(record)


def test_float8_metadata_matches_the_runtime_vocabulary(tmp_path, qualification):
    record, _, _ = evidence(tmp_path, scheme="float8")
    record["quantization"]["symmetric"] = False

    with pytest.raises(qualification.QualificationError, match="float8 must declare"):
        qualification.finalize_record(record)

    record["quantization"].update(symmetric=True, group_size=False)
    with pytest.raises(qualification.QualificationError, match="float8 must declare"):
        qualification.finalize_record(record)


def test_method_must_match_scheme_and_symmetry(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["quantization"]["method"] = "gptq"

    with pytest.raises(qualification.QualificationError, match="inconsistent"):
        qualification.finalize_record(record)


def test_package_revision_must_be_immutable(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["baseline"]["revision"] = "cbff3a3c"
    record["quantization"]["source"] = dict(record["baseline"])

    with pytest.raises(qualification.QualificationError, match="40-character"):
        qualification.finalize_record(record)


def test_float8_record_explicitly_has_no_calibration_dataset(tmp_path, qualification):
    record, _, _ = evidence(tmp_path, scheme="float8")

    finalized = qualification.finalize_record(record)

    assert finalized["quantization"]["calibration"] is None
    assert finalized["outcome"]["passed"] is True


def test_any_error_nan_oom_or_missing_result_forces_fail(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["evaluation"]["reliability"]["candidate_errors"] = 1

    finalized = qualification.finalize_record(record)

    check = finalized["outcome"]["checks"]["reliability_failures"]
    assert check == {"observed": 1, "maximum": 0, "passed": False}
    assert finalized["outcome"]["passed"] is False


def test_oom_run_can_record_zero_throughput_as_fail(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["evaluation"]["reliability"]["candidate_oom_events"] = 1
    record["evaluation"]["performance"]["throughput_rps"]["16"]["candidate"] = 0

    finalized = qualification.finalize_record(record)

    assert finalized["outcome"]["passed"] is False
    assert finalized["outcome"]["checks"]["reliability_failures"]["observed"] == 1


def test_content_level_calibration_overlap_forces_fail(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["quantization"]["calibration_evaluation_overlap_rows"] = 2

    finalized = qualification.finalize_record(record)

    check = finalized["outcome"]["checks"]["calibration_overlap_rows"]
    assert check == {"observed": 2, "maximum": 0, "passed": False}
    assert finalized["outcome"]["passed"] is False


def test_question_kind_counts_must_sum_to_aggregate(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["evaluation"]["quality"]["by_kind"]["choice"]["rows"] -= 1

    with pytest.raises(qualification.QualificationError, match="sum to aggregate"):
        qualification.finalize_record(record)


def test_question_kind_correct_delta_cannot_exceed_changed_answers(
    tmp_path, qualification
):
    record, _, _ = evidence(tmp_path)
    choice = record["evaluation"]["quality"]["by_kind"]["choice"]
    choice.update(candidate_correct=4_000, changed_answers=1)

    with pytest.raises(qualification.QualificationError, match="choice correct-answer"):
        qualification.finalize_record(record)


def test_choice_changed_answers_must_be_mathematically_possible(
    tmp_path, qualification
):
    record, _, _ = evidence(tmp_path)
    quality = record["evaluation"]["quality"]
    quality["by_kind"]["choice"].update(baseline_correct=6_000, candidate_correct=6_000)
    quality.update(baseline_correct=12_681, candidate_correct=12_668)

    with pytest.raises(
        qualification.QualificationError, match="mathematically impossible"
    ):
        qualification.finalize_record(record)


def test_p95_latency_cannot_be_below_median(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["evaluation"]["performance"]["latency_ms"]["candidate"]["p95"] = 1

    with pytest.raises(qualification.QualificationError, match="p95 latency"):
        qualification.finalize_record(record)


def test_peak_memory_cannot_be_below_loaded_memory(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    record["evaluation"]["memory"]["candidate_peak_bytes"] = 1

    with pytest.raises(qualification.QualificationError, match="below loaded memory"):
        qualification.finalize_record(record)


def test_throughput_requires_concurrency_1_4_and_16(tmp_path, qualification):
    record, _, _ = evidence(tmp_path)
    del record["evaluation"]["performance"]["throughput_rps"]["16"]

    with pytest.raises(qualification.QualificationError, match="keys differ"):
        qualification.finalize_record(record)
