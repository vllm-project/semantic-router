"""Read-only final-evaluation gate and exact command planner.

This module never invokes a generator, model, API, scorer, or publisher. It
reads only completed SELECT/CAL lineage, the pre-test freeze declaration,
frozen gold-free CSS prompts, and pinned source files. Commands are printed
for a later, separately authorized evaluation run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.eikos.native import selected_checkpoint as selected_eikos_checkpoint
from training.eikos.published_infer import package_identity as eikos_package_identity
from training.model.calibrate import selected_run
from training.model.calibration import load_calibration
from training.model.infer import checkpoint_fingerprint, load_prompts
from transfer.build import EVALUATION_TASKS

PLAN_VERSION = "decision2-final-evaluation-plan/2"
FREEZE_VERSION = "decision2-pretest-freeze/1"
CSS_PROMPTS_SHA256 = "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6"
CSS_EVALUATION_ITEMS = 6547
ALLOWED_SELECTION_SOURCES = {"train", "select", "cal", "synthetic_dev", "css_pilot"}
EIKOS_ARCHITECTURE = "eikos_semif"
EIKOS_PARITY_PANELS = {
    "dev": ("a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a", 1600),
    "css_pilot": (
        "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
        1430,
    ),
}
PINNED_PROTOCOL_SHA256 = {
    "benchmark/generate.py": "c6569d4c86c3b9ea2ebdda564d8ba4d3428cc807d41d39984379821cfbb9422c",
    "benchmark/score.py": "d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc",
    "benchmark/compare.py": "da15e6ff9efc26a58ee65d36ee0b8f7ff36bd1f9372e97d67f7a3701ef27d5a4",
    "transfer/score.py": "cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca",
    "transfer/compare.py": "17d2264511783c07847bb6d29d6722338a02c92abb825aaa140053e7b4760d46",
    "transfer/normalize_jev.py": "8466f0faa400e90a35253de7986ab256af53f90c649a46ae35db266fe46c19f4",
    "transfer/build.py": "717e31b62d1f9d09b30f1ae2365b2e59cdf0cf40209dda5e9e08b921a1fc184d",
    "publication/generate.py": "a975a1ce9b15a8300d47efe9c6f8bef6d1650ae5599a6789131731c536ce5659",
    "publication/load.py": "08df30c54d4243983c2a8508f74598afd9e469a487be3f390de66d79be0b064d",
    "publication/render.py": "3207b364fa596fd2bc46c927d3bb2e336c89764a72d725a54ef48cf931b4ad15",
    "publication/bundle.py": "b1c1064ffbcfd0fba5a19509d2a5e76ca7935a261d02b8c5d6c8861e5b7c99ed",
    "publication/training_record.py": "310c38f9aeb6eb13f4d10283b40412dcb6189fded8c467dfb1977a800cf714bb",
    "inference/run.py": "b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce",
    "inference/kev.py": "5439c971654a8edc5e14c314b5846fb214d6ffc6dfc6c6f9f2d00e51ac206fc9",
    "inference/laya.py": "64b3782b4d4917fafce9b45544f7602dd599c3394d3b564087fe995d12dc3ea7",
    "inference/this_that.py": "752aaf2ad42abc5f9e7ca364137bd03259676cf6bbd32f350538dd74cfcb5071",
    "inference/kai_lex.py": "6fd22542bc2dd5d385ebdc934ad005fd0c462c40df7303f304ce7ab37085f6df",
    "inference/eikos.py": "818c0d4a153613cd6c84c5b6cb00f96e069be00e438c9dc6d1355723d0e671ca",
    "inference/jevk5.py": "b49b60f61ef43b4278e61d5aeb61019eb6679819809255a3325861023aa24211",
    "clients/jev_api.py": "62887dffa80148cf0030afac3900d111eb891a9125716aad00adebbac44dd68b",
    "training/model/infer.py": "b25382a2667f4a82167d52e35a54dadea6b0838eef407127b23f5dc45b18914b",
    "training/model/calibration.py": "e5161daa82887272d7dd70fd5e99319506b3c25d21c2291a710eda6021c93b15",
    "training/model/calibrate.py": "efc4c164c89a0632056339ac7c95a9cb8c7ef237a04cedea4c9935d4a50b191f",
    "training/model/decision_model.py": "fd9b76cacb1e3598d560122c8b86152b816a0235bd8303496af2effe57ebcae2",
    "training/model/data.py": "632bd60555f63459ff08ffe82c8263ca6a96f75360bc8c06469fcb10f1e2e99c",
    "training/model/lora.py": "7049e35e2a7bf50dd5888902cd0b7030d9bbd448d7d7aba8875ceb89aebff231",
    "training/model/source.py": "ef7b30171c4befb26163ea4d3d6e5add9dc602228a476d6f0ead9d40b450c7e3",
}


@dataclass(frozen=True)
class NativeModel:
    key: str
    label: str
    group: str
    size: str
    model_id: str
    revision: str
    backend: str
    module: str
    model_dir: str = ""
    source_dir: str = ""
    qualification: str = ""


BASELINES = (
    NativeModel(
        "jev",
        "Jev 1.13",
        "hosted",
        "undisclosed",
        "TypeSafe/jev-1.13.0",
        "jev-1.13.0",
        "official-api",
        "clients.jev_api",
        qualification="Official API; server hardware and stochasticity are external.",
    ),
    NativeModel(
        "decider",
        "Decider 4B",
        "open",
        "4B",
        "Mapika/decider-4b",
        "eb5fbdfc9448473ec25e399882912863afbdb70e",
        "decider",
        "inference.run",
        "decider-4b",
        qualification="Native Choice/Noul/Score; eager ROCm path; probabilities rounded to four decimals.",
    ),
    NativeModel(
        "kev",
        "Kev 4B",
        "open",
        "4B",
        "jaredpalmer/kev-4b",
        "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
        "kev",
        "inference.kev",
        "kev-4b",
        "kev-4b-repo",
        "Native typed API; source 6d02f5d; ROCm numeric parity with H200 release unverified; strict 8192-token overflow invalid.",
    ),
    NativeModel(
        "this-that",
        "This-That 1.0",
        "open",
        "2B",
        "flock-io/this-that-model-1.0",
        "3d927195c4f9845efe66c5715883a7a0f42b1239",
        "this-that",
        "inference.this_that",
        "this-that-model-1.0",
        "this-that-model-1.0-source",
        "Native declared-option Choice; Noul/Score are projections, not native heads; source 4efe782c; 1536-token overflow invalid; ROCm unqualified.",
    ),
    NativeModel(
        "laya",
        "Laya Typed",
        "open",
        "0.4B",
        "convaiinnovations/laya-typed-decisions",
        "1a793eb568e6718f15941d08f85432581df534e3",
        "laya",
        "inference.laya",
        "laya-typed-decisions",
        "laya-source",
        "Native typed API; source 4066d5d5; 1024-token context/256-token head truncation counted invalid; ROCm unqualified.",
    ),
    NativeModel(
        "eikos4b",
        "Eikos 4B",
        "open",
        "4B",
        "caiovicentino1/Eikos-4B",
        "582ffb13f19a4da3f455e3db198584190bd7755b",
        "eikos",
        "inference.eikos",
        "Eikos-4B",
        qualification="Native SemIf letter-logit readout and released calibration; one-pass limit 100 choices; pinned SHA256SUMS; BF16 ROCm numeric parity not author-qualified.",
    ),
    NativeModel(
        "jevk5-2b",
        "JevK5 2B",
        "open",
        "2B",
        "alibiserikbay/JevK5-2B",
        "7922d1f55df137b72ef763fced56fd09efc5e99d",
        "jevk5-native-eager",
        "inference.jevk5",
        "JevK5-2B",
        "jevk5-runtime",
        "Native typed API; runtime commit 1e5ae1b; multi-pass for many options; pinned release revision and direct file hashes.",
    ),
    NativeModel(
        "jevk5-4b",
        "JevK5 4B",
        "open",
        "4B",
        "alibiserikbay/JevK5",
        "c4f7fdb3aeab5582336406e78d3bef11bf98833d",
        "jevk5-native-eager",
        "inference.jevk5",
        "JevK5-4B",
        "jevk5-runtime",
        "Native typed API; runtime commit 1e5ae1b; multi-pass for many options; pinned release SHA256SUMS.",
    ),
    NativeModel(
        "jevk5-9b",
        "JevK5 9B",
        "open",
        "9B",
        "alibiserikbay/JevK5-9B",
        "d6521a18a86999190e9d775c915af3d6d6772fc4",
        "jevk5-native-eager",
        "inference.jevk5",
        "JevK5-9B",
        "jevk5-runtime",
        "Native typed API; runtime commit 1e5ae1b; multi-pass for many options; pinned release SHA256SUMS.",
    ),
    NativeModel(
        "eos",
        "Decision 1.0 Eos",
        "decision1",
        "0.8B",
        "llm-semantic-router/Decision-1.0-Eos-0.8B",
        "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
        "eos",
        "inference.run",
        "Decision-1.0-Eos-0.8B",
        qualification="Native typed bundle; requires published qualified ROCm+FLA runtime.",
    ),
    NativeModel(
        "kai",
        "Decision 1.0 Kai",
        "decision1",
        "0.6B",
        "llm-semantic-router/Decision-1.0-Kai-0.6B",
        "7185f514f54b8f93c55998b1e8f9c5cc67f0d029",
        "kai",
        "inference.kai_lex",
        "Decision-1.0-Kai-0.6B",
        qualification="Native typed bundle; strict 1024-token overflow invalid; requires isolated qualified Python/Transformers/tokenizers/NumPy runtime.",
    ),
    NativeModel(
        "lex",
        "Decision 1.0 Lex",
        "decision1",
        "0.6B",
        "llm-semantic-router/Decision-1.0-Lex-0.6B",
        "ee8e74d912fca8328a353c11d174b44da3f91781",
        "lex",
        "inference.kai_lex",
        "Decision-1.0-Lex-0.6B",
        qualification="Native typed bundle; strict 1024-token overflow invalid; requires isolated qualified Python/Transformers/tokenizers/NumPy runtime.",
    ),
    NativeModel(
        "lux",
        "Decision 1.0 Lux",
        "decision1",
        "9B",
        "llm-semantic-router/Decision-1.0-Lux-9B",
        "bd45a30aee8c84032791c245c70f86dee5389cc8",
        "lux",
        "inference.run",
        "Decision-1.0-Lux-9B",
        qualification="Native typed bundle; qualified ROCm+FLA runtime; 16384-token overflow invalid.",
    ),
    NativeModel(
        "nox",
        "Decision 1.0 Nox",
        "decision1",
        "4B",
        "llm-semantic-router/Decision-1.0-Nox-4B",
        "0bb833504965c0eabdb9630b7bbd385cb2fe5cd4",
        "nox",
        "inference.run",
        "Decision-1.0-Nox-4B",
        qualification="Native typed bundle; qualified ROCm+FLA runtime.",
    ),
    NativeModel(
        "sol",
        "Decision 1.0 Sol",
        "decision1",
        "2B",
        "llm-semantic-router/Decision-1.0-Sol-2B",
        "0665a41108e8f0b33a9515c98311c45947b99399",
        "sol",
        "inference.run",
        "Decision-1.0-Sol-2B",
        qualification="Native typed bundle; qualified ROCm+FLA runtime.",
    ),
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def protocol_sources(source_root: Path) -> dict[str, str]:
    observed = {name: sha_file(source_root / name) for name in PINNED_PROTOCOL_SHA256}
    if observed != PINNED_PROTOCOL_SHA256:
        changes = sorted(
            name
            for name, digest in observed.items()
            if digest != PINNED_PROTOCOL_SHA256[name]
        )
        raise ValueError(
            f"Frozen generator/scorer/comparison/publication code differs: {changes}"
        )
    return observed


def frozen_css_prompts(path: Path) -> dict[str, Any]:
    digest = sha_file(path)
    if digest != CSS_PROMPTS_SHA256:
        raise ValueError(
            "CSS evaluation gold-free prompt SHA-256 differs from the frozen panel"
        )
    rows = load_prompts(path)
    if len(rows) != CSS_EVALUATION_ITEMS:
        raise ValueError(
            "CSS evaluation prompt count differs from the frozen 15-task panel"
        )
    return {"sha256": digest, "items": len(rows), "task_count": len(EVALUATION_TASKS)}


def frozen_eikos_candidate(entry: dict[str, Any]) -> dict[str, Any]:
    """Audit a native SemIf package against the selected training run and CAL."""
    required = {
        "key",
        "label",
        "size",
        "model_id",
        "architecture",
        "run_dir",
        "source_path",
        "package_dir",
        "cal_data",
        "calibration",
        "calibration_report",
        "training_data_manifest",
        "parity_reports",
        "selected_checkpoint",
        "model_sha256",
        "calibration_sha256",
        "best_sha256",
        "complete_sha256",
        "provenance_sha256",
    }
    if not required <= set(entry) or set(entry) - required - {
        "rights_attestation",
        "rights_attestation_sha256",
    }:
        raise ValueError("Eikos freeze entry has missing or unknown fields")
    if (
        entry["key"] != "d2-4b"
        or entry["size"] != "4B"
        or entry["model_id"] != "llm-semantic-router/dev-2.0-4b"
    ):
        raise ValueError("Eikos candidate must be the requested dev-2.0-4b identity")
    if any(
        not _is_sha(entry[name])
        for name in (
            "model_sha256",
            "calibration_sha256",
            "best_sha256",
            "complete_sha256",
            "provenance_sha256",
        )
    ):
        raise ValueError("Eikos freeze artifact hashes must be lowercase SHA-256")
    paths = {
        name: Path(entry[name])
        for name in (
            "run_dir",
            "source_path",
            "package_dir",
            "cal_data",
            "calibration",
            "calibration_report",
            "training_data_manifest",
        )
    }
    if any(not path.is_absolute() for path in paths.values()):
        raise ValueError("Eikos freeze paths must be absolute")
    run = paths["run_dir"]
    selected = selected_eikos_checkpoint(run, paths["source_path"])
    if selected["name"] != entry["selected_checkpoint"]:
        raise ValueError("Eikos native SELECT checkpoint changed after freeze")
    if (
        sha_file(run / "NATIVE_BEST.json") != entry["best_sha256"]
        or sha_file(run / "COMPLETE.json") != entry["complete_sha256"]
        or sha_file(run / "provenance.json") != entry["provenance_sha256"]
    ):
        raise ValueError("Eikos native BEST/COMPLETE/provenance changed after freeze")
    package = paths["package_dir"]
    attestation_name = entry.get("rights_attestation")
    rights_attestation = Path(attestation_name) if attestation_name else None
    if rights_attestation is not None:
        if not rights_attestation.is_absolute() or not _is_sha(
            entry.get("rights_attestation_sha256")
        ):
            raise ValueError("Eikos rights attestation path/hash must be frozen")
        if sha_file(rights_attestation) != entry["rights_attestation_sha256"]:
            raise ValueError("Eikos rights attestation changed after freeze")
    elif entry.get("rights_attestation_sha256") is not None:
        raise ValueError("Eikos rights attestation hash has no path")
    identity = eikos_package_identity(
        package, rights_attestation=rights_attestation, require_rights=True
    )
    if (
        identity["model_sha256"] != entry["model_sha256"]
        or identity["calibration_sha256"] != entry["calibration_sha256"]
        or identity["selected_checkpoint"] != selected["name"]
        or identity["rights_attestation_sha256"]
        != entry.get("rights_attestation_sha256")
        or sha_file(paths["calibration"]) != entry["calibration_sha256"]
    ):
        raise ValueError("Eikos package or CAL differs from frozen identity")
    training = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    package_receipt = json.loads(
        (package / "decision2_provenance.json").read_text(encoding="utf-8")
    )
    cal_report = json.loads(paths["calibration_report"].read_text(encoding="utf-8"))
    if (
        package_receipt.get("training_provenance_sha256") != entry["provenance_sha256"]
        or package_receipt.get("adapter_weights_sha256")
        != selected["adapter_weights_sha256"]
        or package_receipt.get("adapter_config_sha256")
        != selected["adapter_config_sha256"]
        or package_receipt.get("training_data_sha256") != training.get("data_sha256")
        or package_receipt.get("training_data_manifest_sha256")
        != sha_file(paths["training_data_manifest"])
        or package_receipt.get("calibration_report_sha256")
        != sha_file(paths["calibration_report"])
        or package_receipt.get("calibration_data_sha256") != sha_file(paths["cal_data"])
        or training.get("data_sha256", {}).get("cal_audited_only")
        != sha_file(paths["cal_data"])
        or cal_report.get("checkpoint") != selected["name"]
        or cal_report.get("adapter_weights_sha256")
        != selected["adapter_weights_sha256"]
        or cal_report.get("calibration_sha256") != entry["calibration_sha256"]
        or cal_report.get("cal_sha256") != sha_file(paths["cal_data"])
    ):
        raise ValueError(
            "Eikos package, data, selection and calibration lineage differ"
        )
    parity_paths = entry["parity_reports"]
    if not isinstance(parity_paths, dict) or set(parity_paths) != set(
        EIKOS_PARITY_PANELS
    ):
        raise ValueError(
            "Eikos freeze needs both predeclared gold-free package parity reports"
        )
    parity_hashes = {}
    for panel, (prompt_sha, count) in EIKOS_PARITY_PANELS.items():
        parity_path = Path(parity_paths[panel])
        if not parity_path.is_absolute():
            raise ValueError("Eikos parity report paths must be absolute")
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        gate = parity.get("predeclared_gate", {})
        if (
            parity.get("inference_variant") != "selected_lora_and_merged_same_process"
            or parity.get("candidate_manifest_sha256") != entry["model_sha256"]
            or parity.get("adapter_weights_sha256")
            != selected["adapter_weights_sha256"]
            or parity.get("calibration_sha256") != entry["calibration_sha256"]
            or parity.get("selected_checkpoint") != selected["name"]
            or parity.get("prompt_sha256") != prompt_sha
            or parity.get("items") != count
            or parity.get("answers") != count
            or parity.get("choice_mismatch_n") != 0
            or parity.get("probability_drift_p99", float("inf")) > 0.005
            or parity.get("probability_drift_max", float("inf")) > 0.02
            or not isinstance(gate, dict)
            or gate.get("pass") is not True
            or gate.get("categorical_mismatches") != 0
            or gate.get("probability_drift_p99_lte") != 0.005
            or gate.get("probability_drift_max_lte") != 0.02
        ):
            raise ValueError(f"Eikos {panel} package failed predeclared parity gate")
        parity_hashes[panel] = sha_file(parity_path)
    return {
        **entry,
        "checkpoint": str(package),
        "max_length": 16000,
        "rights_mode": identity["rights_mode"],
        "cal_data_sha256": sha_file(paths["cal_data"]),
        "package_files_checked": identity["package_files_checked"],
        "parity_report_sha256": parity_hashes,
    }


def frozen_candidates(lock_path: Path) -> tuple[list[dict[str, Any]], str]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or lock.get("freeze_version") != FREEZE_VERSION:
        raise ValueError("Unknown or missing pre-test freeze declaration")
    sources = lock.get("selection_sources")
    if (
        not isinstance(sources, list)
        or not sources
        or len(sources) != len(set(sources))
        or not set(sources) <= ALLOWED_SELECTION_SOURCES
        or "select" not in sources
        or "cal" not in sources
    ):
        raise ValueError(
            "Selection sources must be pre-test TRAIN/SELECT/CAL/DEV/PILOT only"
        )
    entries = lock.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Freeze declaration needs one or more preselected Decision 2.0 candidates"
        )
    candidates = []
    keys: set[str] = set()
    identities: set[str] = set()
    required = {
        "key",
        "label",
        "size",
        "model_id",
        "run_dir",
        "cal_data",
        "calibration",
        "selected_checkpoint",
        "model_sha256",
        "calibration_sha256",
        "best_sha256",
        "complete_sha256",
        "provenance_sha256",
    }
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Candidate freeze entry must be an object")
        if entry.get("architecture") == EIKOS_ARCHITECTURE:
            candidate = frozen_eikos_candidate(entry)
            if candidate["key"] in keys or candidate["model_id"] in identities:
                raise ValueError(
                    "Candidate keys/IDs must be unique predeclared Decision 2.0 identities"
                )
            candidates.append(candidate)
            keys.add(candidate["key"])
            identities.add(candidate["model_id"])
            continue
        if not required <= set(entry) or set(entry) - required - {
            "source_path",
            "architecture",
        }:
            raise ValueError(
                "Candidate freeze entry lacks required identity or has unknown fields"
            )
        if entry.get("architecture", "qwen_dynamic") != "qwen_dynamic":
            raise ValueError("Unknown Decision 2.0 candidate architecture")
        key, model_id = entry["key"], entry["model_id"]
        if (
            not isinstance(key, str)
            or re.fullmatch(r"d2-[a-z0-9-]+", key) is None
            or key in keys
            or not isinstance(model_id, str)
            or not model_id
            or model_id in identities
        ):
            raise ValueError(
                "Candidate keys/IDs must be unique predeclared Decision 2.0 identities"
            )
        size = entry["size"]
        if (
            not isinstance(size, str)
            or re.fullmatch(r"(?:[1-9][0-9]*|0\.[1-9][0-9]*)B", size) is None
            or model_id != f"llm-semantic-router/dev-2.0-{size.lower()}"
        ):
            raise ValueError(
                f"{key}: model ID must use the requested dev-2.0-xxb size name"
            )
        if any(
            not isinstance(entry[name], str) or not entry[name]
            for name in (
                "label",
                "size",
                "run_dir",
                "cal_data",
                "calibration",
                "selected_checkpoint",
            )
        ):
            raise ValueError(
                f"{key}: candidate display name and paths must be nonempty strings"
            )
        if any(
            not _is_sha(entry[name])
            for name in (
                "model_sha256",
                "calibration_sha256",
                "best_sha256",
                "complete_sha256",
                "provenance_sha256",
            )
        ):
            raise ValueError(f"{key}: freeze hashes must be lowercase SHA-256")
        source = Path(entry["source_path"]) if entry.get("source_path") else None
        run_dir, cal_data, calibration = (
            Path(entry[name]) for name in ("run_dir", "cal_data", "calibration")
        )
        if any(
            not path.is_absolute()
            for path in (run_dir, cal_data, calibration, *((source,) if source else ()))
        ):
            raise ValueError(f"{key}: freeze paths must be absolute")
        selected = selected_run(run_dir, cal_data)
        if selected["name"] != entry["selected_checkpoint"] or any(
            selected[f"{name}_sha256"] != entry[f"{name}_sha256"]
            for name in ("best", "complete", "provenance")
        ):
            raise ValueError(
                f"{key}: BEST/COMPLETE/provenance changed after candidate freeze"
            )
        identity = checkpoint_fingerprint(selected["checkpoint"], source)
        if identity["model_sha256"] != entry["model_sha256"]:
            raise ValueError(
                f"{key}: model weights/source/tokenizer changed after freeze"
            )
        if sha_file(calibration) != entry["calibration_sha256"]:
            raise ValueError(f"{key}: CAL temperature artifact changed after freeze")
        _, report = load_calibration(calibration, identity["model_sha256"])
        if (
            report.get("selected_checkpoint") != selected["name"]
            or report.get("cal_sha256") != selected["cal_sha256"]
            or any(
                report.get(f"{name}_sha256") != selected[f"{name}_sha256"]
                for name in ("best", "complete", "provenance")
            )
        ):
            raise ValueError(
                f"{key}: CAL report does not bind the completed selected run"
            )
        max_length = selected["contract"].get("max_length")
        if type(max_length) is not int or max_length < 1:
            raise ValueError(f"{key}: selected training context limit is invalid")
        candidates.append(
            {
                **entry,
                "architecture": "qwen_dynamic",
                "checkpoint": str(selected["checkpoint"]),
                "source_path": str(source) if source else None,
                "max_length": max_length,
                "cal_data_sha256": selected["cal_sha256"],
                "temperature_by_type": report["temperature_by_type"],
            }
        )
        keys.add(key)
        identities.add(model_id)
    return candidates, sha_file(lock_path)


def shell(*parts: Any) -> str:
    return shlex.join(str(part) for part in parts)


def source_command(source_root: Path, *parts: Any) -> str:
    return f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={shlex.quote(str(source_root))} {shell(*parts)}"


def native_command(
    model: NativeModel | dict[str, Any],
    *,
    prompts: Path,
    output: Path,
    model_root: Path,
    external_root: Path,
    source_root: Path,
    python: str,
    kai_lex_python: str,
    fla_path: str,
) -> list[str]:
    env = 'ROCR_VISIBLE_DEVICES="${GPU_ID}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1'
    if isinstance(model, dict):
        if model.get("architecture") == EIKOS_ARCHITECTURE:
            args = [
                python,
                "-m",
                "training.eikos.published_infer",
                "--model-path",
                model["package_dir"],
                "--input",
                prompts,
                "--output",
                output,
                "--model-id",
                model["model_id"],
                "--model-revision",
                model["selected_checkpoint"],
            ]
            if model.get("rights_attestation"):
                args.extend(("--rights-attestation", model["rights_attestation"]))
            return [
                f"{env} PYTHONPATH={shlex.quote(fla_path + ':' + str(source_root))} {shell(*args)}"
            ]
        args = [
            python,
            "-m",
            "training.model.infer",
            "--checkpoint",
            model["checkpoint"],
            "--calibration",
            model["calibration"],
            "--input",
            prompts,
            "--output",
            output,
            "--model-id",
            model["model_id"],
            "--model-revision",
            model["selected_checkpoint"],
            "--max-length",
            model["max_length"],
        ]
        if model["source_path"]:
            args.extend(("--source-path", model["source_path"]))
        return [f"{env} PYTHONPATH={shlex.quote(str(source_root))} {shell(*args)}"]
    if model.key == "jev":
        receipts = output.with_name(
            output.name.replace(".predictions.jsonl", ".receipts.jsonl")
        )
        return [
            f"PYTHONPATH={shlex.quote(str(source_root))} {shell(python, '-m', 'clients.jev_api', '--input', prompts, '--output', receipts, '--model', model.revision)} < \"${{JEV_TOKEN_FILE}}\"",
            f"PYTHONPATH={shlex.quote(str(source_root))} {shell(python, '-m', 'transfer.normalize_jev', '--prompts', prompts, '--receipts', receipts, '--output', output, '--expected-model', model.revision)}",
        ]
    executable = kai_lex_python if model.module == "inference.kai_lex" else python
    args = [executable, "-m", model.module]
    if model.module in ("inference.run", "inference.kai_lex"):
        args.extend(("--backend", model.backend))
    if model.module == "inference.jevk5":
        args.extend(("--size", model.size.lower()))
    args.extend(("--model-path", model_root / model.model_dir))
    if model.module == "inference.jevk5":
        args.extend(("--runtime-path", external_root / model.source_dir))
    elif model.source_dir:
        args.extend(("--source-path", external_root / model.source_dir))
    args.extend(
        ("--model-revision", model.revision, "--input", prompts, "--output", output)
    )
    args.extend(("--device", "cuda:0"))
    pythonpath = (
        fla_path + ":" if model.backend in {"lux", "nox", "sol", "eos"} else ""
    ) + str(source_root)
    return [f"{env} PYTHONPATH={shlex.quote(pythonpath)} {shell(*args)}"]


def build_plan(
    *,
    candidates: list[dict[str, Any]],
    freeze_sha: str,
    css_prompts: Path,
    css_info: dict[str, Any],
    protocol_sha: dict[str, str],
    evaluation_root: Path,
    source_root: Path,
    model_root: Path,
    external_root: Path,
    python: str,
    kai_lex_python: str,
    fla_path: str,
) -> dict[str, Any]:
    final_prompts = evaluation_root / "final.prompts.jsonl"
    final_gold = evaluation_root / "final.gold.jsonl"
    seed = evaluation_root / "final.seed"
    css_gold = css_prompts.with_name("css-evaluation.gold.jsonl")
    models: list[NativeModel | dict[str, Any]] = [*BASELINES, *candidates]
    keys = [
        model.key if isinstance(model, NativeModel) else model["key"]
        for model in models
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate baseline/candidate key")
    model_ids = [
        model.model_id if isinstance(model, NativeModel) else model["model_id"]
        for model in models
    ]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Duplicate baseline/candidate model ID")
    prep = [
        shell("install", "-d", "-m", "700", evaluation_root),
        shell(
            "install",
            "-d",
            "-m",
            "700",
            evaluation_root / "predictions",
            evaluation_root / "reports",
            evaluation_root / "comparisons",
        ),
        shell(
            python,
            "-c",
            "import os,secrets,sys; fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(fd,secrets.token_bytes(32)); os.close(fd)",
            seed,
        ),
        source_command(
            source_root,
            python,
            "-m",
            "benchmark.generate",
            "--split",
            "final",
            "--seed-file",
            seed,
            "--groups-per-family",
            "100",
            "--output",
            final_gold,
            "--prompts-output",
            final_prompts,
        ),
        shell("sha256sum", final_prompts, css_prompts),
    ]
    inference: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    publication_models = []
    prediction_paths = []
    for model in models:
        key = model.key if isinstance(model, NativeModel) else model["key"]
        model_id = (
            model.model_id if isinstance(model, NativeModel) else model["model_id"]
        )
        revision = (
            model.revision
            if isinstance(model, NativeModel)
            else model["selected_checkpoint"]
        )
        backend = (
            model.backend
            if isinstance(model, NativeModel)
            else (
                "eikos-semif-native"
                if model.get("architecture") == EIKOS_ARCHITECTURE
                else "decision2-native-calibrated"
            )
        )
        label = model.label if isinstance(model, NativeModel) else model["label"]
        group = model.group if isinstance(model, NativeModel) else "decision2"
        size = model.size if isinstance(model, NativeModel) else model["size"]
        pred = {
            panel: evaluation_root / "predictions" / f"{key}.{panel}.predictions.jsonl"
            for panel in ("final", "css-evaluation")
        }
        prediction_paths.extend(pred.values())
        if key == "jev":
            prediction_paths.extend(
                path.with_name(
                    path.name.replace(".predictions.jsonl", ".receipts.jsonl")
                )
                for path in pred.values()
            )
        if isinstance(model, dict):
            prediction_paths.extend(
                Path(str(path) + ".manifest.json") for path in pred.values()
            )
        cmds = []
        for panel, prompts in (
            ("final", final_prompts),
            ("css-evaluation", css_prompts),
        ):
            cmds.extend(
                native_command(
                    model,
                    prompts=prompts,
                    output=pred[panel],
                    model_root=model_root,
                    external_root=external_root,
                    source_root=source_root,
                    python=python,
                    kai_lex_python=kai_lex_python,
                    fla_path=fla_path,
                )
            )
        inference.append(
            {
                "key": key,
                "model_id": model_id,
                "revision": revision,
                "backend": backend,
                "qualification": (
                    model.qualification
                    if isinstance(model, NativeModel)
                    else (
                        "Frozen native Eikos SemIf package with independent hard CAL; packaged BF16 parity and ROCm qualification reported separately."
                        if model.get("architecture") == EIKOS_ARCHITECTURE
                        else "Frozen Decision 2.0 checkpoint with independent per-type CAL; no final or CSS evaluation calibration."
                    )
                ),
                "predictions": {panel: str(path) for panel, path in pred.items()},
                "receipts": (
                    {
                        panel: str(
                            path.with_name(
                                path.name.replace(
                                    ".predictions.jsonl", ".receipts.jsonl"
                                )
                            )
                        )
                        for panel, path in pred.items()
                    }
                    if key == "jev"
                    else {}
                ),
                "manifests": (
                    {
                        panel: str(path) + ".manifest.json"
                        for panel, path in pred.items()
                    }
                    if isinstance(model, dict)
                    else {}
                ),
                "commands": cmds,
            }
        )
        final_report = evaluation_root / "reports" / f"{key}.final.score.v2.json"
        css_report = evaluation_root / "reports" / f"{key}.css-evaluation.score.v2.json"
        scoring.append(
            {
                "key": key,
                "reports": {
                    "final": str(final_report),
                    "css-evaluation": str(css_report),
                },
                "commands": [
                    shell("test", "!", "-e", final_report)
                    + " && "
                    + source_command(
                        source_root,
                        python,
                        "-m",
                        "benchmark.score",
                        "--gold",
                        final_gold,
                        "--predictions",
                        pred["final"],
                        "--model-id",
                        model_id,
                        "--model-revision",
                        revision,
                        "--backend",
                        backend,
                        "--output",
                        final_report,
                    ),
                    shell("test", "!", "-e", css_report)
                    + " && "
                    + source_command(
                        source_root,
                        python,
                        "-m",
                        "transfer.score",
                        "--gold",
                        css_gold,
                        "--predictions",
                        pred["css-evaluation"],
                        "--output",
                        css_report,
                    ),
                ],
            }
        )
        publication_models.append(
            {
                "key": key,
                "label": label,
                "group": group,
                "size": size,
                "benchmark_report": str(final_report),
                "css_report": str(css_report),
            }
        )
    for candidate in candidates:
        for baseline in BASELINES:
            key = f"{candidate['key']}-vs-{baseline.key}"
            first = (
                evaluation_root
                / "predictions"
                / f"{candidate['key']}.final.predictions.jsonl"
            )
            second = (
                evaluation_root
                / "predictions"
                / f"{baseline.key}.final.predictions.jsonl"
            )
            css_first = (
                evaluation_root
                / "predictions"
                / f"{candidate['key']}.css-evaluation.predictions.jsonl"
            )
            css_second = (
                evaluation_root
                / "predictions"
                / f"{baseline.key}.css-evaluation.predictions.jsonl"
            )
            benchmark_out = evaluation_root / "comparisons" / f"{key}.final.paired.json"
            css_out = evaluation_root / "comparisons" / f"{key}.css.paired.json"
            pairs.append(
                {
                    "key": key,
                    "candidate": candidate["key"],
                    "baseline": baseline.key,
                    "reports": {
                        "final": str(benchmark_out),
                        "css-evaluation": str(css_out),
                    },
                    "commands": [
                        shell("test", "!", "-e", benchmark_out)
                        + " && "
                        + source_command(
                            source_root,
                            python,
                            "-m",
                            "benchmark.compare",
                            "--gold",
                            final_gold,
                            "--left",
                            first,
                            "--right",
                            second,
                            "--left-name",
                            candidate["model_id"],
                            "--right-name",
                            baseline.model_id,
                            "--iterations",
                            "5000",
                            "--seed",
                            "20260926",
                            "--output",
                            benchmark_out,
                        ),
                        shell("test", "!", "-e", css_out)
                        + " && "
                        + source_command(
                            source_root,
                            python,
                            "-m",
                            "transfer.compare",
                            "--gold",
                            css_gold,
                            "--predictions-a",
                            css_first,
                            "--predictions-b",
                            css_second,
                            "--model-a",
                            candidate["model_id"],
                            "--model-b",
                            baseline.model_id,
                            "--replicates",
                            "5000",
                            "--seed",
                            "20260926",
                            "--output",
                            css_out,
                        ),
                    ],
                }
            )
    publication_pairs = [
        {
            "new": candidate["key"],
            "old": baseline.key,
            "css_comparison_report": str(
                evaluation_root
                / "comparisons"
                / f"{candidate['key']}-vs-{baseline.key}.css.paired.json"
            ),
        }
        for candidate in candidates
        for baseline in BASELINES
    ]
    return {
        "plan_version": PLAN_VERSION,
        "status": "commands_only_not_executed",
        "freeze_manifest_sha256": freeze_sha,
        "frozen_candidates": [
            {
                **{
                    name: candidate[name]
                    for name in (
                        "key",
                        "model_id",
                        "selected_checkpoint",
                        "model_sha256",
                        "calibration_sha256",
                        "best_sha256",
                        "complete_sha256",
                        "provenance_sha256",
                        "max_length",
                    )
                },
                "architecture": candidate.get("architecture", "qwen_dynamic"),
                **(
                    {
                        "parity_reports": candidate["parity_reports"],
                        "parity_report_sha256": candidate["parity_report_sha256"],
                    }
                    if "parity_report_sha256" in candidate
                    else {}
                ),
            }
            for candidate in candidates
        ],
        "css_evaluation_prompts": {"path": str(css_prompts), **css_info},
        "protocol_source_sha256": protocol_sha,
        "planner_source_sha256": sha_file(Path(__file__)),
        "auditor_source_sha256": sha_file(
            Path(__file__).with_name("audit_final_eval.py")
        ),
        "synthetic_final_policy": "Fresh 32-byte private seed after every 2.0 checkpoint/CAL is frozen; final families differ from DEV; no final feedback for selection.",
        "preparation_commands": prep,
        "inference": inference,
        "raw_prediction_hash_command": shell("sha256sum", *prediction_paths)
        + " > "
        + shlex.quote(str(evaluation_root / "RAW_PREDICTIONS.sha256")),
        "scoring": scoring,
        "paired_ci": pairs,
        "publication_config": {
            "title": "Decision 2.0 frozen family-disjoint final evaluation",
            "models": publication_models,
            "comparison_pairs": publication_pairs,
        },
        "publication_config_path": str(evaluation_root / "publication-config.json"),
        "card_artifacts_path": str(evaluation_root / "card-artifacts"),
        "raw_prediction_hashes_path": str(evaluation_root / "RAW_PREDICTIONS.sha256"),
        "publication_command": source_command(
            source_root,
            python,
            "-m",
            "publication.generate",
            "--config",
            evaluation_root / "publication-config.json",
            "--output-dir",
            evaluation_root / "card-artifacts",
        ),
        "required_report_versions": {
            "synthetic_final": "typed-decision-report/2",
            "css_evaluation": "css-transfer-score/2",
            "card_artifacts": "decision-model-card-artifacts/2",
        },
        "audit_rules": [
            "Freeze all 2.0 candidates and per-type CAL before executing preparation_commands; never use either final panel for selection, calibration, prompt changes, or threshold tuning.",
            "Collect one pinned native model per process; do not use allow-unvalidated-runtime for qualified 1.0 bundles; preserve explicit invalid/overflow rows.",
            "Hash every original prediction before scoring and verify every v2 report predictions_sha256 matches those bytes. Jev receipts and normalized predictions need separate hashes.",
            "Require every final report to have the same new gold digest and all four final families; require CSS report exactly the 15 evaluation tasks and 6547-item denominator.",
            "Pair 2.0 versus each baseline on identical item IDs; synthetic CI resamples four-variant groups within family, CSS CI resamples items within each of 15 tasks.",
            "Use publication.generate only after all v2 reports and paired CSS comparison reports pass; inspect rank/matrix manifest SHA-256 before any model-card publication.",
        ],
    }


def markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Locked final evaluation command plan",
        "",
        f"Plan `{plan['plan_version']}`; commands have not been executed. Freeze SHA-256: `{plan['freeze_manifest_sha256']}`.",
        "",
        f"CSS 15-task gold-free prompts: `{plan['css_evaluation_prompts']['sha256']}` ({plan['css_evaluation_prompts']['items']} items).",
        "",
        "## Required gates",
        "",
    ]
    lines.extend(f"- {rule}" for rule in plan["audit_rules"])
    lines += [
        "",
        "## 1. Generate fresh family-disjoint synthetic final after freeze",
        "",
        "```bash",
        *plan["preparation_commands"],
        "```",
        "",
        "## 2. Native inference (schedule GPU/API calls separately)",
        "",
    ]
    for model in plan["inference"]:
        lines += [
            f"### {model['key']} — {model['model_id']} @ {model['revision']}",
            "",
            model["qualification"],
            "",
            "```bash",
            *model["commands"],
            "```",
            "",
        ]
    lines += [
        "## 3. Freeze raw prediction hashes before scoring",
        "",
        "```bash",
        plan["raw_prediction_hash_command"],
        "```",
        "",
        "## 4. Score with exact v2 scorer sources",
        "",
    ]
    for model in plan["scoring"]:
        lines += [f"### {model['key']}", "", "```bash", *model["commands"], "```", ""]
    lines += ["## 5. Paired confidence intervals", ""]
    for pair in plan["paired_ci"]:
        lines += [f"### {pair['key']}", "", "```bash", *pair["commands"], "```", ""]
    lines += [
        "## 6. Rank, matrix, and score table",
        "",
        "Write the `publication_config` object from the JSON plan to the declared config path, then run:",
        "",
        "```bash",
        plan["publication_command"],
        "```",
        "",
        "Publication is a separate later decision. The CSS panel measures Choice only; This-That Noul/Score are generic-option projections, and smaller models' overflow/unsupported outputs remain invalid in the denominator.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--css-prompts", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--kai-lex-python", default="/work/envs/kai-lex/bin/python")
    parser.add_argument("--fla-path", default="/opt/decision-fla")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()
    for name in (
        "freeze_manifest",
        "css_prompts",
        "source_root",
        "evaluation_root",
        "model_root",
        "external_root",
    ):
        if not getattr(args, name).is_absolute():
            parser.error(f"--{name.replace('_', '-')} must be an absolute path")
    for name in ("kai_lex_python", "fla_path"):
        if not Path(getattr(args, name)).is_absolute():
            parser.error(f"--{name.replace('_', '-')} must be an absolute path")
    if args.evaluation_root.exists():
        raise FileExistsError(
            "Final evaluation output directory already exists; use a fresh path"
        )
    protocol = protocol_sources(args.source_root)
    css = frozen_css_prompts(args.css_prompts)
    candidates, freeze_sha = frozen_candidates(args.freeze_manifest)
    plan = build_plan(
        candidates=candidates,
        freeze_sha=freeze_sha,
        css_prompts=args.css_prompts,
        css_info=css,
        protocol_sha=protocol,
        evaluation_root=args.evaluation_root,
        source_root=args.source_root,
        model_root=args.model_root,
        external_root=args.external_root,
        python=args.python,
        kai_lex_python=args.kai_lex_python,
        fla_path=args.fla_path,
    )
    plan["freeze_manifest_path"] = str(args.freeze_manifest)
    plan["source_root"] = str(args.source_root)
    print(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else markdown(plan)
    )


if __name__ == "__main__":
    main()
