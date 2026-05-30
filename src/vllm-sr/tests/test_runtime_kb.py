from pathlib import Path

import yaml
from cli.commands.runtime_kb import (
    _load_runtime_kb_bootstrap_state,
    _runtime_kb_bootstrap_state_path,
    _write_runtime_kb_bootstrap_state,
)
from cli.commands.runtime_support import resolve_effective_config_path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_resolve_effective_config_path_seeds_relative_kb_assets_into_runtime_store(
    tmp_path: Path,
):
    kb_root = tmp_path / "knowledge_bases" / "privacy"
    kb_root.mkdir(parents=True)
    (kb_root / "labels.json").write_text(
        '{"labels":{"safe":{"exemplars":["hello"]}}}', encoding="utf-8"
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "global": {
                    "model_catalog": {
                        "kbs": [
                            {
                                "name": "privacy_kb",
                                "source": {
                                    "path": "knowledge_bases/privacy/",
                                    "manifest": "labels.json",
                                },
                            }
                        ]
                    }
                },
            },
            sort_keys=False,
        )
    )

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
        platform=None,
    )

    assert effective_path == tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["global"]["model_catalog"]["kbs"][0]["source"]["path"]
        == "knowledge_bases/privacy/"
    )
    assert (
        effective_path.parent / "knowledge_bases" / "privacy" / "labels.json"
    ).exists()


def test_resolve_effective_config_path_seeds_builtin_kb_assets_once(tmp_path: Path):
    test_cases = [
        ("privacy_kb", "knowledge_bases/privacy/", "privacy"),
        ("mmlu_kb", "knowledge_bases/mmlu/", "mmlu"),
    ]

    for kb_name, source_path, bundled_dir in test_cases:
        case_dir = tmp_path / kb_name
        case_dir.mkdir(parents=True, exist_ok=True)
        builtin_manifest = (
            REPO_ROOT / "config" / "knowledge_bases" / bundled_dir / "labels.json"
        )
        assert builtin_manifest.exists()

        config_path = case_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "global": {
                        "model_catalog": {
                            "kbs": [
                                {
                                    "name": kb_name,
                                    "source": {
                                        "path": source_path,
                                        "manifest": "labels.json",
                                    },
                                }
                            ]
                        }
                    },
                },
                sort_keys=False,
            )
        )

        effective_path = resolve_effective_config_path(
            config_path=config_path,
            algorithm=None,
            setup_mode=False,
            platform=None,
        )

        assert effective_path == case_dir / ".vllm-sr" / "runtime-config.yaml"
        staged_manifest = (
            effective_path.parent / "knowledge_bases" / bundled_dir / "labels.json"
        )
        assert staged_manifest.exists()
        effective = yaml.safe_load(effective_path.read_text())
        assert (
            effective["global"]["model_catalog"]["kbs"][0]["source"]["path"]
            == source_path
        )

        staged_manifest.unlink()
        staged_manifest.parent.rmdir()
        effective_path = resolve_effective_config_path(
            config_path=config_path,
            algorithm=None,
            setup_mode=False,
            platform=None,
        )
        assert effective_path == case_dir / ".vllm-sr" / "runtime-config.yaml"
        assert not staged_manifest.exists()


def test_runtime_kb_bootstrap_state_ignores_corrupt_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\n")
    state_path = _runtime_kb_bootstrap_state_path(config_path)
    state_path.write_text("processed: [unterminated", encoding="utf-8")

    assert _load_runtime_kb_bootstrap_state(config_path) == set()


def test_runtime_kb_bootstrap_state_write_replaces_file_without_temp_leftovers(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\n")

    _write_runtime_kb_bootstrap_state(
        config_path,
        {"knowledge_bases/mmlu", "knowledge_bases/privacy"},
    )

    state_path = _runtime_kb_bootstrap_state_path(config_path)
    data = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    assert data == {
        "version": 1,
        "processed": ["knowledge_bases/mmlu", "knowledge_bases/privacy"],
    }
    assert _load_runtime_kb_bootstrap_state(config_path) == {
        "knowledge_bases/mmlu",
        "knowledge_bases/privacy",
    }
    assert list(state_path.parent.glob(f".{state_path.name}.*.tmp")) == []
