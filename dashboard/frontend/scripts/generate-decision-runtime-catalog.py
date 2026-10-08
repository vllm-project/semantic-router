#!/usr/bin/env python3
"""Project canonical runtime registry metadata into the lightweight dashboard catalog.

No model-runtime dependencies are imported. Run with --check in validation to
reject a stale projection without requiring torch, a GPU, or the model cache.
"""

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "src/model-runtime/vllm_srun"
OUTPUT = ROOT / "dashboard/frontend/src/pages/decisionRuntimeCatalog.generated.json"
BACKEND_OUTPUT = (
    ROOT / "src/semantic-router/pkg/modelservice/decision_catalog.generated.json"
)
RELEASE_OUTPUT = RUNTIME / "registry/releases.generated.json"


def release_projection():
    """Read the runtime's stdlib-only release tables, without loading plugins."""
    sys.path.insert(0, str(RUNTIME.parent))
    try:
        releases = {}
        for path in sorted((RUNTIME / "registry/tables").glob("*.py")):
            if path.stem in {"__init__", "common"}:
                continue
            table = importlib.import_module(f"vllm_srun.registry.tables.{path.stem}")
            for model in table.MODELS:
                if model.repo_id in releases:
                    raise ValueError(f"duplicate release: {model.repo_id}")
                releases[model.repo_id] = model.revision
        return dict(sorted(releases.items()))
    finally:
        sys.path.pop(0)


def assignment(path, name):
    tree = ast.parse(path.read_text())
    return next(
        node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    )


def projection():
    """Read literal release entries and shared System One question kinds."""
    models = []
    organization = ast.literal_eval(
        assignment(RUNTIME / "registry/tables/common.py", "ORG")
    )
    kinds = ast.literal_eval(assignment(RUNTIME / "systemone.py", "QUESTION_TYPES"))
    # Fail closed if the families stop sharing these capabilities. Then the
    # projection must become family-specific before exposing deployment choices.
    decision1_kinds = assignment(RUNTIME / "families/decision1/questions.py", "KINDS")
    if (
        not isinstance(decision1_kinds, ast.Name)
        or decision1_kinds.id != "QUESTION_TYPES"
    ):
        raise ValueError("Decision 1.0 question kinds need an updated projection")
    for family, expected in (("decision1", "KINDS"), ("decision2", None)):
        family_tree = ast.parse((RUNTIME / f"families/{family}/family.py").read_text())
        descriptor = next(
            node
            for node in ast.walk(family_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "descriptor"
        )
        capabilities = next(
            value
            for node in ast.walk(descriptor)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and key.value == "question_types"
        )
        if expected:
            if not (
                isinstance(capabilities, ast.Call)
                and len(capabilities.args) == 1
                and isinstance(capabilities.args[0], ast.Name)
                and capabilities.args[0].id == expected
            ):
                raise ValueError(f"{family} capabilities need an updated projection")
        elif tuple(ast.literal_eval(capabilities)) != tuple(kinds):
            raise ValueError(
                f"{family} capabilities no longer match shared System One kinds"
            )
    for family in ("decision2", "decision1"):
        tree = ast.parse((RUNTIME / f"registry/tables/{family}.py").read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "BuiltinModel":
                continue
            fields = {field.arg: field.value for field in node.keywords}
            name = fields["repo_id"].values[-1].value.lstrip("/")
            models.append(
                {
                    "id": f"{organization}/{name}",
                    "name": name,
                    "provider": organization,
                    "family": ast.literal_eval(fields["family"]),
                    "revision": ast.literal_eval(fields["revision"]),
                    "backbone": ast.literal_eval(fields["backbone"]),
                    "minMemoryGiB": ast.literal_eval(fields["min_device_memory_gib"]),
                }
            )
    return {"questionTypes": kinds, "models": models}


def backend_projection(data):
    """The shared semantic task resolver consumes the same native facts."""
    models = [
        {
            "id": item["id"],
            "family": item["family"],
            "question_types": data["questionTypes"],
        }
        for item in data["models"]
    ]
    kinds_node = assignment(RUNTIME / "families/vela2/request.py", "QUESTION_TYPES")
    kinds = []
    for kind in kinds_node.elts:
        if (
            isinstance(kind, ast.Starred)
            and isinstance(kind.value, ast.Name)
            and kind.value.id == "SYSTEM_ONE_TYPES"
        ):
            kinds.extend(data["questionTypes"])
        elif (
            isinstance(kind, ast.Starred)
            and isinstance(kind.value, ast.Name)
            and kind.value.id == "LABELLED_TYPES"
        ):
            kinds.extend(
                ast.literal_eval(
                    assignment(RUNTIME / "families/vela2/request.py", "LABELLED_TYPES")
                )
            )
        else:
            kinds.append(ast.literal_eval(kind))
    organization = ast.literal_eval(
        assignment(RUNTIME / "registry/tables/common.py", "ORG")
    )
    entries = assignment(RUNTIME / "registry/tables/vela2.py", "MODELS")
    for entry in entries.elts:
        if (
            not isinstance(entry, ast.Call)
            or not isinstance(entry.func, ast.Name)
            or entry.func.id != "_vela"
        ):
            raise ValueError("Vela release entries require an updated task projection")
        size = ast.literal_eval(entry.args[0])
        models.append(
            {
                "id": f"{organization}/Vela-2.0-{size}",
                "family": "vela2",
                "question_types": kinds,
            }
        )
    return {"models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = projection()
    content = json.dumps(data, indent=2) + "\n"
    backend = json.dumps(backend_projection(data), indent=2) + "\n"
    releases = json.dumps(release_projection(), indent=2) + "\n"
    if args.check:
        if (
            not OUTPUT.exists()
            or json.loads(OUTPUT.read_text()) != json.loads(content)
            or not BACKEND_OUTPUT.exists()
            or json.loads(BACKEND_OUTPUT.read_text()) != json.loads(backend)
            or not RELEASE_OUTPUT.exists()
            or json.loads(RELEASE_OUTPUT.read_text()) != json.loads(releases)
        ):
            raise SystemExit(
                "Decision runtime projection is stale; run dashboard/frontend/scripts/generate-decision-runtime-catalog.py"
            )
    else:
        OUTPUT.write_text(content)
        BACKEND_OUTPUT.write_text(backend)
        RELEASE_OUTPUT.write_text(releases)


if __name__ == "__main__":
    main()
