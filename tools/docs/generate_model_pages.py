#!/usr/bin/env python3
"""Generate the website's Supported models pages from the built-in model catalog.

Every fact comes from the sources the Router and the runtime use: the Router's
decision catalog (question types, input modalities, span heads), the runtime's
registry tables (revisions, parameters, backbones, memory, base packages,
reduced copies), the recorded reference answers, the Router's calibrated
thresholds and trained families, and the families that load quantized
checkpoints. tools/docs/supported_models.yaml holds the editorial text. Nothing
from the runtime is imported: the tables are read with ast, as the Dashboard's
decision runtime catalog reads them, so --check needs no torch and no GPU.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "src/semantic-router/pkg/config/decision_catalog.generated.json"
RUNTIME = ROOT / "src/model-runtime/vllm_srun"
RECORDS = ROOT / "src/model-runtime/docs/records"
PRESETS = ROOT / "src/semantic-router/pkg/presets"
EDITORIAL = ROOT / "tools/docs/supported_models.yaml"
PAGES = ROOT / "website/docs/model-runtime/supported-models"
ZH_PAGES = (
    ROOT
    / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/model-runtime/supported-models"
)
RECORDS_URL = "https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records"
GENERATED_BY = "tools/docs/generate_model_pages.py"
BEGIN = "<!-- BEGIN GENERATED MODEL TABLE -->"
END = "<!-- END GENERATED MODEL TABLE -->"

# The precision ``--dtype auto`` runs per family and backbone, on a GPU and on
# a CPU (each family's ``DtypePolicy``; website engine-args "Precision").
PRECISION = {
    ("decision3", "qwen3_5_text"): ("BF16", "BF16"),
    ("vela3_d3", "qwen3_5_text"): ("BF16", "BF16"),
    ("decision2", "qwen3"): ("BF16", "FP32"),
    ("decision2", "qwen3_5_text"): ("BF16", "FP32"),
    ("decision1", "qwen3_5_text"): ("BF16", "FP32"),
    ("decision1", "modernbert"): ("FP32", "FP32"),
    ("vela2", "qwen3_5_text"): ("BF16", "FP32"),
    ("vela2", "modernbert"): ("FP32", "FP32"),
    ("vela3", "ultrabert"): ("FP32", "FP32"),
}
BYTES = {"BF16": 2, "FP32": 4}
BACKBONES = {
    "qwen3": "Qwen3",
    "qwen3_5_text": "Qwen3.5 text",
    "modernbert": "ModernBERT",
    "ultrabert": "UltraBERT",
}
# The device-class and hardware keys of recorded reference answers
# (``Accelerator.golden_reference``).
DEVICES = {
    "cpu": "CPU",
    "cpu:x86_64:avx2:bf16_emulated": "x86-64 CPU with AVX2 (BF16 emulated)",
    "cpu:x86_64:avx512:bf16_native": "x86-64 CPU with AVX-512 BF16",
    "rocm": "AMD GPU (ROCm)",
    "rocm:gfx942": "AMD Instinct MI300X or MI325X (gfx942)",
    "cuda": "NVIDIA GPU (CUDA)",
    "cuda:sm_89": "NVIDIA Ada (sm_89)",
    "cuda:sm_90": "NVIDIA Hopper (sm_90)",
    "npu": "Ascend NPU",
    "npu:Ascend950DT": "Ascend 950DT",
}
CLASSES = {"cpu": "a CPU", "rocm": "an AMD GPU", "cuda": "an NVIDIA GPU"}
REDUCED = {
    "float32-packed": "a pre-packed FP32 copy of its linear layers on x86 CPUs with oneDNN",
    "bfloat16": "a BF16 copy of its linear layers",
    "int8": "a dynamic int8 copy of its linear layers on x86 CPUs",
}
# with_recorded(models, file, entry field, model field) in a registry table.
WITH_RECORDED_ARGS = 4
# Backbones whose models the docs show on a CPU first: the encoders and Qwen3
# (Decision 2.0 Kai-0.6B); the Qwen3.5 decoders are shown on a GPU first.
CPU_FIRST_BACKBONES = frozenset({"modernbert", "ultrabert", "qwen3"})
# The compressed-tensors formats quantization.parse accepts.
QUANTIZED_FORMATS = ("pack-quantized", "float-quantized", "nvfp4-pack-quantized")
PRESET_NAMES = {
    "jailbreak": "jailbreak",
    "domain": "domain",
    "pii": "pii",
    "fact_check": "fact check",
    "feedback": "user feedback",
    "harm": "safety",
    "modality": "modality",
}
ALL_TYPES = ("choice", "noul", "score", "set", "span")
TABLE_HEADERS = {
    "en": ("Model", "Parameters", "Answers", "Spans", "Reads", "Device memory"),
    "zh-Hans": ("模型", "参数量", "回答的问题类型", "Span", "输入", "显存"),
}


@dataclass
class Model:
    """One catalog model and the facts its page shows."""

    repo: str
    family: str
    question_types: list[str]
    modalities: list[str]
    span_heads: list[str]
    revision: str
    parameters: int
    backbone: str
    memory: float
    base: tuple[str, str] | None
    reduced: dict[str, str]
    references: list[str]
    calibration: dict[str, float]
    editorial: dict
    family_dir: str
    family_label: str
    trained: bool
    quantizable: bool
    position: int = 0
    records: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.repo.split("/", 1)[1]

    @property
    def page(self) -> str:
        return self.editorial["page"]

    @property
    def label(self) -> str:
        return re.sub(r"^Decision-\d\.\d-", "", self.name)

    @property
    def family_doc(self) -> str:
        # Docs-root links resolve in every locale, also from this English-only page.
        return f"model-runtime/supported-models/{self.family_dir}/overview.md"

    @property
    def precision(self) -> tuple[str, str]:
        return PRECISION[(self.family, self.backbone)]


def fail(message: str) -> None:
    raise SystemExit(f"{GENERATED_BY}: {message}")


def assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.value
    fail(f"no {name} assignment")
    raise AssertionError


def evaluate(node: ast.expr, bindings: dict):
    """A literal, a bound name, an f-string of bound names, or ``a or b``."""
    if isinstance(node, ast.Name):
        if node.id not in bindings:
            fail(f"unbound name {node.id} in a registry table")
        return bindings[node.id]
    if isinstance(node, ast.JoinedStr):
        return "".join(
            (
                str(evaluate(part.value, bindings))
                if isinstance(part, ast.FormattedValue)
                else part.value
            )
            for part in node.values
        )
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        value = None
        for operand in node.values:
            value = evaluate(operand, bindings)
            if value:
                return value
        return value
    return ast.literal_eval(node)


def registry_entries(family: str, organization: str) -> dict[str, dict]:
    """The pinned entries of one family table, by repository."""
    path = RUNTIME / f"registry/tables/{family}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    factories = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    models = assignment(tree, "MODELS")
    while isinstance(models, ast.Name):
        models = assignment(tree, models.id)
    if not isinstance(models, (ast.Tuple, ast.List)):
        fail(f"{path.name}: MODELS is not a literal tuple")
    entries = {}
    for call in models.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            fail(f"{path.name}: an entry is not a call")
        outer = {"ORG": organization}
        if call.func.id == "BuiltinModel":
            bindings, constructor = outer, call
        else:
            factory = factories[call.func.id]
            params = factory.args.args
            defaults = factory.args.defaults
            bindings = dict(outer)
            for param, default in zip(
                params[len(params) - len(defaults) :], defaults, strict=True
            ):
                bindings[param.arg] = evaluate(default, outer)
            for param, value in zip(params, call.args, strict=False):
                bindings[param.arg] = evaluate(value, outer)
            for keyword in call.keywords:
                bindings[keyword.arg] = evaluate(keyword.value, outer)
            constructor = next(
                n.value for n in factory.body if isinstance(n, ast.Return)
            )
        fields = {k.arg: k.value for k in constructor.keywords}
        entry = {
            name: evaluate(fields[name], bindings)
            for name in (
                "repo_id",
                "revision",
                "family",
                "loaded_parameters",
                "backbone",
                "min_device_memory_gib",
            )
        }
        entry["base"] = evaluate(fields["base"], bindings) if "base" in fields else None
        entry["reduced"] = (
            evaluate(fields["reduced"], bindings) if "reduced" in fields else {}
        ) or {}
        entries[entry["repo_id"]] = entry
    return entries


def recorded_references(family: str) -> dict[str, list[str]]:
    """The device keys of the reference answers recorded for each pinned revision."""
    path = RUNTIME / f"registry/tables/{family}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    files = [
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "with_recorded"
        and len(node.args) == WITH_RECORDED_ARGS
        and ast.literal_eval(node.args[3]) == "golden_answers"
    ]
    references: dict[str, list[str]] = {}
    for name in files:
        recorded = json.loads((RUNTIME / "registry" / name).read_text(encoding="utf-8"))
        for repo, entry in recorded.items():
            references[repo] = {
                "revision": entry["revision"],
                "keys": sorted(entry["answers"]),
            }
    return references


def go_constants(path: Path) -> dict[str, str]:
    return dict(
        re.findall(r'^\s*(\w+)\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.M)
    )


def calibrated_thresholds() -> dict[str, dict[str, float]]:
    """The Router's calibrated preset thresholds by served model (presets/calibration.go)."""
    constants = go_constants(PRESETS / "presets.go")
    source = (PRESETS / "calibration.go").read_text(encoding="utf-8")
    block = re.search(
        r"var calibration = map\[string\]map\[string\]float64\{\n(.*?)\n\}",
        source,
        re.S,
    )
    if not block:
        fail("presets/calibration.go changed shape; update calibrated_thresholds")
    thresholds = {}
    for model, body in re.findall(
        r'^\s*"([^"]+)":\s*\{([^}]*)\},?\s*$', block.group(1), re.M
    ):
        values = {}
        for constant, value in re.findall(r"(\w+):\s*([0-9.]+)", body):
            if constant not in constants:
                fail(f"calibration.go names an unknown preset constant {constant}")
            values[constants[constant]] = float(value)
        thresholds[model] = values
    if not thresholds:
        fail("presets/calibration.go lists no calibrated model")
    return thresholds


def trained_families() -> set[str]:
    """The families with trained bindings for the built-in questions (presets/bindings.go)."""
    source = (PRESETS / "bindings.go").read_text(encoding="utf-8")
    families = set(re.findall(r'^\s*Family\w+\s*=\s*"([^"]+)"', source, re.M))
    if not families:
        fail("presets/bindings.go declares no trained family")
    return families


def quantizing_families(families: set[str]) -> set[str]:
    """The families whose packages may hold quantized weights (quantization.parse)."""
    formats = (RUNTIME / "quantization.py").read_text(encoding="utf-8")
    for name in QUANTIZED_FORMATS:
        if f'"{name}"' not in formats:
            fail(
                f"quantization.py no longer accepts {name}; update the quantization text"
            )
    return {
        family
        for family in families
        if "quantization.parse("
        in (RUNTIME / f"families/{family}/family.py").read_text(encoding="utf-8")
    }


def load_models() -> list[Model]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["models"]
    editorial = yaml.safe_load(EDITORIAL.read_text(encoding="utf-8"))
    families = {item["family"] for item in catalog}
    unknown = families - set(editorial["families"])
    if unknown:
        fail(f"{EDITORIAL.name} has no family entry for {sorted(unknown)}")
    organization = ast.literal_eval(
        assignment(
            ast.parse((RUNTIME / "registry/tables/common.py").read_text()), "ORG"
        )
    )
    registry = {}
    references = {}
    for family in sorted(families):
        registry.update(registry_entries(family, organization))
        references.update(recorded_references(family))
    calibration = calibrated_thresholds()
    trained = trained_families()
    quantizable = quantizing_families(families)
    catalog_repos = {item["id"] for item in catalog}
    stale = set(editorial["models"]) - catalog_repos
    if stale:
        fail(
            f"{EDITORIAL.name} describes models the catalog does not list: {sorted(stale)}"
        )
    unknown_calibration = set(calibration) - catalog_repos
    if unknown_calibration:
        fail(
            f"calibrated thresholds for models outside the catalog: {sorted(unknown_calibration)}"
        )
    models = []
    for item in catalog:
        repo = item["id"]
        if repo not in editorial["models"]:
            fail(f"{EDITORIAL.name} has no entry for {repo}; add its summary")
        if repo not in registry:
            fail(f"{repo} is in the Router catalog but in no registry table")
        entry = registry[repo]
        if entry["family"] != item["family"]:
            fail(
                f"{repo}: the catalog says {item['family']}, the registry {entry['family']}"
            )
        if (entry["family"], entry["backbone"]) not in PRECISION:
            fail(f"{repo}: no precision for {entry['family']} on {entry['backbone']}")
        if entry["backbone"] not in BACKBONES:
            fail(f"{repo}: no display name for backbone {entry['backbone']}")
        recorded = references.get(repo)
        keys = (
            recorded["keys"]
            if recorded and recorded["revision"] == entry["revision"]
            else []
        )
        for key in keys:
            if key not in DEVICES:
                fail(f"{repo}: no label for reference device {key}")
        for kind in entry["reduced"].values():
            if kind not in REDUCED:
                fail(f"{repo}: no text for reduced copy {kind}")
        text = editorial["models"][repo]
        for record in text.get("records", []):
            if not (RECORDS / record).is_file():
                fail(f"{repo}: record {record} does not exist")
        family = editorial["families"][item["family"]]
        models.append(
            Model(
                repo=repo,
                family=item["family"],
                question_types=item["question_types"],
                modalities=item["modalities"],
                span_heads=item.get("span_heads", []),
                revision=entry["revision"],
                parameters=entry["loaded_parameters"],
                backbone=entry["backbone"],
                memory=entry["min_device_memory_gib"],
                base=tuple(entry["base"]) if entry["base"] else None,
                reduced=entry["reduced"],
                references=keys,
                calibration=calibration.get(repo, {}),
                editorial=text,
                family_dir=family["dir"],
                family_label=family["label"],
                trained=item["family"] in trained,
                quantizable=item["family"] in quantizable,
                records=text.get("records", []),
            )
        )
    by_dir: dict[str, list[Model]] = {}
    for model in models:
        by_dir.setdefault(model.family_dir, []).append(model)
    for group in by_dir.values():
        pages = [m.page for m in group]
        if len(set(pages)) != len(pages):
            fail(f"duplicate page names in {group[0].family_dir}: {pages}")
        order = list(editorial["families"])
        ranked = sorted(group, key=lambda m: (order.index(m.family), m.parameters))
        for rank, model in enumerate(ranked, start=2):
            model.position = rank
    return models


def billions(parameters: int) -> str:
    return f"{parameters / 1e9:.3g}B"


def gigabytes(parameters: int, precision: str) -> str:
    return f"{parameters * BYTES[precision] / 1e9:.1f} GB"


def memory(value: float) -> str:
    return f"{value:g} GiB"


def yes_no(model: Model, kind: str) -> str:
    if kind in model.question_types:
        if kind == "span":
            heads = ", ".join(f"`{h}`" for h in model.span_heads) or "its"
            return f"Yes, on the {heads} span head" + (
                "s" if len(model.span_heads) > 1 else ""
            )
        return "Yes"
    if kind == "set" and "noul" in model.question_types:
        return "Yes, asked as one `noul` per label"
    if kind == "span":
        return "No: bind span questions, such as PII and hallucination, to a span-capable model"
    return "No"


def front_matter(model: Model) -> str:
    summary = " ".join(model.editorial["summary"].split())
    description = summary.split(". ")[0].rstrip(".") + "."
    return (
        "---\n"
        f"title: {json.dumps(model.name)}\n"
        f"sidebar_label: {json.dumps(model.label)}\n"
        f"sidebar_position: {model.position}\n"
        f"description: {json.dumps(description)}\n"
        "custom_edit_url: null\n"
        "---\n"
    )


def deploy_section(model: Model) -> list[str]:
    gpu_first = model.backbone not in CPU_FIRST_BACKBONES
    tabs = [
        ("cpu", "CPU", "cpu"),
        ("rocm", "AMD GPU", "rocm"),
        ("cuda", "NVIDIA GPU", "cuda"),
    ]
    default = "rocm" if gpu_first else "cpu"
    lines = [
        "## Deploy it",
        "",
        "Serve it on its own in Engine mode, to ask it directly:",
        "",
        '<Tabs groupId="platform">',
    ]
    for value, label, platform in tabs:
        lines += [
            f'<TabItem value="{value}" label="{label}"{" default" if value == default else ""}>',
            "",
            "```bash",
            f"vllm-sr serve {model.repo} --engine --platform {platform}",
            "```",
            "",
        ]
        if value == "cpu" and model.backbone == "qwen3_5_text":
            lines += [
                "Serve it on a CPU from the CPU image, as `--platform cpu` does: the ROCm",
                "image's PyTorch has no CPU LAPACK, which this backbone needs there.",
                "",
            ]
        lines += ["</TabItem>"]
    lines += ["</Tabs>", "", "Or make it the Router's decision model:", ""]
    yaml_lines = [
        "global:",
        "  model_catalog:",
        "    deployments:",
        "      primary:",
        "        provider: model_runtime",
        f"        artifact: {model.repo}",
        "        device: auto",
    ]
    if "span" not in model.question_types:
        yaml_lines += [
            "      spans:",
            "        provider: model_runtime",
            "        artifact: vllm-sr/vela3-fast",
            "    bindings:",
            "      pii: spans",
            "      hallucination: spans",
        ]
    yaml_lines += ["    system:", "      decision_model: {deployment: primary}"]
    lines += ['```yaml title="config.yaml"', *yaml_lines, "```", ""]
    lines += [
        "`device: auto` takes the first validated GPU with enough free memory, else",
        "the CPU; name `rocm:0` or `cuda:0` to pin a GPU.",
    ]
    if "span" not in model.question_types:
        lines += [
            "",
            "This model answers no span questions, so the example keeps `pii` and",
            "`hallucination` on a span-capable deployment; without those bindings a",
            "`pii` rule fails to load with `span_required`.",
        ]
    lines += [
        "",
        "To run it on another host and attach the Router to it:",
        "",
        "```bash",
        f"vllm-srun serve {model.repo} --device auto --host 0.0.0.0 --port 8100",
        "```",
        "",
        "Then set `endpoint: http://<host>:8100` and",
        f"`model: {model.name}` on the deployment; see",
        "[Attach to a runtime you run](model-runtime/deploy/connect-the-router.md#attach-to-a-runtime-you-run).",
        "",
    ]
    return lines


def render_page(model: Model) -> str:
    gpu, cpu = model.precision
    summary = " ".join(model.editorial["summary"].split())
    lines = [
        front_matter(model),
        f"<!-- Generated by {GENERATED_BY} from the built-in model catalog. Do not edit directly. -->",
        "",
        "import Tabs from '@theme/Tabs'",
        "import TabItem from '@theme/TabItem'",
        "",
        f"# {model.name}",
        "",
        summary,
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Repository | [`{model.repo}`](https://huggingface.co/{model.repo}), pinned at `{model.revision[:12]}` |",
        f"| Family | [{model.family_label}]({model.family_doc}) |",
        f"| Parameters | {billions(model.parameters)} loaded |",
        f"| Backbone | {BACKBONES[model.backbone]} |",
    ]
    if model.base:
        lines.append(
            f"| Base package | [`{model.base[0]}`](https://huggingface.co/{model.base[0]}) at `{model.base[1][:12]}` |"
        )
    lines += [
        "| Precision (`--dtype auto`) | "
        + (
            f"{gpu} on every device"
            if gpu == cpu
            else f"{gpu} on a GPU, {cpu} on a CPU"
        )
        + " |",
        f"| Weights | about {gigabytes(model.parameters, gpu)} at {gpu}"
        + (f", {gigabytes(model.parameters, cpu)} at {cpu}" if cpu != gpu else "")
        + " |",
        f"| Device memory | {memory(model.memory)} or more |",
        f"| Reads | {', '.join(model.modalities)} |",
        "",
        "## Question types",
        "",
        "| Type | Answers |",
        "| --- | --- |",
    ]
    lines += [f"| `{kind}` | {yes_no(model, kind)} |" for kind in ALL_TYPES]
    api_types = model.editorial.get("api_question_types", [])
    if api_types:
        lines.append(
            f"| {', '.join(f'`{t}`' for t in api_types)} | On the [System One API](system-one/use/api.md) only; routing rules do not ask them |"
        )
    lines += [""]
    if "image" in model.modalities:
        lines += [
            "Rules that read the turn's images (`input.image`) or videos (`input.video`) can",
            "resolve to it.",
            "",
        ]
    lines += ["## Built-in questions", ""]
    if model.trained:
        lines += [
            f"The {model.family_label} family was trained on the Router's built-in",
            "questions, so the Router asks each one in its trained form; the",
            "[support matrix](system-one/reference/support-matrix.md) lists which.",
        ]
    else:
        lines += [
            f"The {model.family_label} family answers the Router's built-in questions",
            "zero-shot, from their canonical wording; the",
            "[support matrix](system-one/reference/support-matrix.md) shows what that",
            "covers.",
        ]
    lines += [""]
    if model.calibration:
        lines += [
            "A rule without a `predicate` takes this model's calibrated threshold:",
            "",
            "| Preset | Threshold |",
            "| --- | ---: |",
        ]
        lines += [
            f"| {PRESET_NAMES.get(preset, preset)} (`{preset}`) | {value:g} |"
            for preset, value in model.calibration.items()
        ]
        lines += [""]
    else:
        lines += [
            ":::note",
            "This model has no calibrated thresholds: a built-in question without a",
            "`predicate` uses 0.5, and the configuration load warns with",
            "`uncalibrated_preset`.",
            ":::",
            "",
        ]
    lines += ["## Hardware", ""]
    if model.references:
        lines += [
            "Each time it loads, the model compares its answers with the reference",
            "answers recorded for its revision. They are recorded for:",
            "",
        ]
        lines += [f"- {DEVICES[key]} (`{key}`)" for key in model.references]
        classes = {key.split(":", 1)[0] for key in model.references}
        missing = [CLASSES[c] for c in CLASSES if c not in classes]
        if any(":" in key for key in model.references):
            lines += [
                "",
                "Where a hardware reference and its device class's are both recorded, the",
                "hardware one is checked.",
            ]
        if missing:
            lines += [
                "",
                f"On {' or '.join(missing)} no reference is recorded, so the model serves",
                "there with its self-check reported as `unverified`.",
            ]
        lines += [""]
    else:
        lines += [
            "No reference answers are recorded for this revision in the registry, so its",
            "self-check reports `unverified`; its records compare it with the package's",
            "own runtime.",
        ]
    lines += [
        "[Choose a model](model-runtime/get-started/choose-a-model.md#hardware) lists the",
        "validated hardware.",
        "",
        "## Quantization",
        "",
    ]
    if model.quantizable:
        lines += [
            f"The {model.family_label} family loads quantized packages in the",
            "compressed-tensors format: FP8 (W8A8), int4 (W4A16, GPTQ or AWQ) and NVFP4",
            "(W4A4). A quantized package is its own model, with its own answers.",
        ]
        packages = model.editorial.get("quantized_packages", [])
        if packages:
            lines += [
                "",
                "Published packages of this model:",
                "",
                *[f"- [`{p}`](https://huggingface.co/{p})" for p in packages],
            ]
        lines += [
            "",
            "[Quantization](model-runtime/quantization/overview.md) compares the formats and",
            "the hardware each needs.",
        ]
    else:
        lines += [
            f"The runtime loads no quantized packages of the {model.family_label} family:",
            "it runs at its published precision.",
        ]
        if model.reduced:
            copies = "; ".join(REDUCED[kind] for kind in model.reduced.values())
            lines += [
                f"Batches that merge requests run {copies}, which its records show changes",
                "no decision.",
            ]
    lines += [""]
    lines += deploy_section(model)
    lines += ["## Learn more", ""]
    lines += [f"- [Model card](https://huggingface.co/{model.repo}) on Hugging Face"]
    lines += [
        f"- [{record.removesuffix('.md')}]({RECORDS_URL}/{record}) record"
        for record in model.records
    ]
    lines += [
        f"- [{model.family_label}]({model.family_doc}): the family's sizes and how they differ"
    ]
    return "\n".join(lines).rstrip() + "\n"


def table(models: list[Model], locale: str, prefix: str) -> str:
    headers = TABLE_HEADERS[locale]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for model in sorted(models, key=lambda m: (m.family_dir, m.position)):
        spans = ", ".join(model.span_heads) if "span" in model.question_types else "—"
        rows.append(
            "| "
            + " | ".join(
                [
                    f"[{model.name}]({prefix}{model.page})",
                    billions(model.parameters),
                    ", ".join(model.question_types),
                    spans,
                    ", ".join(model.modalities),
                    memory(model.memory),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def splice(path: Path, content: str) -> str:
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        fail(f"{path} has no generated table markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n\n{content}\n\n{END}{tail}"


def outputs(models: list[Model]) -> dict[Path, str]:
    files: dict[Path, str] = {}
    by_dir: dict[str, list[Model]] = {}
    for model in models:
        files[PAGES / model.family_dir / f"{model.page}.md"] = render_page(model)
        by_dir.setdefault(model.family_dir, []).append(model)
    for locale, root in (("en", PAGES), ("zh-Hans", ZH_PAGES)):
        overview = root / "overview.md"
        if overview.is_file():
            files[overview] = splice(overview, family_tables(by_dir, locale))
        for directory, group in by_dir.items():
            page = root / directory / "overview.md"
            if page.is_file():
                files[page] = splice(page, table(group, locale, "./"))
    return files


def family_tables(by_dir: dict[str, list[Model]], locale: str) -> str:
    order = sorted(by_dir, key=_family_order)
    parts = []
    for directory in order:
        label = by_dir[directory][0].family_label
        parts.append(
            f"### {label}\n\n" + table(by_dir[directory], locale, f"./{directory}/")
        )
    return "\n\n".join(parts)


def _family_order(directory: str) -> int:
    metadata = json.loads(
        (PAGES / directory / "_category_.json").read_text(encoding="utf-8")
    )
    return metadata["position"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail on drift without writing"
    )
    args = parser.parse_args(argv)
    models = load_models()
    expected = outputs(models)
    generated = {
        path
        for path in PAGES.glob("*/*.md")
        if path.name != "overview.md"
        and GENERATED_BY in path.read_text(encoding="utf-8")
    }
    stale = sorted(generated - set(expected))
    drift = [
        path
        for path, content in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if args.check:
        if drift or stale:
            for path in [*drift, *stale]:
                shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
                print(f"stale: {shown}", file=sys.stderr)
            print(
                "Supported models pages are stale. Run make docs-model-pages.",
                file=sys.stderr,
            )
            return 1
        return 0
    for path in stale:
        path.unlink()
    for path in drift:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected[path], encoding="utf-8")
    print(f"Wrote {len(drift)} and removed {len(stale)} Supported models pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
