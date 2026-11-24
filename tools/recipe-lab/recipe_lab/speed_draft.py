"""Create a dataset-free, fixed-binding derivative of the installed v0.5 Speed recipe."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(rows, name, kind):
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"{kind}: expected a list of mappings")
    matches = [row for row in rows if row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {kind} named {name!r}")
    return matches[0]


def validate_cards(providers_doc, strong, fast, tool_model):
    if not isinstance(providers_doc, dict) or providers_doc.get("version") != "v0.5":
        raise ValueError(
            "requires a v0.5 provider configuration; migrate old inputs explicitly"
        )
    if not strong or not fast or strong == fast or tool_model not in (strong, fast):
        raise ValueError(
            "supply two distinct model names; the tool model must be one of them"
        )
    provider = providers_doc.get("providers")
    if not isinstance(provider, dict):
        raise ValueError("missing providers mapping")
    models = provider.get("models", [])
    destinations = []
    tool_capable = set()
    for name in (strong, fast):
        card = _unique(models, name, "providers.models entry")
        caps = card.get("capabilities")
        if not isinstance(caps, list) or not all(isinstance(c, str) for c in caps):
            raise ValueError(
                f"{name}: declare capabilities on the actual provider model"
            )
        caps = {c.strip().lower() for c in caps}
        if caps.isdisjoint({"text", "chat"}):
            raise ValueError(f"{name}: text generation capability is required")
        if caps.intersection({"tools", "tool_use"}):
            tool_capable.add(name)
        elif name == tool_model:
            raise ValueError(
                f"{name}: the tools lane requires declared tools capability"
            )
        for field in ("context_window_size", "max_output_tokens"):
            value = card.get(field)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"{name}: declare a known positive {field} on providers.models"
                )
        if card["context_window_size"] <= 1:
            raise ValueError(
                f"{name}: context must have room for input and generated output"
            )
        backends = card.get("backend_refs")
        if (
            not isinstance(backends, list)
            or not backends
            or not all(isinstance(ref, dict) for ref in backends)
        ):
            raise ValueError(f"{name}: configure a real backend_refs entry")
        # Two aliases on the same shared API can be valid (different served
        # model IDs). Catch only an identical declared destination; physical
        # deployment identity must still be verified with real requests.
        refs = {
            json.dumps(
                {k: v for k, v in ref.items() if k not in {"name", "weight"}},
                sort_keys=True,
            )
            for ref in backends
        }
        destinations.append((card.get("provider_model_id") or name, refs))
    if destinations[0] == destinations[1]:
        raise ValueError("the two roles resolve to the same declared model and backend")
    return tool_capable


def _bindings(strong, fast, tool_model):
    return {
        k: [{"model": m}]
        for k, m in (("tools", tool_model), ("reasoning", strong), ("fast", fast))
    }


def specialize(
    config, public_model, strong, fast, tool_model, *, agent_streaming=False
):
    """Specialize just the newly bound Speed scope; preserve siblings and provider cards."""
    tool_capable = validate_cards(config, strong, fast, tool_model)
    result = copy.deepcopy(config)
    matches = [
        e
        for e in result.get("entrypoints", [])
        if public_model in e.get("model_names", [])
    ]
    if len(matches) != 1 or matches[0].get("recipe") != "speed":
        raise ValueError(
            f"{public_model!r} must resolve uniquely to the installed speed recipe"
        )
    speed_entries = [e for e in result["entrypoints"] if e.get("recipe") == "speed"]
    if len(speed_entries) != 1 or speed_entries[0].get("model_names") != [public_model]:
        raise ValueError(
            "refusing to specialize a Speed scope shared with another public name"
        )
    target = _unique(result.get("recipes", []), "speed", "recipe")
    required = target.get("candidate_requirements", {})
    if (
        required.get("context") != "known_limits"
        or required.get("capabilities") != "declared"
    ):
        raise ValueError(
            "Speed must retain scoped known_limits and declared capabilities"
        )
    bindings = _bindings(strong, fast, tool_model)
    # The native static selector prefers the first eligible model when there
    # are no domain scores for the candidates in this recipe's registry.
    if len(tool_capable) == 1 and any(
        score["model"] in (strong, fast)
        for domain in target.get("signals", {}).get("domains", [])
        for score in domain.get("model_scores", [])
    ):
        raise ValueError("scoped domain model_scores can override candidate order")
    # YAML anchors may share decision or plugin objects across lanes and recipes.
    # Each lane gets its own copy before applying model-specific defaults.
    decisions = [copy.deepcopy(lane) for lane in target.get("decisions", [])]
    result["recipes"] = [
        {**recipe, "decisions": decisions} if recipe is target else recipe
        for recipe in result["recipes"]
    ]
    if (
        len(decisions) != len(bindings)
        or {d["name"] for d in decisions} != bindings.keys()
    ):
        raise ValueError("unrecognized Speed lanes; inspect the installed template")
    for lane in decisions:
        name = lane["name"]
        if lane.get("modelRefs") != bindings[name]:
            raise ValueError(
                f"{name}: native composition did not bind the intended singleton"
            )
        algo = lane.get("algorithm", {})
        if algo.get("type") != "multi_factor" or algo.get("minimum_candidates") != 1:
            raise ValueError(
                f"{name}: unexpected upstream selector; refusing to replace it"
            )
        lane["algorithm"] = {"type": "static"}
        if agent_streaming:
            lane.setdefault("reliability", {})["total_timeout"] = "0s"
        refs = lane["modelRefs"]
        if refs[0]["model"] not in tool_capable:
            # The native gate sees the full protocol (including tool_choice
            # and historical calls/results), regardless of the selected lane.
            refs.append({"model": tool_model})
        # Preserve the template's missing-value-only output defaults when possible.
        # An explicit client budget is never replaced; native known_limits handles it.
        cards = [
            _unique(result["providers"]["models"], ref["model"], "model")
            for ref in refs
        ]
        for plugin in lane.get("plugins", []):
            if plugin.get("type") == "request_params":
                params = plugin["configuration"]
                if "default_max_tokens" in params:
                    params["default_max_tokens"] = min(
                        params["default_max_tokens"],
                        *(
                            min(c["max_output_tokens"], c["context_window_size"] - 1)
                            for c in cards
                        ),
                    )
    return result


def _tool(executable):
    resolved = shutil.which(str(executable))
    if not resolved:
        raise ValueError(f"required executable not found: {executable}")
    return Path(resolved).resolve()


def _run(argv):
    # Keep native diagnostics on stderr; do not write them into the shareable manifest.
    cmd = subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE, text=True)
    if cmd.returncode:
        raise ValueError(
            f"{Path(argv[0]).name} failed ({cmd.returncode}): {cmd.stdout.strip()}"
        )
    return cmd.stdout


def _dump(path, value):
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def generate(args):
    try:
        from .report_delivery import write_report
    except ImportError:
        # This file also runs standalone (the scenario pages call it by path
        # and the regression test loads it directly); resolve the report
        # helper through whichever layout is active.
        try:
            from report_delivery import write_report
        except ImportError:
            from recipe_lab.report_delivery import write_report

    cli, router = _tool(args.vllm_sr), _tool(args.router)
    agent_streaming = getattr(args, "agent_streaming", False)
    providers, out = args.providers.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError(f"refusing to replace existing output: {out}")
    provider_hash = _hash(providers)
    tool_model = args.tool_model or args.strong
    public = args.public_model
    validate_cards(
        yaml.safe_load(providers.read_bytes()), args.strong, args.fast, tool_model
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".speed-draft-") as tmp:
        stage = Path(tmp)
        # For freshly initialized configurations, depend on the installed package's
        # contents, not on the checkout templates used during development.
        meta = json.loads(
            _run(
                [
                    cli,
                    "recipe",
                    "builtin",
                    "export",
                    "mom-v1",
                    "--catalog-version",
                    args.catalog_version,
                    "--output-dir",
                    stage / "template",
                ]
            )
        )
        template_path = stage / "template/config.yaml"
        if yaml.safe_load(template_path.read_bytes()).get("version") != "v0.5":
            raise ValueError("installed mom-v1 catalog is not v0.5")
        package = stage / "package"
        package.mkdir(mode=0o700)
        primary_bindings = _bindings(args.strong, args.fast, tool_model)
        native_bindings = stage / "primary-bindings.yaml"
        _dump(native_bindings, primary_bindings)
        native = stage / "native.yaml"
        receipt = json.loads(
            _run(
                [
                    cli,
                    "recipe",
                    "builtin",
                    "init",
                    "speed",
                    "--bundle",
                    "mom-v1",
                    "--catalog-version",
                    args.catalog_version,
                    "--config",
                    providers,
                    "--bindings",
                    native_bindings,
                    "--model-name",
                    public,
                    "--output",
                    native,
                ]
            )
        )
        if (
            meta["catalog_version"] != receipt["catalog_version"]
            or receipt.get("valid") is not True
            or _hash(providers) != provider_hash
        ):
            raise ValueError(
                "template composition did not verify against the same inputs"
            )
        document = yaml.safe_load(native.read_bytes())
        result = specialize(
            document,
            public,
            args.strong,
            args.fast,
            tool_model,
            agent_streaming=agent_streaming,
        )
        speed = _unique(result["recipes"], "speed", "recipe")
        bindings = {d["name"]: d["modelRefs"] for d in speed["decisions"]}
        _dump(package / "bindings.yaml", bindings)
        config = package / "config.yaml"
        _dump(config, result)
        _run([cli, "config", "validate", "--offline", "--config", config])
        checked = json.loads(
            _run(
                [
                    router,
                    "-validate-config",
                    "-config",
                    config,
                    "-gateway",
                    "standalone",
                ]
            )
        )
        if checked.get("valid") is not True:
            raise ValueError("native Router did not report valid=true")
        manifest = {
            "schema": "recipe-lab-v05-speed-draft-v1",
            "config_version": "v0.5",
            "status": "statically_validated",
            "gateway_started": False,
            "business_quality_measured": False,
            "public_model": public,
            "bindings": bindings,
            "native_primary_bindings": primary_bindings,
            "template": {
                "bundle": "mom-v1",
                "recipe": "speed",
                "catalog_version": meta["catalog_version"],
                "bundle_sha256": meta["sha256"],
                "config_sha256": _hash(template_path),
            },
            "toolchain": {
                "cli_version": _run([cli, "--version"]).strip(),
                "cli_launcher_sha256": _hash(cli),
                "router_sha256": _hash(router),
                "generator_sha256": _hash(Path(__file__)),
            },
            "sha256": {
                "providers": provider_hash,
                "native_composition": _hash(native),
                "bindings.yaml": _hash(package / "bindings.yaml"),
                "config.yaml": _hash(config),
            },
            "changes": [
                "Each lane has a user-selected primary. Static selection takes the first "
                "eligible binding in the listed order without quality or price measurements.",
                "Omitted-output defaults are bounded by every listed candidate's declared "
                "limits. Explicit caller limits remain subject to native eligibility.",
            ],
            "router_warning_count": len(checked.get("warnings") or []),
        }
        if agent_streaming:
            manifest["changes"].append(
                "All three Speed decisions set reliability.total_timeout to 0s for "
                "long agent calls. The configured stream idle policy still applies."
            )
        if bindings != primary_bindings:
            manifest["changes"].append(
                "A lane whose primary lacks tools also lists the designated tool backend. "
                "Native eligibility checks tool definitions, choice and history, as well "
                "as other hard capabilities and context/output limits, before static selection. "
                "Decision priorities, user requests and tool choices are preserved."
            )
        (package / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_report(package / "report.md", [manifest])
        if out.exists():
            raise ValueError(f"output appeared during generation: {out}")
        package.rename(out)
    return manifest


def main(argv=None, *, prog=None):
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("--providers", type=Path, required=True)
    p.add_argument("--strong", required=True)
    p.add_argument("--fast", required=True)
    p.add_argument(
        "--tool-model", help="Tool-capable backend among strong/fast (default: strong)"
    )
    p.add_argument("--public-model", default="vllm-sr/recipe-lab")
    p.add_argument(
        "--agent-streaming",
        action="store_true",
        help="Allow long calls on Speed decisions while retaining the stream idle timeout",
    )
    p.add_argument("--catalog-version", default="latest")
    p.add_argument("--vllm-sr", default="vllm-sr")
    p.add_argument(
        "--router", required=True, help="Native Router binary used by the deployment"
    )
    p.add_argument(
        "--out", required=True, type=Path, help="New private draft directory"
    )
    args = p.parse_args(argv)
    try:
        result = generate(args)
    except (
        OSError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        p.exit(2, f"speed-draft: {exc}\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.out),
                "report": str((args.out / "report.md").resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
