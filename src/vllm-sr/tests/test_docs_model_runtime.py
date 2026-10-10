"""Every snippet of the model runtime user docs is checked.

Configuration fragments are validated by the CLI's parser and validator, the
runtime's models file by the runtime's own loader, request bodies against the
runtime's OpenAPI contract, `vllm-sr` command lines by the CLI's own option
parser, and the migration example by `vllm-sr config migrate`. The engine-mode
CLI integration test reads the Quickstart's requests from the page and sends
them to a running runtime.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import jsonschema
import pytest
import yaml
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_migration import migrate_config_data  # noqa: E402
from cli.config_migration_notes import MigrationNotes  # noqa: E402
from cli.main import main  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402
from cli.validator import validate_user_config  # noqa: E402

DOCS = REPO_ROOT / "website" / "docs"
PAGES = [
    *sorted((DOCS / "model-runtime").rglob("*.md")),
    DOCS / "tutorials/global/model-runtime.md",
    DOCS / "tutorials/signal/learned/decision.md",
    DOCS / "tutorials/algorithm/selection/decision.md",
]
RUNTIME = REPO_ROOT / "src" / "model-runtime" / "vllm_srun"
FENCE = re.compile(r"^```(\w*)([^\n]*)\n(.*?)^```", re.M | re.S)
TITLE = re.compile(r'title="([^"]+)"')
# Runtime surfaces only; router requests (/v1/chat/completions) are covered by E2E.
CURL_BODY = re.compile(
    r"curl[^\n]*?(/v1/(?:decisions|systemone|classify|embeddings|rerank|bundle))"
    r"(?:(?!\ncurl).)*?-d '(\{.*?\})'",
    re.S,
)
REQUEST_SCHEMAS = {
    "/v1/decisions": "DecisionRequest",
    "/v1/classify": "ClassifyRequest",
    "/v1/embeddings": "EmbeddingsRequest",
    "/v1/rerank": "RerankRequest",
    "/v1/bundle": "BundleRequest",
}
# Fragments that are not router configuration documents.
NON_ROUTER_TITLES = {"legacy.yaml", "legacy.migrated.yaml", "models.yaml"}


@dataclass(frozen=True)
class Block:
    page: Path
    language: str
    title: str
    text: str
    # "```yaml alternative" marks a fragment that replaces the page's earlier
    # fragments instead of adding to them.
    alternative: bool = False

    @property
    def where(self) -> str:
        return f"{self.page.relative_to(REPO_ROOT)}:{self.title or self.language}"


def _blocks(language: str) -> list[Block]:
    blocks = []
    for page in PAGES:
        for match in FENCE.finditer(page.read_text(encoding="utf-8")):
            if match.group(1) != language:
                continue
            meta = match.group(2)
            title = TITLE.search(meta)
            blocks.append(
                Block(
                    page,
                    language,
                    title.group(1) if title else "",
                    match.group(3),
                    alternative="alternative" in meta.split(),
                )
            )
    return blocks


def _block(page: str, title: str) -> Block:
    for block in _blocks("yaml") + _blocks("json") + _blocks("text"):
        if block.page == DOCS / page and block.title == title:
            return block
    raise AssertionError(f"{page} has no block titled {title!r}")


def _router_fragments() -> list[Block]:
    fragments = []
    for block in _blocks("yaml"):
        if block.title in NON_ROUTER_TITLES or "apiVersion" in block.text:
            continue
        assert isinstance(yaml.safe_load(block.text), dict), block.where
        fragments.append(block)
    return fragments


def _published_images() -> dict[str, tuple]:
    """The images CI builds and publishes: name -> (context, Dockerfile, platforms)."""
    ci = REPO_ROOT / "tools" / "ci"
    sys.path.insert(0, str(ci))
    try:
        spec = importlib.util.spec_from_file_location(
            "image_artifacts", ci / "image_artifacts.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ci))
    return module.DEFINITIONS


def test_kubernetes_manifests_parse_and_run_images_the_repository_builds():
    manifests = [block for block in _blocks("yaml") if "apiVersion" in block.text]
    published = _published_images()
    runtime_commands = []

    assert manifests
    for block in manifests:
        documents = list(yaml.safe_load_all(block.text))
        assert all(document["kind"] for document in documents), block.where
        for document in documents:
            pod = document.get("spec", {}).get("template", {}).get("spec", {})
            for container in pod.get("containers", []):
                registry, _, image = container["image"].rpartition("/")
                assert registry == "ghcr.io/vllm-project/semantic-router", block.where
                assert image.split(":")[0] in published, block.where
                _, dockerfile, _ = published[image.split(":")[0]]
                assert (REPO_ROOT / dockerfile).is_file(), block.where
                command = [*container.get("command", []), *container.get("args", [])]
                if command[:1] == ["vllm-srun"]:
                    runtime_commands.append((block.where, command[1:]))

    assert runtime_commands
    for where, arguments in runtime_commands:
        _assert_runtime_options(where, arguments)


def _merge(base: dict, fragment: dict) -> dict:
    for key, value in fragment.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _model_names(value) -> set[str]:
    names = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "modelRefs":
                names |= {ref["model"] for ref in child}
            names |= _model_names(child)
    elif isinstance(value, list):
        for child in value:
            names |= _model_names(child)
    return names


def _complete(fragment: dict) -> dict:
    """A minimal valid configuration around a fragment."""
    if "version" in fragment:
        return fragment
    models = sorted(_model_names(fragment) | {"answer-model"})
    base = {
        "version": "v0.3",
        "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
        "providers": {
            "defaults": {"model": "answer-model"},
            "models": [
                {
                    "name": name,
                    "backend_refs": [
                        {"name": "primary", "endpoint": "vllm:8000", "protocol": "http"}
                    ],
                }
                for name in models
            ],
        },
        "routing": {"modelCards": [{"name": name} for name in models]},
    }
    return _merge(base, fragment)


def _page_configs() -> list[tuple[str, dict]]:
    """Each fragment merged with the fragments before it on its page.

    A page builds a configuration step by step ("describe a deployment", then
    "bind it"), so a fragment is checked the way a reader applies it.
    """
    configs, page, merged = [], None, {}
    for index, block in enumerate(_router_fragments()):
        if block.page != page:
            page, merged = block.page, {}
        fragment = yaml.safe_load(block.text)
        if "version" in fragment or block.alternative:
            merged = {}
        merged = _merge(merged, fragment)
        configs.append((f"{block.where}#{index}", copy.deepcopy(merged)))
    return configs


@pytest.mark.parametrize(
    "where, fragment", _page_configs(), ids=[where for where, _ in _page_configs()]
)
def test_router_config_fragments_validate(where, fragment, tmp_path):
    config = _complete(fragment)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    errors = validate_user_config(parse_user_config(str(path)))

    assert [str(error) for error in errors] == [], where


def _runtime_config_module():
    spec = importlib.util.spec_from_file_location(
        "vllm_srun_config", RUNTIME / "config.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_models_file_example_loads_with_the_runtime_loader(tmp_path):
    block = _block("model-runtime/reference.md", "models.yaml")
    path = tmp_path / "models.yaml"
    path.write_text(block.text, encoding="utf-8")

    models = _runtime_config_module().load_models_file(path)

    assert [model.name for model in models] == ["vela-domain", "decision-kai"]


OPENAPI_URI = "urn:vllm-srun:openapi"


def _openapi_validator(schema_name: str) -> jsonschema.Draft4Validator:
    openapi = yaml.safe_load((RUNTIME / "api" / "openapi.yaml").read_text())
    resource = Resource.from_contents(openapi, default_specification=DRAFT4)
    return jsonschema.Draft4Validator(
        {"$ref": f"{OPENAPI_URI}#/components/schemas/{schema_name}"},
        registry=Registry().with_resource(OPENAPI_URI, resource),
    )


def _validate_request(path: str, body: dict, where: str) -> None:
    schema_path = "/v1/decisions" if path == "/v1/systemone" else path
    validator = _openapi_validator(REQUEST_SCHEMAS[schema_path])
    problems = [error.message for error in validator.iter_errors(body)]
    assert problems == [], f"{where} {path}: {problems}"


def _curl_requests() -> list[tuple[str, str, dict]]:
    requests = []
    for block in _blocks("bash"):
        for match in CURL_BODY.finditer(block.text):
            requests.append((block.where, match.group(1), json.loads(match.group(2))))
    return requests


def test_curl_request_bodies_follow_the_runtime_contract():
    requests = _curl_requests()

    assert len(requests) >= 8
    for where, path, body in requests:
        _validate_request(path, body, where)


def test_json_request_examples_follow_the_runtime_contract():
    examples = [block for block in _blocks("json") if block.title.startswith("POST ")]

    assert {block.title for block in examples} == {
        f"POST {path}" for path in REQUEST_SCHEMAS
    }
    for block in examples:
        _validate_request(
            block.title.removeprefix("POST "), json.loads(block.text), block.where
        )


def test_reference_decision_responses_have_the_documented_shape():
    responses = [
        block
        for block in _blocks("json")
        if block.title == "Response" and "reference.md" in str(block.where)
    ]
    validator = _openapi_validator("DecisionResponse")

    assert len(responses) >= 2
    for block in responses:
        # The examples omit usage; everything they show must conform.
        problems = [
            error.message
            for error in validator.iter_errors(json.loads(block.text))
            if error.validator != "required"
        ]
        assert problems == [], block.where


def test_quickstart_response_example_has_the_documented_shape():
    response = json.loads(_block("model-runtime/quickstart.md", "Response").text)
    validator = _openapi_validator("DecisionResponse")
    # The illustrative response omits usage; everything it shows must conform.
    problems = [
        error.message
        for error in validator.iter_errors(response)
        if error.validator != "required"
    ]
    assert problems == []
    assert set(response["answers"]) == {"kind", "reasoning"}


def _commands(program: str) -> list[tuple[str, list[str]]]:
    commands = []
    for block in _blocks("bash"):
        logical = block.text.replace("\\\n", " ")
        for raw in logical.splitlines():
            line = raw.strip()
            if line.startswith(program + " "):
                commands.append((block.where, shlex.split(line)[1:]))
    return commands


def _parse_cli(arguments: list[str]) -> None:
    """Parse a command line the way `vllm-sr` would, without running it."""
    command, rest = main, list(arguments)
    context = click.Context(main, info_name="vllm-sr")
    while isinstance(command, click.Group):
        assert rest, f"vllm-sr {' '.join(arguments)} names no command"
        name = rest.pop(0)
        sub = command.get_command(context, name)
        if sub is None:
            raise click.UsageError(f"no command {name!r}")
        command, context = sub, click.Context(sub, info_name=name, parent=context)
    command.make_context(context.info_name, rest, parent=context.parent)


def test_vllm_sr_commands_parse():
    commands = _commands("vllm-sr")

    assert len(commands) >= 10
    for where, arguments in commands:
        try:
            _parse_cli(arguments)
        except click.ClickException as error:
            raise AssertionError(
                f"{where}: vllm-sr {' '.join(arguments)}: {error}"
            ) from error


def test_translations_in_sync_show_the_english_snippets():
    """A translated page that claims to be current shows the snippets the tests check."""
    translated = (
        DOCS.parent / "i18n" / "zh-Hans" / "docusaurus-plugin-content-docs" / "current"
    )
    checked = 0
    for page in PAGES:
        translation = translated / page.relative_to(DOCS)
        if not translation.is_file():
            continue
        text = translation.read_text(encoding="utf-8")
        if re.search(r"^\s*outdated:\s*true\s*$", text, re.M):
            continue
        assert FENCE.findall(text) == FENCE.findall(
            page.read_text(encoding="utf-8")
        ), translation.relative_to(REPO_ROOT)
        checked += 1
    assert checked


def test_documented_image_builds_use_dockerfiles_that_exist():
    builds = [
        (block.where, dockerfile)
        for block in _blocks("bash")
        for dockerfile in re.findall(
            r"docker (?:buildx )?build\b[^\n]*?-f (\S+)",
            block.text.replace("\\\n", " "),
        )
    ]

    assert builds
    for where, dockerfile in builds:
        assert (REPO_ROOT / dockerfile).is_file(), f"{where}: {dockerfile}"


def test_documented_environment_variables_are_read_by_the_runtime_or_router():
    documented = {
        name
        for page in PAGES
        for name in re.findall(r"\bVLLM_SRUN_[A-Z_]+\b", page.read_text())
    }
    router = REPO_ROOT / "src" / "semantic-router" / "pkg"
    sources = "\n".join(
        path.read_text()
        for path in (
            *RUNTIME.rglob("*.py"),
            *(router / "modelservice").glob("*.go"),
        )
    )

    assert documented
    assert sorted(name for name in documented if f'"{name}"' not in sources) == []


def _assert_runtime_options(where: str, arguments: list[str]) -> None:
    source = (RUNTIME / "cli.py").read_text()
    assert f'"{arguments[0]}"' in source, f"{where}: no subcommand {arguments[0]}"
    for argument in arguments[1:]:
        if argument.startswith("--"):
            assert f'"{argument}"' in source, f"{where}: no option {argument}"


def test_vllm_srun_commands_use_real_options():
    commands = _commands("vllm-srun")

    assert commands
    for where, arguments in commands:
        _assert_runtime_options(where, arguments)


def test_migration_example_is_what_the_tool_writes():
    legacy = yaml.safe_load(_block("model-runtime/migrate.md", "legacy.yaml").text)
    documented = yaml.safe_load(
        _block("model-runtime/migrate.md", "legacy.migrated.yaml").text
    )
    notes = MigrationNotes()

    migrated = migrate_config_data(legacy, notes)

    assert migrated == documented
    assert migrate_config_data(migrated) == migrated
    page = (DOCS / "model-runtime/migrate.md").read_text()
    for note in notes:
        if note.path.endswith("vela-domain") or note.path.endswith(
            "vela-domain.precision"
        ):
            assert note.message in page, note.message
