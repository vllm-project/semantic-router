"""The APIs the Router speaks, as its API registry (src/semantic-router/pkg/apiregistry) names them.

An API is named by the API, not its vendor; ``openai`` and ``anthropic`` are accepted as aliases of
``chat_completions`` and ``messages`` wherever an API is named (``providers.models[].api_format`` values and
``listeners[].apis`` keys) and read as those names. A schema ID names whose specification and which version.
The generated config schema lists the same names (tests check the two agree).
"""

from __future__ import annotations

API_NAMES = (
    "chat_completions",
    "responses",
    "messages",
    "images",
    "videos",
    "systemone",
)
ALIASES = {"openai": "chat_completions", "anthropic": "messages"}
# The APIs a listener without apis serves.
DEFAULT_LISTENER_APIS = (
    "chat_completions",
    "responses",
    "messages",
    "images",
    "videos",
)
# Each API's schema IDs; the first is an upstream's default, except systemone's.
SCHEMAS = {
    "chat_completions": ("openai/v1",),
    "responses": ("openai/v1",),
    "messages": ("anthropic/2023-06-01",),
    "images": ("openai/v1",),
    "videos": ("openai/v1",),
    "systemone": ("vllm-sr/5.0", "vllm-sr/4.0", "systemone/0.2"),
}
# The schemas a listener may serve a versioned API in; the first is the default.
LISTENER_SCHEMAS = {"systemone": ("vllm-sr/5.0", "vllm-sr/4.0")}
UPSTREAM_DEFAULTS = {"systemone": "systemone/0.2"}
# The settings an API takes on a listener besides schema and models.
SETTINGS = {"systemone": ("max_deadline_ms",)}
# Every request path each API answers (RE2, anchored), the Azure OpenAI forms
# among them, as the Router gates them; the generated Envoy listener routes by
# them too.
PATTERNS = {
    "chat_completions": r"^(/v1/chat/completions|/openai/v1/chat/completions|/openai/deployments/.+/chat/completions)$",
    "responses": r"^(/v1/responses(/[^/]+(/input_items)?)?|/openai/responses|/openai/v1/responses)$",
    "messages": r"^/v1/messages(/count_tokens)?$",
    "images": r"^/v1/images/generations$",
    "videos": r"^/v1/videos(/[^/]+(/content)?)?$",
    "systemone": r"^/v1/(systemone(/batch|/models)?|decisions)$",
}
# The model listing every listener serves.
MODELS_PATH = "/v1/models"


def served_path_regex(apis: tuple[str, ...]) -> str:
    """One RE2 expression a whole request path matches when one of the APIs, or the model listing, serves it."""
    alternatives = [PATTERNS[name].removeprefix("^").removesuffix("$") for name in apis]
    return "(" + "|".join([*alternatives, MODELS_PATH]) + ")"


def accepted_names() -> tuple[str, ...]:
    return (*API_NAMES, *ALIASES)


def canonical_api(name: str) -> str:
    """The API a name or alias names; ValueError for neither."""
    if name in API_NAMES:
        return name
    if name in ALIASES:
        return ALIASES[name]
    raise ValueError(
        f"api {name!r} is not supported; the apis are {', '.join(API_NAMES)} "
        f"(openai and anthropic are aliases of chat_completions and messages)"
    )


def upstream_schema(api: str, schema: str | None) -> str:
    """The schema an upstream speaks an API in: its own, else the API's upstream default; ValueError for one
    the API does not have."""
    name = canonical_api(api)
    if schema is None:
        return UPSTREAM_DEFAULTS.get(name, SCHEMAS[name][0])
    if schema not in SCHEMAS[name]:
        raise ValueError(
            f"schema {schema!r} is not a schema of api {name}; its schemas are {', '.join(SCHEMAS[name])}"
        )
    return schema
