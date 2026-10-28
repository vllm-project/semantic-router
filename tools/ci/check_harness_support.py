#!/usr/bin/env python3
"""Verify the harness support table against captures, replays, and protection."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import yaml

MANIFEST_PATH = Path("e2e/testcases/testdata/agent_clients/harness-support.yaml")
CAPTURE_DIR = Path("e2e/testcases/testdata/agent_clients")
GUIDE_PATH = Path("website/docs/installation/protocol-compatibility.md")
REPLAY_PATH = Path("e2e/testcases/protocol_codec_agent_client_replay.go")
PROTECTION_CONFIG_PATH = Path("src/semantic-router/pkg/config/learning_config.go")

BEGIN_MARKER = "<!-- BEGIN HARNESS SUPPORT TABLE -->"
END_MARKER = "<!-- END HARNESS SUPPORT TABLE -->"

LEVEL_ORDER = ("R0", "R1", "R2", "R3")
LEVEL_HEADER = ("Level", "Held unit", "Identity required", "Configuration")
CLIENT_HEADER = (
    "Client",
    "Version",
    "Status",
    "Paths",
    "Identity headers",
    "Level",
    "Limits",
)
LEVEL_FIELDS = ("held", "identity", "configuration")
LEVEL_ROLES = {
    "R0": frozenset(),
    "R1": frozenset({"session", "conversation"}),
    "R2": frozenset({"session"}),
    "R3": frozenset({"session", "agent"}),
}
LOOP_REQUESTS = ("turn-0", "turn-1")
REQUESTS = frozenset({*LOOP_REQUESTS, "subagent"})
ROLES = frozenset({"session", "conversation", "agent"})
STATUSES = frozenset({"verified", "broken"})
VERSION = re.compile(r"\d+(?:\.\d+)+$")
PLACEHOLDER = re.compile(r"placeholder-[a-z0-9-]+$")
HEADER_NAME = re.compile(r"[A-Za-z0-9-]+$")
TABLE_ROW = re.compile(r"^\|\s*(?P<first>[^|]+?)\s*\|(?P<rest>.*)\|$")
SEPARATOR_CELL = re.compile(r":?-{3,}:?")
IDENTITY_STRUCT = re.compile(
    r"type RouterLearningIdentityHeadersConfig struct \{([^}]*)\}",
    re.DOTALL,
)

# The replay cases and the backend format each one drives. A renamed case or
# backend fails together with the manifest that cites it.
REPLAY_CASES = {
    "protocol-codec-chat-backend-agent-client-replay": "openai.chat.v1",
    "protocol-codec-responses-backend-agent-client-replay": "openai.responses.v1",
    "protocol-codec-anthropic-backend-agent-client-replay": "anthropic.messages.v1",
}
REPLAY_MARKERS = (
    "func (capture agentClientCapture) identityHeaders",
    "agentClientRequestHeaders",
)


def normalize(value: str) -> str:
    """Drop markdown code ticks and collapse whitespace for cell comparison."""
    return re.sub(r"\s+", " ", value.replace("`", "")).strip()


def format_paths(paths: list[str]) -> str:
    return ", ".join(paths)


def format_identity(headers: list[dict[str, object]]) -> str:
    if not headers:
        return "None"
    parts = []
    for header in headers:
        requests = header.get("on")
        if not isinstance(requests, list):
            requests = []
        request_list = ", ".join(str(item) for item in requests)
        parts.append(f"{header.get('name')} ({request_list})")
    return "; ".join(parts)


def validate(repo_root: Path) -> list[str]:
    """Return disagreements between the manifest, captures, replay, and guide."""
    manifest_path = repo_root / MANIFEST_PATH
    if not manifest_path.is_file():
        return [f"harness support manifest is missing: {MANIFEST_PATH}"]
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        return [f"harness support manifest is invalid YAML: {error}"]

    errors = validate_manifest(repo_root, manifest)
    errors.extend(
        validate_guide(repo_root, manifest if isinstance(manifest, dict) else {})
    )
    return errors


def validate_manifest(repo_root: Path, manifest: object) -> list[str]:
    if not isinstance(manifest, dict):
        return ["harness support manifest must be a mapping"]
    errors = validate_levels(manifest.get("levels"))
    clients = manifest.get("clients")
    if not isinstance(clients, list) or not clients:
        return errors + ["harness support manifest must list clients"]

    capture_dir = repo_root / CAPTURE_DIR
    replay_text = read_text(repo_root / REPLAY_PATH)
    if replay_text is None:
        errors.append(f"replay file is missing: {REPLAY_PATH}")
        replay_text = ""
    else:
        for marker in REPLAY_MARKERS:
            if marker not in replay_text:
                errors.append(f"replay is missing identity marker: {marker}")
        for case, backend in REPLAY_CASES.items():
            if f'Register("{case}"' not in replay_text:
                errors.append(f"replay does not register {case}")
            if f'"{backend}"' not in replay_text:
                errors.append(f"replay does not mention backend {backend}")

    agent_ready, agent_error = protection_agent_ready(repo_root)
    if agent_error:
        errors.append(agent_error)

    referenced: list[str] = []
    seen_clients: set[str] = set()
    for index, client in enumerate(clients):
        client_errors, names = validate_client(capture_dir, client, index, agent_ready)
        errors.extend(client_errors)
        if isinstance(client, dict) and isinstance(client.get("client"), str):
            if client["client"] in seen_clients:
                errors.append(f"duplicate client: {client['client']}")
            seen_clients.add(client["client"])
        referenced.extend(names)

    if len(referenced) != len(set(referenced)):
        errors.append("a capture is listed for more than one version")
    on_disk = (
        sorted(path.name for path in capture_dir.glob("*.json"))
        if capture_dir.is_dir()
        else []
    )
    if sorted(referenced) != on_disk:
        errors.append(
            "capture files and manifest evidence disagree: "
            f"manifest={sorted(referenced)} disk={on_disk}"
        )
    return errors


def validate_levels(levels: object) -> list[str]:
    if not isinstance(levels, dict):
        return ["harness support levels must be a mapping"]
    errors: list[str] = []
    if set(levels) != set(LEVEL_ORDER):
        errors.append("levels must define R0, R1, R2, and R3")
    for level in LEVEL_ORDER:
        entry = levels.get(level)
        if not isinstance(entry, dict):
            errors.append(f"{level} must be a mapping")
            continue
        for field in LEVEL_FIELDS:
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{level}.{field} must be text")
    return errors


def validate_client(
    capture_dir: Path,
    client: object,
    index: int,
    agent_ready: bool,
) -> tuple[list[str], list[str]]:
    prefix = f"clients[{index}]"
    if not isinstance(client, dict):
        return [f"{prefix} must be a mapping"], []
    client_id = client.get("client")
    display_name = client.get("display_name")
    versions = client.get("versions")
    errors: list[str] = []
    if not isinstance(client_id, str) or not client_id:
        errors.append(f"{prefix}.client must be text")
        client_id = prefix
    if not isinstance(display_name, str) or not display_name.strip():
        errors.append(f"{client_id}: display_name must be text")
    if not isinstance(versions, list) or not versions:
        return errors + [f"{client_id}: versions must be a non-empty list"], []

    names: list[str] = []
    statuses: list[str] = []
    for version_index, version in enumerate(versions):
        version_errors, version_names, status = validate_version(
            capture_dir, client_id, version, version_index, agent_ready
        )
        errors.extend(version_errors)
        names.extend(version_names)
        if status:
            statuses.append(status)
    if "broken" in statuses and "verified" not in statuses:
        errors.append(
            f"{client_id}: a broken version must stay beside a verified version"
        )
    return errors, names


def validate_version(
    capture_dir: Path,
    client_id: str,
    version: object,
    index: int,
    agent_ready: bool,
) -> tuple[list[str], list[str], str]:
    prefix = f"{client_id} versions[{index}]"
    if not isinstance(version, dict):
        return [f"{prefix} must be a mapping"], [], ""
    version_id = version.get("version")
    if not isinstance(version_id, str) or VERSION.fullmatch(version_id) is None:
        label = prefix
        errors = [f"{prefix}: version must be an exact dotted number, not a range"]
    else:
        label = f"{client_id} {version_id}"
        errors = []

    status = version.get("status")
    if status not in STATUSES:
        errors.append(f"{label}: status must be verified or broken")
        status_text = ""
    else:
        status_text = str(status)
    errors.extend(validate_status_dates(label, version, status_text))

    level = version.get("level")
    if level not in LEVEL_ROLES:
        errors.append(f"{label}: level must be one of {', '.join(LEVEL_ORDER)}")
        level = ""
    if level == "R3" and not agent_ready:
        errors.append(
            f"{label}: R3 requires identity.headers.agent in protection config"
        )

    paths = version.get("paths")
    if (
        not isinstance(paths, list)
        or not paths
        or not all(isinstance(path, str) and path for path in paths)
    ):
        errors.append(f"{label}: paths must list ingress paths")
        paths = []
    limits = version.get("limits")
    if not isinstance(limits, str) or not limits.strip() or "|" in limits:
        errors.append(f"{label}: limits must be text without a table separator")

    identity = version.get("identity")
    identity_errors, identity_headers = validate_identity(label, identity, str(level))
    errors.extend(identity_errors)

    evidence = version.get("evidence")
    evidence_errors, names = validate_evidence(
        capture_dir,
        label,
        client_id,
        version_id if isinstance(version_id, str) else "",
        paths,
        identity_headers,
        evidence,
    )
    errors.extend(evidence_errors)
    return errors, names, status_text


def validate_status_dates(
    label: str, version: dict[str, object], status: str
) -> list[str]:
    errors: list[str] = []
    last_verified = version.get("last_verified")
    known_broken = version.get("known_broken_since")
    if status == "verified":
        if date_error := require_date(last_verified, f"{label}: last_verified"):
            errors.append(date_error)
        if known_broken is not None:
            errors.append(
                f"{label}: a verified version does not set known_broken_since"
            )
    elif status == "broken":
        if date_error := require_date(known_broken, f"{label}: known_broken_since"):
            errors.append(date_error)
        if last_verified is not None:
            errors.append(
                f"{label}: a broken version records known_broken_since, not last_verified"
            )
    return errors


def require_date(value: object, label: str) -> str | None:
    if not isinstance(value, str):
        return f"{label} must be a YYYY-MM-DD string"
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return f"{label} must be a YYYY-MM-DD string"
    return None


def validate_identity(
    label: str,
    identity: object,
    level: str,
) -> tuple[list[str], list[dict[str, object]]]:
    if not isinstance(identity, list):
        return [f"{label}: identity must be a list"], []
    errors: list[str] = []
    headers: list[dict[str, object]] = []
    roles: list[str] = []
    names: set[str] = set()
    for index, header in enumerate(identity):
        header_errors, parsed = validate_identity_header(
            f"{label} identity[{index}]", header
        )
        errors.extend(header_errors)
        if parsed is None:
            continue
        if parsed["name"] in names:
            errors.append(f"{label}: duplicate identity header {parsed['name']}")
        names.add(str(parsed["name"]))
        roles.append(str(parsed["role"]))
        headers.append(parsed)
    if level == "R0" and headers:
        errors.append(f"{label}: R0 records no identity header")
    if level in LEVEL_ROLES and level != "R0":
        missing = LEVEL_ROLES[level] - set(roles)
        if missing:
            errors.append(
                f"{label}: {level} is missing identity roles: {', '.join(sorted(missing))}"
            )
        errors.extend(validate_level_requests(label, level, headers))
    return errors, headers


def validate_identity_header(
    label: str,
    header: object,
) -> tuple[list[str], dict[str, object] | None]:
    if not isinstance(header, dict):
        return [f"{label} must be a mapping"], None
    name = header.get("name")
    role = header.get("role")
    placeholder = header.get("placeholder")
    requests = header.get("on")
    errors: list[str] = []
    if not isinstance(name, str) or HEADER_NAME.fullmatch(name) is None:
        errors.append(f"{label}: name must be a header token")
    if role not in ROLES:
        errors.append(f"{label}: role must be session, conversation, or agent")
    if not isinstance(placeholder, str) or PLACEHOLDER.fullmatch(placeholder) is None:
        errors.append(f"{label}: placeholder must be a placeholder-* token")
    if (
        not isinstance(requests, list)
        or not requests
        or not all(isinstance(item, str) and item in REQUESTS for item in requests)
        or len(requests) != len(set(requests))
    ):
        errors.append(
            f"{label}: on must list turn-0, turn-1, or subagent without duplicates"
        )
        requests = []
    if errors:
        return errors, None
    return [], {"name": name, "role": role, "placeholder": placeholder, "on": requests}


def validate_level_requests(
    label: str, level: str, headers: list[dict[str, object]]
) -> list[str]:
    by_role = {str(header["role"]): header for header in headers}
    errors: list[str] = []
    if level in {"R1", "R2", "R3"} and "session" in by_role:
        session_on = set(by_role["session"]["on"])
        if not set(LOOP_REQUESTS) <= session_on:
            errors.append(f"{label}: session header must be sent on turn-0 and turn-1")
    if level == "R1" and "conversation" in by_role and "session" in by_role:
        if list(by_role["conversation"]["on"]) != list(by_role["session"]["on"]):
            errors.append(
                f"{label}: session and conversation headers must be sent on the same requests"
            )
    if level == "R3" and "agent" in by_role and "session" in by_role:
        agent_on = set(by_role["agent"]["on"])
        session_on = set(by_role["session"]["on"])
        if "subagent" not in agent_on:
            errors.append(f"{label}: agent header must be sent on the subagent request")
        if not agent_on <= session_on:
            errors.append(
                f"{label}: agent header requests must also carry the session header"
            )
    return errors


def validate_evidence(
    capture_dir: Path,
    label: str,
    client_id: str,
    version_id: str,
    paths: list[str],
    identity: list[dict[str, object]],
    evidence: object,
) -> tuple[list[str], list[str]]:
    if not isinstance(evidence, dict):
        return [f"{label}: evidence must be a mapping"], []
    captures = evidence.get("captures")
    cases = evidence.get("e2e")
    errors: list[str] = []
    if (
        not isinstance(captures, list)
        or not captures
        or not all(
            isinstance(name, str) and name.endswith(".json") for name in captures
        )
    ):
        return [f"{label}: evidence.captures must list capture json files"], []
    if (
        not isinstance(cases, list)
        or not cases
        or not all(isinstance(name, str) for name in cases)
    ):
        errors.append(f"{label}: evidence.e2e must list replay cases")
        cases = []
    unknown_cases = [name for name in cases if name not in REPLAY_CASES]
    if unknown_cases:
        errors.append(f"{label}: unknown replay cases: {', '.join(unknown_cases)}")

    loaded: list[dict[str, object]] = []
    for name in captures:
        path = capture_dir / name
        if not path.is_file():
            errors.append(f"{label}: capture is missing: {name}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append(f"{label}: {name} is invalid JSON: {error}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{label}: {name} must be a JSON object")
            continue
        errors.extend(
            validate_capture(label, name, client_id, version_id, identity, payload)
        )
        loaded.append(payload)

    capture_paths = {payload.get("path") for payload in loaded}
    if loaded and capture_paths != set(paths):
        errors.append(f"{label}: ingress paths and capture paths disagree")
    declared = set()
    for payload in loaded:
        backends = payload.get("backends")
        if isinstance(backends, dict):
            declared.update(str(backend) for backend in backends)
    covered = {REPLAY_CASES[name] for name in cases if name in REPLAY_CASES}
    if loaded and declared != covered:
        errors.append(
            f"{label}: replay cases and capture backends disagree: "
            f"cases={sorted(covered)} captures={sorted(declared)}"
        )
    if "live_lane" in evidence and evidence.get("live_lane") is not None:
        live_lane = evidence.get("live_lane")
        if not isinstance(live_lane, str) or not live_lane.strip():
            errors.append(f"{label}: live_lane must be text when set")
    return errors, list(captures)


def validate_capture(
    label: str,
    name: str,
    client_id: str,
    version_id: str,
    identity: list[dict[str, object]],
    payload: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    if payload.get("client") != client_id or payload.get("version") != version_id:
        errors.append(f"{label}: {name} client or version does not match the manifest")
    recorded = payload.get("identity")
    headers = recorded.get("headers") if isinstance(recorded, dict) else None
    if not isinstance(headers, list):
        return errors + [f"{label}: {name} identity.headers must be a list"]
    if len(headers) != len(identity):
        errors.append(f"{label}: {name} identity headers do not match the manifest")
        return errors
    by_name = {}
    for header in headers:
        if not isinstance(header, dict) or not isinstance(header.get("name"), str):
            errors.append(f"{label}: {name} has an identity header without a name")
            continue
        by_name[header["name"]] = header
    needs_subagent = False
    for expected in identity:
        got = by_name.get(expected["name"])
        if not isinstance(got, dict):
            errors.append(
                f"{label}: {name} is missing identity header {expected['name']}"
            )
            continue
        if (
            got.get("placeholder") != expected["placeholder"]
            or got.get("on") != expected["on"]
        ):
            errors.append(
                f"{label}: {name} identity header {expected['name']} does not match the manifest"
            )
        if "subagent" in expected["on"]:
            needs_subagent = True
    if needs_subagent and not isinstance(payload.get("subagent"), dict):
        errors.append(
            f"{label}: {name} lists a subagent header without a subagent request"
        )
    return errors


def protection_agent_ready(repo_root: Path) -> tuple[bool, str | None]:
    text = read_text(repo_root / PROTECTION_CONFIG_PATH)
    if text is None:
        return False, f"protection config is missing: {PROTECTION_CONFIG_PATH}"
    match = IDENTITY_STRUCT.search(text)
    if match is None:
        return False, "RouterLearningIdentityHeadersConfig is missing"
    return 'yaml:"agent' in match.group(1), None


def validate_guide(repo_root: Path, manifest: dict[str, object]) -> list[str]:
    guide = read_text(repo_root / GUIDE_PATH)
    if guide is None:
        return [f"protocol guide is missing: {GUIDE_PATH}"]
    if guide.count(BEGIN_MARKER) != 1 or guide.count(END_MARKER) != 1:
        return [
            "protocol guide must contain one harness support begin marker and one end marker"
        ]
    section = guide.split(BEGIN_MARKER, 1)[1].split(END_MARKER, 1)[0]
    level_rows, client_rows, errors = parse_tables(section)
    if not isinstance(manifest.get("levels"), dict) or not isinstance(
        manifest.get("clients"), list
    ):
        return errors
    errors.extend(compare_levels(manifest["levels"], level_rows))
    errors.extend(compare_clients(manifest["clients"], client_rows))
    return errors


def parse_tables(section: str) -> tuple[list[list[str]], list[list[str]], list[str]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    errors: list[str] = []
    for line in section.splitlines():
        match = TABLE_ROW.match(line)
        if match is None:
            if current:
                tables.append(current)
                current = None
            continue
        cells = [
            match.group("first").strip(),
            *[cell.strip() for cell in match.group("rest").split("|")],
        ]
        if cells and all(SEPARATOR_CELL.fullmatch(cell) for cell in cells):
            continue
        if current is None:
            current = []
        current.append(cells)
    if current:
        tables.append(current)
    if len(tables) != 2:
        return (
            [],
            [],
            errors
            + ["harness support section must contain a level table and a client table"],
        )
    level_rows, level_errors = table_rows(tables[0], LEVEL_HEADER, "level")
    client_rows, client_errors = table_rows(tables[1], CLIENT_HEADER, "client")
    return level_rows, client_rows, errors + level_errors + client_errors


def table_rows(
    table: list[list[str]],
    header: tuple[str, ...],
    kind: str,
) -> tuple[list[list[str]], list[str]]:
    if not table:
        return [], [f"{kind} table is empty"]
    normalized = [[normalize(cell) for cell in row] for row in table]
    if tuple(normalized[0]) != header:
        return [], [f"{kind} table columns do not match the contract"]
    rows = normalized[1:]
    return rows, []


def compare_levels(levels: object, rows: list[list[str]]) -> list[str]:
    if not isinstance(levels, dict):
        return []
    expected = []
    for level in LEVEL_ORDER:
        entry = levels.get(level)
        if not isinstance(entry, dict):
            continue
        expected.append(
            [
                level,
                normalize(str(entry.get("held", ""))),
                normalize(str(entry.get("identity", ""))),
                normalize(str(entry.get("configuration", ""))),
            ]
        )
    if rows != expected:
        return ["level table does not match the manifest"]
    return []


def compare_clients(clients: object, rows: list[list[str]]) -> list[str]:
    if not isinstance(clients, list):
        return []
    expected: list[list[str]] = []
    for client in clients:
        if not isinstance(client, dict) or not isinstance(client.get("versions"), list):
            continue
        for version in client["versions"]:
            if not isinstance(version, dict):
                continue
            identity = version.get("identity")
            paths = version.get("paths")
            expected.append(
                [
                    normalize(str(client.get("display_name", ""))),
                    normalize(str(version.get("version", ""))),
                    normalize(str(version.get("status", ""))),
                    normalize(format_paths(paths if isinstance(paths, list) else [])),
                    normalize(
                        format_identity(identity if isinstance(identity, list) else [])
                    ),
                    normalize(str(version.get("level", ""))),
                    normalize(str(version.get("limits", ""))),
                ]
            )
    if rows != expected:
        return ["client table does not match the manifest"]
    return []


def read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    args = parser.parse_args()
    errors = validate(args.repo_root.resolve())
    if errors:
        for error in errors:
            print(f"harness support: {error}")
        return 1
    print("harness support: manifest, captures, replay, and guide agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
