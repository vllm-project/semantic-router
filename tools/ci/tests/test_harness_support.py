import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "ci" / "check_harness_support.py"
SPEC = importlib.util.spec_from_file_location("check_harness_support", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
harness_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness_check)


def capture_headers(identity: object) -> list[dict[str, object]]:
    if not isinstance(identity, list):
        return []
    return [
        {
            "name": header.get("name"),
            "placeholder": header.get("placeholder"),
            "on": header.get("on"),
        }
        for header in identity
        if isinstance(header, dict)
    ]


class HarnessSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def write_contract(self, **overrides: object) -> dict[str, object]:
        version: dict[str, object] = {
            "version": "1.2.3",
            "status": "verified",
            "last_verified": "2026-09-26",
            "paths": ["/v1/messages"],
            "identity": [],
            "level": "R0",
            "limits": "The capture has no identity header.",
            "evidence": {
                "captures": ["example-1.2.3.json"],
                "e2e": ["protocol-codec-chat-backend-agent-client-replay"],
            },
        }
        version.update(overrides)
        manifest: dict[str, object] = {
            "levels": {
                level: {
                    "held": held,
                    "identity": identity,
                    "configuration": configuration,
                }
                for level, held, identity, configuration in (
                    (
                        "R0",
                        "Each request is routed independently",
                        "None",
                        "Protection off, or the required header is absent",
                    ),
                    (
                        "R1",
                        "One conversation",
                        "Session and conversation headers",
                        "scope: conversation",
                    ),
                    (
                        "R2",
                        "One run, subagents included",
                        "Session header",
                        "scope: session",
                    ),
                    (
                        "R3",
                        "Each agent of a run separately",
                        "Session and agent headers",
                        "scope: session with identity.headers.agent",
                    ),
                )
            },
            "clients": [
                {
                    "client": "example",
                    "display_name": "Example",
                    "versions": [version],
                }
            ],
        }
        manifest_path = self.root / harness_check.MANIFEST_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        identity = version["identity"]
        capture: dict[str, object] = {
            "client": "example",
            "version": version["version"],
            "path": "/v1/messages",
            "identity": {"headers": capture_headers(identity)},
            "backends": {"openai.chat.v1": ["tools"]},
        }
        if isinstance(identity, list) and any(
            isinstance(header, dict) and "subagent" in header.get("on", [])
            for header in identity
        ):
            capture["subagent"] = {"model": "example"}
        capture_path = self.root / harness_check.CAPTURE_DIR / "example-1.2.3.json"
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_path.write_text(json.dumps(capture), encoding="utf-8")

        replay = self.root / harness_check.REPLAY_PATH
        replay.parent.mkdir(parents=True, exist_ok=True)
        replay.write_text(
            "\n".join(
                [
                    "func (capture agentClientCapture) identityHeaders",
                    "agentClientRequestHeaders",
                    *(
                        f'Register("{case}"\n"{backend}"'
                        for case, backend in harness_check.REPLAY_CASES.items()
                    ),
                ]
            ),
            encoding="utf-8",
        )
        config = self.root / harness_check.PROTECTION_CONFIG_PATH
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            "type RouterLearningIdentityHeadersConfig struct {\n"
            '\tSession *string `yaml:"session,omitempty"`\n'
            '\tConversation *string `yaml:"conversation,omitempty"`\n'
            "}\n",
            encoding="utf-8",
        )
        guide = self.root / harness_check.GUIDE_PATH
        guide.parent.mkdir(parents=True, exist_ok=True)
        guide.write_text(self.render_guide(manifest), encoding="utf-8")
        return manifest

    def render_guide(self, manifest: dict[str, object]) -> str:
        levels = manifest["levels"]
        assert isinstance(levels, dict)
        level_lines = [
            "| " + " | ".join(harness_check.LEVEL_HEADER) + " |",
            "| " + " | ".join("---" for _ in harness_check.LEVEL_HEADER) + " |",
        ]
        for level in harness_check.LEVEL_ORDER:
            entry = levels[level]
            assert isinstance(entry, dict)
            level_lines.append(
                "| "
                + " | ".join(
                    [
                        level,
                        str(entry["held"]),
                        str(entry["identity"]),
                        str(entry["configuration"]),
                    ]
                )
                + " |"
            )
        client_lines = [
            "| " + " | ".join(harness_check.CLIENT_HEADER) + " |",
            "| " + " | ".join("---" for _ in harness_check.CLIENT_HEADER) + " |",
        ]
        clients = manifest["clients"]
        assert isinstance(clients, list)
        for client in clients:
            assert isinstance(client, dict)
            versions = client["versions"]
            assert isinstance(versions, list)
            for version in versions:
                assert isinstance(version, dict)
                identity = version["identity"]
                paths = version["paths"]
                assert isinstance(identity, list) and isinstance(paths, list)
                client_lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(client["display_name"]),
                            str(version["version"]),
                            str(version["status"]),
                            harness_check.format_paths([str(path) for path in paths]),
                            harness_check.format_identity(identity),
                            str(version["level"]),
                            str(version["limits"]),
                        ]
                    )
                    + " |"
                )
        return "\n".join(
            [
                harness_check.BEGIN_MARKER,
                "",
                *level_lines,
                "",
                *client_lines,
                "",
                harness_check.END_MARKER,
                "",
            ]
        )

    def test_complete_contract_passes(self) -> None:
        self.write_contract()
        self.assertEqual(harness_check.validate(self.root), [])

    def test_table_and_missing_capture_fail(self) -> None:
        self.write_contract()
        guide = self.root / harness_check.GUIDE_PATH
        guide.write_text(
            guide.read_text(encoding="utf-8").replace("Example", "Other", 1),
            encoding="utf-8",
        )
        (self.root / harness_check.CAPTURE_DIR / "example-1.2.3.json").unlink()
        errors = harness_check.validate(self.root)
        self.assertTrue(any("client table" in error for error in errors), errors)
        self.assertTrue(any("capture is missing" in error for error in errors), errors)

    def test_range_placeholder_replay_and_request_coverage_fail(self) -> None:
        self.write_contract(
            version="1.x",
            level="R2",
            identity=[
                {
                    "name": "x-session-id",
                    "role": "session",
                    "placeholder": "placeholder-session",
                    "on": ["turn-0"],
                },
                {
                    "name": "x-conversation-id",
                    "role": "conversation",
                    "placeholder": "real-session",
                    "on": ["turn-0", "turn-1"],
                },
            ],
            evidence={
                "captures": ["example-1.2.3.json"],
                "e2e": ["missing-replay"],
            },
        )
        errors = harness_check.validate(self.root)
        self.assertTrue(any("exact dotted number" in error for error in errors), errors)
        self.assertTrue(any("placeholder-* token" in error for error in errors), errors)
        self.assertTrue(
            any("unknown replay cases" in error for error in errors), errors
        )
        self.assertTrue(any("turn-0 and turn-1" in error for error in errors), errors)

    def test_r3_and_broken_status_fail_without_evidence_fields(self) -> None:
        manifest = self.write_contract(
            level="R3",
            last_verified=None,
            identity=[
                {
                    "name": "x-session-id",
                    "role": "session",
                    "placeholder": "placeholder-session",
                    "on": ["turn-0", "turn-1", "subagent"],
                },
                {
                    "name": "x-agent-id",
                    "role": "agent",
                    "placeholder": "placeholder-agent",
                    "on": ["subagent"],
                },
            ],
        )
        errors = harness_check.validate(self.root)
        self.assertTrue(
            any("identity.headers.agent" in error for error in errors), errors
        )
        self.assertTrue(any("last_verified" in error for error in errors), errors)

        manifest["clients"][0]["versions"] = [
            {
                "version": "1.2.4",
                "status": "broken",
                "paths": ["/v1/messages"],
                "identity": [],
                "level": "R0",
                "limits": "Broken before a capture could be refreshed.",
                "evidence": {
                    "captures": ["example-1.2.3.json"],
                    "e2e": ["protocol-codec-chat-backend-agent-client-replay"],
                },
            }
        ]
        (self.root / harness_check.MANIFEST_PATH).write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        (self.root / harness_check.GUIDE_PATH).write_text(
            self.render_guide(manifest),
            encoding="utf-8",
        )
        errors = harness_check.validate(self.root)
        self.assertTrue(any("known_broken_since" in error for error in errors), errors)
        self.assertTrue(
            any("beside a verified version" in error for error in errors), errors
        )

    def test_repository_contract_passes(self) -> None:
        self.assertEqual(harness_check.validate(REPO_ROOT), [])
