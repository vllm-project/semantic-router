from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_integration_agent_task_bench as agent_task_integration
import yaml
from cli.parser import parse_user_config
from cli.sr_bench.transport import SESSION_ID_HEADER


class TestAgentTaskBenchmarkConfig(unittest.TestCase):
    def test_router_fixture_uses_canonical_backend_and_trace_policy(self) -> None:
        case = agent_task_integration.TestAgentTaskRouterIntegration(
            methodName="test_native_and_shared_smoke_tasks_use_real_router_sessions"
        )
        with tempfile.TemporaryDirectory() as directory:
            case.test_dir = directory
            case._write_session_config(mock_container="agent-task-upstream")
            path = Path(directory) / "config.yaml"
            parse_user_config(config_path=str(path), log_summary=False)
            config = yaml.safe_load(path.read_text())
        router = config["global"]["router"]
        self.assertEqual(router["headers"]["debug"], "any")
        self.assertEqual(
            router["learning"]["protection"]["identity"]["headers"]["session"],
            SESSION_ID_HEADER,
        )
        backend = config["providers"]["models"][0]["backend_refs"][0]
        self.assertEqual(backend["endpoint"], "http://agent-task-upstream:18080/v1")
        self.assertNotIn("base_url", backend)


if __name__ == "__main__":
    unittest.main()
