import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import router_calibration_support


class DeployConfigTest(unittest.TestCase):
    def test_deploy_config_uses_put_replace_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            yaml_path = Path(tempdir) / "router.yaml"
            dsl_path = Path(tempdir) / "router.dsl"
            yaml_path.write_text("version: v0.3\n", encoding="utf-8")
            dsl_path.write_text('ROUTE fallback { MODEL "qwen" }\n', encoding="utf-8")

            with mock.patch.object(
                router_calibration_support,
                "http_json",
                return_value=(200, {"status": "success"}),
            ) as http_json:
                result = router_calibration_support.deploy_config(
                    "http://router.example:8080",
                    yaml_path,
                    dsl_path,
                )

            self.assertEqual(result, {"status": "success"})
            http_json.assert_called_once_with(
                "PUT",
                "http://router.example:8080/config/router",
                {
                    "yaml": "version: v0.3\n",
                    "dsl": 'ROUTE fallback { MODEL "qwen" }\n',
                },
            )


if __name__ == "__main__":
    unittest.main()
