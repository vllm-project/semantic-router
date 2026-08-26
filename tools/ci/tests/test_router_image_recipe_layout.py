from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTER_DOCKERFILES = (
    REPO_ROOT / "src" / "vllm-sr" / "Dockerfile",
    REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.rocm",
    REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.cuda",
    REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc",
    REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc-rocm",
)
CLI_CONTAINER_START_ENVIRONMENT = (
    REPO_ROOT / "src" / "vllm-sr" / "cli" / "container_start_environment.py"
)
BUILT_IN_RECIPE_ROOT = "/app/recipes/built-in/latest/mom-v1"


def final_stage(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    starts = list(re.finditer(r"(?m)^FROM\s+", content))
    if not starts:
        raise AssertionError(f"{path} does not contain a Docker build stage")
    return content[starts[-1].start() :]


class RouterImageRecipeLayoutTests(unittest.TestCase):
    def test_published_router_images_share_one_runtime_asset_root(self) -> None:
        for dockerfile in ROUTER_DOCKERFILES:
            with self.subTest(dockerfile=dockerfile.relative_to(REPO_ROOT)):
                stage = final_stage(dockerfile)
                self.assertRegex(
                    stage,
                    r"(?m)^ENV\s+VLLM_SR_CONFIG_BASE_DIR=/app\s*$",
                )
                for asset in ("config.yaml", "metadata.yaml"):
                    expected = (
                        "COPY config/recipes/built-in/latest/mom-v1/"
                        f"{asset} {BUILT_IN_RECIPE_ROOT}/{asset}"
                    )
                    self.assertIn(expected, stage)

    def test_cli_uses_the_published_router_asset_root(self) -> None:
        content = CLI_CONTAINER_START_ENVIRONMENT.read_text(encoding="utf-8")
        self.assertIn('common_env["VLLM_SR_CONFIG_BASE_DIR"] = "/app"', content)


if __name__ == "__main__":
    unittest.main()
