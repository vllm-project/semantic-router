import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_relative_path(*parts: str) -> str:
    return str(Path(*parts))


AGENTGATEWAY_BLOG = REPO_ROOT / _repo_relative_path(
    "website", "blog", "2026-06-28-agentgateway-semantic-brain-homelab.md"
)
EXTERNAL_PROCESSOR_TYPE = (
    "type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor"
)
ENVOY_CONFIGS = (
    "bench/cpu-vs-gpu/envoy-bench.yaml",
    "deploy/local/envoy.yaml",
    "deploy/kserve/configmap-envoy-config.yaml",
    "deploy/kserve/configmap-envoy-config-simulator.yaml",
    "deploy/openshift/envoy-openshift.yaml",
    "deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
    "deploy/kubernetes/aibrix/aigw-resources/gwapi-resources.yaml",
    "deploy/kubernetes/dynamo/dynamo-resources/gwapi-resources.yaml",
    "deploy/kubernetes/istio/envoyfilter.yaml",
    "deploy/kubernetes/response-api/gwapi-resources.yaml",
    "deploy/kubernetes/routing-strategies/aigw-resources/gwapi-resources.yaml",
    "deploy/kubernetes/streaming/aigw-resources/gwapi-resources.yaml",
    _repo_relative_path("e2e", "config", "envoy-authorino-test.yaml"),
    _repo_relative_path(
        "e2e", "profiles", "anthropic-shim", "gateway-resources", "gwapi-resources.yaml"
    ),
    _repo_relative_path(
        "e2e", "profiles", "authz-rbac", "gateway-resources", "gwapi-resources.yaml"
    ),
    _repo_relative_path(
        "e2e",
        "profiles",
        "ml-model-selection",
        "gateway-resources",
        "gwapi-resources.yaml",
    ),
    _repo_relative_path(
        "e2e", "profiles", "multi-endpoint", "gateway-resources", "gwapi-resources.yaml"
    ),
)


def _walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _load_envoy_config(path: Path):
    manifests = [doc for doc in yaml.safe_load_all(path.read_text()) if doc is not None]
    configs = []
    for manifest in manifests:
        if manifest.get("kind") == "ConfigMap" and "envoy.yaml" in manifest.get(
            "data", {}
        ):
            embedded_config = re.sub(
                r"\{\{[^{}]+\}\}",
                "placeholder",
                manifest["data"]["envoy.yaml"],
            )
            configs.append(yaml.safe_load(embedded_config))
        else:
            configs.append(manifest)
    return configs


@pytest.mark.parametrize("relative_path", ENVOY_CONFIGS)
def test_checked_in_extproc_configs_explicitly_fail_closed(relative_path):
    config = _load_envoy_config(REPO_ROOT / relative_path)
    ext_proc_configs = [
        value
        for value in _walk(config)
        if isinstance(value, dict) and value.get("@type") == EXTERNAL_PROCESSOR_TYPE
    ]

    assert ext_proc_configs, relative_path
    for ext_proc_config in ext_proc_configs:
        assert "failure_mode_allow" in ext_proc_config, relative_path
        assert ext_proc_config["failure_mode_allow"] is False, relative_path


def test_agentgateway_blog_does_not_recommend_extproc_fail_open():
    blog = AGENTGATEWAY_BLOG.read_text()

    assert "failureMode: failClosed" in blog
    assert "failureMode: failOpen" not in blog
    assert "`failOpen`" not in blog
    assert "Fourteen selection algorithms" not in blog
    assert "ELO, RL-driven" not in blog
    assert "ten single-model selectors" in blog
    assert "type: router_dc" in blog
    assert "type: multi_factor" not in blog
