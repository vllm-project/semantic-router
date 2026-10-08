#!/usr/bin/env python3
"""Render the chart in both gateway modes and assert what each one deploys."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

RELEASE = "gw"
ROUTER = f"{RELEASE}-semantic-router"
LISTENER_PORT = 8899
API_PORT = 8080
CLI_LISTENER = {"name": "http-8899", "address": "0.0.0.0", "port": 8899}


def render(chart: str, values: dict | None = None, *sets: str) -> list[dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
        yaml.safe_dump(values or {}, handle)
        handle.flush()
        command = ["helm", "template", RELEASE, chart, "-f", handle.name]
        for value in sets:
            command += ["--set", value]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RenderError(result.stderr)
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


class RenderError(Exception):
    pass


def render_fails(chart: str, values: dict | None, expected: str, *sets: str) -> None:
    try:
        render(chart, values, *sets)
    except RenderError as error:
        assert expected in str(error), f"expected {expected!r} in: {error}"
        return
    raise AssertionError(f"render should fail with {expected!r}")


def find(documents: list[dict], kind: str, name: str) -> dict | None:
    for document in documents:
        if document["kind"] == kind and document["metadata"]["name"] == name:
            return document
    return None


def router_container(documents: list[dict]) -> dict:
    deployment = find(documents, "Deployment", ROUTER)
    assert deployment, "Router Deployment missing"
    return deployment["spec"]["template"]["spec"]["containers"][0]


def router_volumes(documents: list[dict]) -> list[dict]:
    deployment = find(documents, "Deployment", ROUTER)
    return deployment["spec"]["template"]["spec"]["volumes"]


def service_ports(documents: list[dict], name: str = ROUTER) -> dict[str, tuple]:
    service = find(documents, "Service", name)
    assert service, f"Service {name} missing"
    return {p["name"]: (p["port"], p["targetPort"]) for p in service["spec"]["ports"]}


def probe_targets(container: dict) -> dict[str, dict]:
    targets = {}
    for probe in ("startupProbe", "livenessProbe", "readinessProbe"):
        handler = {
            k: v
            for k, v in container[probe].items()
            if k in ("httpGet", "grpc", "tcpSocket")
        }
        targets[probe] = handler
    return targets


def dashboard_env(documents: list[dict]) -> dict[str, str]:
    deployment = find(documents, "Deployment", f"{ROUTER}-dashboard")
    assert deployment, "Dashboard Deployment missing"
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    return {entry["name"]: entry.get("value") for entry in env}


def check_standalone_default(chart: str) -> None:
    documents = render(chart)
    container = router_container(documents)
    assert container["args"][:2] == [
        "-gateway=standalone",
        "-listener-address=0.0.0.0",
    ], container["args"]
    ports = {p["name"]: p["containerPort"] for p in container["ports"]}
    assert ports == {"http-8899": 8899, "metrics": 9190, "classify-api": 8080}, ports
    assert probe_targets(container) == {
        "startupProbe": {
            "httpGet": {"path": "/ready", "port": "http-8899", "scheme": "HTTP"}
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": "http-8899", "scheme": "HTTP"}
        },
        "readinessProbe": {
            "httpGet": {"path": "/ready", "port": "http-8899", "scheme": "HTTP"}
        },
    }, probe_targets(container)
    assert service_ports(documents) == {
        "http-8899": (8899, "http-8899"),
        "classify-api": (8080, 8080),
    }
    assert (
        find(documents, "Service", f"{ROUTER}-headless") is None
    ), "standalone renders no ext_proc headless Service"
    assert all(volume["name"] != "listener-tls" for volume in router_volumes(documents))


def check_extproc(chart: str) -> None:
    documents = render(chart, None, "gateway.mode=extproc")
    container = router_container(documents)
    assert container["args"] == ["-gateway=extproc", "--secure=false"], container[
        "args"
    ]
    ports = {p["name"]: p["containerPort"] for p in container["ports"]}
    assert ports == {"grpc": 50051, "metrics": 9190, "classify-api": 8080}, ports
    assert probe_targets(container) == {
        "startupProbe": {"grpc": {"port": 50051}},
        "livenessProbe": {"tcpSocket": {"port": 50051}},
        "readinessProbe": {"grpc": {"port": 50051}},
    }, probe_targets(container)
    assert service_ports(documents) == {
        "grpc": (50051, 50051),
        "classify-api": (8080, 8080),
    }
    assert service_ports(documents, f"{ROUTER}-headless") == {"grpc": (50051, 50051)}


def check_listener_ports_fail_closed(chart: str) -> None:
    def config_with(*listeners: dict) -> dict:
        return {"config": {"listeners": list(listeners)}}

    render_fails(
        chart,
        config_with({"name": "api", "port": 8080}),
        "listener api uses port 8080, which the Router API",
    )
    render_fails(
        chart, config_with({"name": "m", "port": 9190}), "which the Router metrics"
    )
    render_fails(
        chart,
        config_with({"name": "grpc-50051", "port": 50051}),
        "which the ext_proc port",
    )
    render_fails(
        chart,
        config_with({"name": "a", "port": 8899}, {"name": "b", "port": 8899}),
        "which listener a already takes",
    )
    render_fails(
        chart,
        {"configOverride": {"version": "v0.3", "listeners": []}},
        "the config has none",
    )
    render(
        chart,
        {"configOverride": {"version": "v0.3", "listeners": []}},
        "gateway.mode=extproc",
    )
    render_fails(chart, None, "standalone", "gateway.mode=native")
    render_fails(
        chart,
        {"args": ["--secure=false", "-gateway=extproc"]},
        "set gateway.mode instead",
    )


def check_listener_tls(chart: str) -> None:
    tls = {"cert_file": "certs/tls.crt", "key_file": "certs/tls.key"}
    https = {"name": "https-8443", "address": "0.0.0.0", "port": 8443, "tls": tls}
    render_fails(
        chart, {"config": {"listeners": [https]}}, "set gateway.tls.secretName"
    )

    documents = render(
        chart,
        {
            "config": {"listeners": [https]},
            "gateway": {"tls": {"secretName": "router-tls"}},
        },
    )
    container = router_container(documents)
    mounts = [
        mount for mount in container["volumeMounts"] if mount["name"] == "listener-tls"
    ]
    assert mounts == [
        {"name": "listener-tls", "mountPath": "/app/config/certs", "readOnly": True}
    ], mounts
    volumes = [
        volume
        for volume in router_volumes(documents)
        if volume["name"] == "listener-tls"
    ]
    assert volumes == [
        {"name": "listener-tls", "secret": {"secretName": "router-tls"}}
    ], volumes
    assert probe_targets(container)["readinessProbe"] == {
        "httpGet": {"path": "/ready", "port": "https-8443", "scheme": "HTTPS"}
    }
    assert service_ports(documents)["https-8443"] == (8443, "https-8443")

    mixed = {
        "config": {"listeners": [https, CLI_LISTENER]},
        "gateway": {"tls": {"secretName": "router-tls"}},
    }
    container = router_container(render(chart, mixed))
    assert probe_targets(container)["livenessProbe"]["httpGet"]["port"] == "http-8899"


def check_cli_kubernetes_values(chart: str) -> None:
    # The keys the CLI's kubernetes target writes for --gateway and --platform.
    for platform, repository, resource in (
        ("amd", "ghcr.io/vllm-project/semantic-router/vllm-sr-rocm", "amd.com/gpu"),
        (
            "nvidia",
            "ghcr.io/vllm-project/semantic-router/vllm-sr-cuda",
            "nvidia.com/gpu",
        ),
    ):
        for mode in ("standalone", "extproc"):
            values = {
                "gateway": {"mode": mode},
                "image": {"repository": repository},
                "resources": {"limits": {resource: 1}},
                "configOverride": {"version": "v0.3", "listeners": [CLI_LISTENER]},
            }
            container = router_container(render(chart, values))
            assert container["image"].startswith(repository + ":"), (
                platform,
                container["image"],
            )
            assert container["args"][0] == f"-gateway={mode}", (
                platform,
                container["args"],
            )
            limits = container["resources"]["limits"]
            assert limits[resource] == 1 and "memory" in limits and "cpu" in limits, (
                platform,
                limits,
            )


def check_dashboard_and_ingress(chart: str) -> None:
    enabled = {"dashboard": {"enabled": True}, "ingress": {"enabled": True}}
    documents = render(chart, enabled)
    env = dashboard_env(documents)
    assert env["VLLM_SR_GATEWAY"] == "standalone", env
    assert env["TARGET_ENVOY_URL"] == f"http://{ROUTER}:8899", env
    ingress = find(documents, "Ingress", ROUTER)
    assert (
        ingress["spec"]["rules"][0]["http"]["paths"][0]["backend"]["service"]["port"][
            "number"
        ]
        == LISTENER_PORT
    )

    documents = render(chart, enabled, "gateway.mode=extproc")
    env = dashboard_env(documents)
    assert env["VLLM_SR_GATEWAY"] == "extproc" and "TARGET_ENVOY_URL" not in env, env
    ingress = find(documents, "Ingress", ROUTER)
    assert (
        ingress["spec"]["rules"][0]["http"]["paths"][0]["backend"]["service"]["port"][
            "number"
        ]
        == API_PORT
    )


CHECKS = (
    check_standalone_default,
    check_extproc,
    check_listener_ports_fail_closed,
    check_listener_tls,
    check_cli_kubernetes_values,
    check_dashboard_and_ingress,
)


def main() -> None:
    chart = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(Path(__file__).parent / "semantic-router")
    )
    for check in CHECKS:
        check(chart)
        print(json.dumps({"check": check.__name__, "status": "ok"}))


if __name__ == "__main__":
    main()
