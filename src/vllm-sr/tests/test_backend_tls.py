"""A backend's tls block: Envoy's cluster trust and the files serve mounts."""

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli import container_cli, container_start  # noqa: E402
from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402

SYSTEM_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"

# The transport socket an https cluster had before backends took a tls block.
SYSTEM_TRUST_SOCKET = """    transport_socket:
      name: envoy.transport_sockets.tls
      typed_config:
        "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext
        sni: llm.internal.example
        auto_sni_san_validation: true
        common_tls_context:
          tls_params:
            tls_minimum_protocol_version: TLSv1_2
            tls_maximum_protocol_version: TLSv1_3
          validation_context:
            trusted_ca:
              filename: /etc/ssl/certs/ca-certificates.crt
"""


def _document(defaults="", backends=""):
    return f"""version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: private-llm
{defaults}  models:
    - name: private-llm
      provider_model_id: private-llm
      backend_refs:
{backends}recipes:
  - name: default
"""


HTTPS_BACKEND = """        - name: internal
          provider: openai-compatible
          endpoint: https://llm.internal.example/v1
"""


def _render(tmp_path, monkeypatch, document):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(document)
    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", "localhost")
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", "localhost")
    generate_envoy_config_from_user_config(
        parse_user_config(str(config_path), log_summary=False),
        str(output_path),
        log_summary=False,
    )
    return output_path.read_text()


def _tls_context(rendered, cluster="model_private_2dllm_cluster"):
    for item in yaml.safe_load(rendered)["static_resources"]["clusters"]:
        if item["name"] == cluster:
            return item["transport_socket"]["typed_config"]
    raise AssertionError(f"cluster {cluster!r} not found")


def test_https_cluster_without_tls_renders_the_system_bundle_byte_for_byte(
    tmp_path, monkeypatch
):
    rendered = _render(tmp_path, monkeypatch, _document(backends=HTTPS_BACKEND))

    assert SYSTEM_TRUST_SOCKET in rendered


def test_backend_tls_sets_the_clusters_trust_server_name_and_client_certificate(
    tmp_path, monkeypatch
):
    rendered = _render(
        tmp_path,
        monkeypatch,
        _document(
            backends="""        - name: internal
          provider: openai-compatible
          endpoint: https://llm-lb.internal.example/v1
          tls:
            ca_file: /etc/vllm-sr/certs/example-corp-ca.pem
            server_name: llm.internal.example
            cert_file: /etc/vllm-sr/certs/router.crt
            key_file: /etc/vllm-sr/certs/router.key
"""
        ),
    )

    tls_context = _tls_context(rendered)
    assert tls_context["sni"] == "llm.internal.example"
    assert tls_context["auto_sni_san_validation"] is True
    common = tls_context["common_tls_context"]
    assert common["validation_context"]["trusted_ca"]["filename"] == (
        "/etc/vllm-sr/certs/example-corp-ca.pem"
    )
    assert common["tls_certificates"] == [
        {
            "certificate_chain": {"filename": "/etc/vllm-sr/certs/router.crt"},
            "private_key": {"filename": "/etc/vllm-sr/certs/router.key"},
        }
    ]


def test_default_ca_reaches_https_backends_that_name_none(tmp_path, monkeypatch):
    rendered = _render(
        tmp_path,
        monkeypatch,
        """version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: private-llm
    tls:
      ca_file: /etc/vllm-sr/certs/example-corp-ca.pem
  models:
    - name: private-llm
      provider_model_id: private-llm
      backend_refs:
        - provider: openai-compatible
          endpoint: https://llm.internal.example/v1
    - name: partner-llm
      provider_model_id: partner-llm
      backend_refs:
        - provider: openai-compatible
          endpoint: https://partner.example.com/v1
          tls:
            ca_file: /etc/vllm-sr/certs/partner-ca.pem
    - name: local-llm
      provider_model_id: local-llm
      backend_refs:
        - provider: vllm
          endpoint: vllm.internal.example:8000
recipes:
  - name: default
""",
    )

    private = _tls_context(rendered)
    assert private["sni"] == "llm.internal.example"
    assert private["common_tls_context"]["validation_context"]["trusted_ca"][
        "filename"
    ] == ("/etc/vllm-sr/certs/example-corp-ca.pem")
    assert "tls_certificates" not in private["common_tls_context"]
    partner = _tls_context(rendered, "model_partner_2dllm_cluster")
    assert partner["common_tls_context"]["validation_context"]["trusted_ca"][
        "filename"
    ] == ("/etc/vllm-sr/certs/partner-ca.pem")
    clusters = yaml.safe_load(rendered)["static_resources"]["clusters"]
    local = next(c for c in clusters if c["name"] == "model_local_2dllm_cluster")
    assert "transport_socket" not in local
    assert SYSTEM_BUNDLE not in rendered


@pytest.mark.parametrize(
    ("defaults", "backends", "expected"),
    (
        (
            "",
            """        - provider: vllm
          endpoint: vllm.internal.example:8000
          tls: {ca_file: /etc/vllm-sr/certs/example-corp-ca.pem}
""",
            "tls applies to an https:// endpoint",
        ),
        (
            "",
            HTTPS_BACKEND + "          tls: {ca_file: certs/example-corp-ca.pem}\n",
            "tls.ca_file must be an absolute path",
        ),
        (
            "",
            HTTPS_BACKEND
            + "          tls: {cert_file: /etc/vllm-sr/certs/router.crt}\n",
            "tls.cert_file and tls.key_file go together",
        ),
        (
            "",
            HTTPS_BACKEND
            + '          tls: {server_name: "llm.internal.example:443"}\n',
            "tls.server_name must be a host name",
        ),
        (
            "",
            HTTPS_BACKEND + "          tls: {insecure_skip_verify: true}\n",
            "Envoy and the Router always verify a chat backend",
        ),
        (
            "    tls: {ca_file: example-corp-ca.pem}\n",
            HTTPS_BACKEND,
            "tls.ca_file must be an absolute path",
        ),
        (
            "",
            HTTPS_BACKEND
            + "          tls: {ca_file: /etc/vllm-sr/certs/a.pem}\n"
            + HTTPS_BACKEND.replace("internal", "replica", 1)
            + "          tls: {ca_file: /etc/vllm-sr/certs/b.pem}\n",
            "TLS files differ",
        ),
    ),
)
def test_backend_tls_is_validated_like_the_routers(
    tmp_path, monkeypatch, defaults, backends, expected
):
    with pytest.raises((ConfigParseError, ValueError), match=re.escape(expected)):
        _render(tmp_path, monkeypatch, _document(defaults, backends))


def _start(tmp_path, monkeypatch, gateway, document):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(document)
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_: {"router": "img", "envoy": "img", "dashboard": "img"},
    )
    monkeypatch.setattr(
        container_start, "_render_split_envoy_config", lambda *a, **k: None
    )
    commands = []

    def fake_run(cmd, capture_output, text, check, env=None):
        commands.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "127.0.0.1", "port": 8899}],
        network_name="vllm-sr-network",
        minimal=True,
        gateway=gateway,
    )
    assert rc == 0
    return {cmd[cmd.index("--name") + 1]: cmd for cmd in commands if "--name" in cmd}


def _mounts(command):
    return [command[i + 1] for i, arg in enumerate(command) if arg == "-v"]


@pytest.mark.parametrize("gateway", ("extproc", "standalone"))
def test_serve_mounts_the_backend_tls_files_read_only(tmp_path, monkeypatch, gateway):
    certs = tmp_path / "certs"
    certs.mkdir()
    for name in ("example-corp-ca.pem", "partner-ca.pem", "router.crt", "router.key"):
        (certs / name).write_text("pem")
    document = _document(
        defaults=f"    tls:\n      ca_file: {certs}/example-corp-ca.pem\n",
        backends=HTTPS_BACKEND
        + f"""          tls:
            ca_file: {certs}/partner-ca.pem
            cert_file: {certs}/router.crt
            key_file: {certs}/router.key
""",
    )

    containers = _start(tmp_path, monkeypatch, gateway, document)

    expected = [
        f"{certs}/{name}:{certs}/{name}:ro,z"
        for name in (
            "example-corp-ca.pem",
            "partner-ca.pem",
            "router.crt",
            "router.key",
        )
    ]
    readers = ["vllm-sr-router-container"]
    if gateway == "extproc":
        readers.append("vllm-sr-envoy-container")
    for name in readers:
        mounts = _mounts(containers[name])
        assert all(mount in mounts for mount in expected), (name, mounts)


def test_serve_refuses_a_backend_tls_file_that_is_not_there(tmp_path, monkeypatch):
    document = _document(
        backends=HTTPS_BACKEND
        + "          tls: {ca_file: /nonexistent/example-corp-ca.pem}\n",
    )
    with pytest.raises(ValueError, match=r"backend_refs\[0\]\.tls\.ca_file"):
        _start(tmp_path, monkeypatch, "extproc", document)
