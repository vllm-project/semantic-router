"""Recipe receipts bind source and runtime without retaining deployment secrets."""

import json
from unittest.mock import Mock, patch

import pytest
from cli.sr_bench import provenance, snapshots
from cli.sr_bench.contracts import plan
from cli.sr_bench.report import make_report
from cli.sr_bench.store import Store

HASH = "a" * 64
SOURCE_HASH = "b" * 64
TARGET = {
    "id": "balance",
    "kind": "mom",
    "model": "balance",
    "base_url": "http://inference.test/v1",
    "config_hash": HASH,
    "preview_url": "http://router.test/api/v1/routing/preview",
    "capture_recipe": True,
}
HASHES = {
    "source_config_hash": SOURCE_HASH,
    "generated_runtime_hash": HASH,
    "active_runtime_hash": HASH,
    "activation_status": "active",
}


def response(document, *, etag=None, status=200):
    result = Mock(status_code=status, headers={"ETag": etag})
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.iter_content.return_value = [json.dumps(document).encode()]
    return result


def sequence(document, *, before=None, after=None, etag=f'"{SOURCE_HASH}"'):
    return [
        response(HASHES if before is None else before),
        response(document, etag=etag),
        response(HASHES if after is None else after),
    ]


def test_observed_snapshot_binds_distinct_source_and_runtime_hashes(monkeypatch):
    document = {
        "recipes": [{"name": "balance", "routing": {"api_key": "private"}}],
        "providers": {"models": [{"api_key": "backend-secret"}]},
        "global": {"router": {"learning": {"enabled": True}}},
        "evaluation": {"source": "https://source.test/file?token=private"},
        "routing": {"max_tokens": 4096, "endpoint": "http://private.test"},
    }
    monkeypatch.setenv("TEST_RECIPE_KEY", "private-token")
    target = {**TARGET, "preview_api_key_env": "TEST_RECIPE_KEY"}
    with patch.object(snapshots.requests, "get", side_effect=sequence(document)) as get:
        receipt = snapshots.capture_recipes({"targets": [target]})["balance"]
    assert [call.args[0] for call in get.call_args_list] == [
        "http://router.test/api/v1/config/hash",
        "http://router.test/api/v1/config",
        "http://router.test/api/v1/config/hash",
    ]
    assert all(not call.kwargs["allow_redirects"] for call in get.call_args_list)
    assert all(
        call.kwargs["headers"]["Authorization"] == "Bearer private-token"
        for call in get.call_args_list
    )
    assert receipt["config_hash"] == HASH
    assert receipt["source_config_hash"] == SOURCE_HASH != HASH
    assert receipt["generated_runtime_hash"] == receipt["active_runtime_hash"] == HASH
    assert receipt["source"] == "router_config_api_bracketed_hashes"
    assert receipt["learning"] == {"enabled": True}
    assert receipt["routing"]["max_tokens"] == 4096
    assert receipt["redactions"]
    assert "providers" not in receipt
    assert "private" not in json.dumps(receipt)


@pytest.mark.parametrize(
    "field",
    [
        "auth_token",
        "authToken",
        "APIKey",
        "api-key",
        "refresh_token",
        "client_secret",
        "password",
        "private_key",
        "headers",
        "customHeaders",
        "request_headers",
        "metadata",
        "address",
        "ip_address",
        "hostnames",
        "bind_address",
        "source_path",
        "modelPath",
        "socket",
        "cache_dir",
        "private_url",
        "callback_uri",
        "connection_string",
        "tls_certificate",
    ],
)
def test_secret_and_deployment_fields_are_redacted_wholesale(field):
    value = (
        {"X-Custom": "opaque-secret"} if "header" in field.lower() else "opaque-secret"
    )
    with patch.object(
        snapshots.requests, "get", side_effect=sequence({"recipes": [{field: value}]})
    ):
        receipt = snapshots.capture_recipes({"targets": [TARGET]})["balance"]
    assert receipt["recipes"][0][field] == "<redacted>"
    assert "opaque-secret" not in json.dumps(receipt)
    assert "X-Custom" not in json.dumps(receipt)


@pytest.mark.parametrize(
    "value",
    [
        "https://private.example/config",
        "file:///etc/key",
        "s3://bucket/object",
        "/home/operator/data",
        "../private/config",
        "~/credentials",
        r"C:\private\key",
        r"\\private\share",
        "10.20.30.40",
        "fd00::1",
        "use node.internal",
        "Bearer secret-value",
        "Basic encoded-secret",
    ],
)
def test_private_values_under_unrecognized_keys_are_redacted(value):
    with patch.object(
        snapshots.requests,
        "get",
        side_effect=sequence({"recipes": [{"extension": value}]}),
    ):
        receipt = snapshots.capture_recipes({"targets": [TARGET]})["balance"]
    assert receipt["recipes"][0]["extension"] == "<redacted>"


def test_model_identity_routing_numbers_and_safe_environment_references_survive():
    recipe = {
        "model": "qwen/qwen3.8-27b",
        "max_tokens": 4096,
        "temperature": 1,
        "api_key_env": "MODEL_API_KEY",
        "auth_token_env": "actual-private-value",
        "algorithm": {"type": "static", "weights": [0.3, 0.7]},
    }
    with patch.object(
        snapshots.requests, "get", side_effect=sequence({"recipes": [recipe]})
    ):
        receipt = snapshots.capture_recipes({"targets": [TARGET]})["balance"]
    assert receipt["recipes"][0] == {**recipe, "auth_token_env": "<redacted>"}


@pytest.mark.parametrize("etag", [None, '"other"', f'"{HASH}"', f'W/"{SOURCE_HASH}"'])
def test_config_etag_must_match_observed_source_hash(etag):
    with (
        patch.object(
            snapshots.requests, "get", side_effect=sequence({"recipes": []}, etag=etag)
        ) as get,
        pytest.raises(ValueError, match="source configuration acknowledgement"),
    ):
        snapshots.capture_recipes({"targets": [TARGET]})
    assert get.call_count == 2


@pytest.mark.parametrize(
    "hashes",
    [
        {**HASHES, "generated_runtime_hash": "c" * 64},
        {**HASHES, "active_runtime_hash": "c" * 64},
        {**HASHES, "activation_status": "pending"},
        {**HASHES, "activation_status": "failed"},
        {**HASHES, "source_config_hash": "bad"},
        {**HASHES, "source_config_hash": None},
        {},
        [],
    ],
)
@pytest.mark.parametrize("observation", ["before", "after"])
def test_inactive_or_unacknowledged_runtime_is_rejected(hashes, observation):
    with (
        patch.object(
            snapshots.requests,
            "get",
            side_effect=sequence({"recipes": []}, **{observation: hashes}),
        ),
        pytest.raises(ValueError, match="Recipe capture"),
    ):
        snapshots.capture_recipes({"targets": [TARGET]})


def test_source_change_during_capture_is_rejected_even_when_runtime_is_unchanged():
    with (
        patch.object(
            snapshots.requests,
            "get",
            side_effect=sequence(
                {"recipes": []}, after={**HASHES, "source_config_hash": "c" * 64}
            ),
        ),
        pytest.raises(ValueError, match="changed during capture"),
    ):
        snapshots.capture_recipes({"targets": [TARGET]})


@pytest.mark.parametrize("redirect_at", [0, 1, 2])
def test_redirects_are_rejected_at_each_observation(redirect_at):
    replies = sequence({"recipes": []})
    replies[redirect_at] = response({}, status=302)
    with (
        patch.object(snapshots.requests, "get", side_effect=replies) as get,
        pytest.raises(ValueError, match="HTTP 302"),
    ):
        snapshots.capture_recipes({"targets": [TARGET]})
    assert get.call_count == redirect_at + 1


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@router.test/api/v1/routing/preview",
        "http://router.test/arbitrary",
        "http://router.test/api/v1/routing/preview?redirect=other",
        "http://router.test/api/v1/routing/preview#fragment",
        "http://router.test:invalid/api/v1/routing/preview",
        "http://router.test\n/api/v1/routing/preview",
        "http://router.test%2fother/api/v1/routing/preview",
        "http://router.test\\other/api/v1/routing/preview",
        "file:///api/v1/routing/preview",
        None,
    ],
)
def test_capture_cannot_redirect_credentials_to_an_arbitrary_url(url):
    with patch.object(snapshots.requests, "get") as get, pytest.raises(ValueError):
        snapshots.capture_recipes({"targets": [{**TARGET, "preview_url": url}]})
    get.assert_not_called()


def test_byte_and_shared_absolute_time_bounds(monkeypatch):
    monkeypatch.setattr(snapshots, "MAX_CONFIG_BYTES", 4)
    with (
        patch.object(snapshots.requests, "get", side_effect=sequence({"recipes": []})),
        pytest.raises(ValueError, match="size limit"),
    ):
        snapshots.capture_recipes({"targets": [TARGET]})
    monkeypatch.setattr(snapshots, "MAX_CONFIG_BYTES", 4096)
    with (
        patch.object(
            snapshots.requests, "get", side_effect=sequence({"recipes": []})
        ) as get,
        patch.object(snapshots.time, "monotonic", side_effect=[0, 0, 1, 1, 2, 16]),
        pytest.raises(ValueError, match="deadline"),
    ):
        snapshots.capture_recipes({"targets": [TARGET]})
    assert get.call_count == 2


def test_transport_errors_do_not_expose_the_request_url():
    with (
        patch.object(
            snapshots.requests,
            "get",
            side_effect=snapshots.requests.ConnectionError(
                "private address and credentials"
            ),
        ),
        pytest.raises(ValueError) as error,
    ):
        snapshots.capture_recipes({"targets": [TARGET]})
    assert str(error.value) == "Recipe capture transport failed"


def test_opt_in_is_explicit_and_single_models_have_no_recipe():
    with patch.object(snapshots.requests, "get") as get:
        assert (
            snapshots.capture_recipes(
                {"targets": [{**TARGET, "capture_recipe": False}]}
            )
            == {}
        )
    get.assert_not_called()
    with pytest.raises(ValueError, match="MoM"):
        snapshots.capture_recipes({"targets": [{**TARGET, "kind": "single"}]})
    with pytest.raises(ValueError, match="boolean"):
        snapshots.capture_recipes({"targets": [{**TARGET, "capture_recipe": 1}]})


def test_recipe_receipt_survives_store_reopen_and_report(tmp_path):
    manifest = plan(
        {
            "version": "sr-bench-1.0",
            "name": "receipt-test",
            "mode": "preview",
            "cost_policy": "capability_only",
            "targets": [TARGET],
            "cases": [
                {
                    "id": "q1",
                    "benchmark": "mmlu-pro",
                    "messages": [{"role": "user", "content": "Return A"}],
                    "answer": "A",
                }
            ],
        }
    )
    with (
        patch.object(
            snapshots.requests,
            "get",
            side_effect=sequence(
                {"recipes": [{"name": "balance", "auth_token": "private"}]}
            ),
        ),
        patch.object(provenance.importlib.metadata, "distributions", return_value=[]),
    ):
        runner = provenance.capture_runner(manifest)
    store = Store(tmp_path)
    run, created = store.create(manifest, provenance=runner)
    assert created
    store.db.close()
    report = make_report(Store(tmp_path), run["id"])
    assert (
        report["provenance"]["runner"]["recipe_snapshots"] == runner["recipe_snapshots"]
    )
    assert "private" not in json.dumps(report["provenance"]["runner"])
    assert (
        report["provenance"]["runner"]["recipe_snapshots"]["balance"][
            "active_runtime_hash"
        ]
        == HASH
    )
