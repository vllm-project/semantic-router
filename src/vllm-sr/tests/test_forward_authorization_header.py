"""Config-contract tests for backend_refs[].forward_authorization_header.

Mirrors the Go loader (validateCanonicalContract): the flag is a per-backend
opt-in to forward the caller's inbound Authorization verbatim, and it is
incompatible with api_format: anthropic until the Anthropic-native path
implements forwarding.
"""

import pytest

from cli.models import BackendRef, Model


def test_backend_ref_forward_authorization_header_field():
    br = BackendRef(
        name="gateway",
        base_url="https://litellm.example.com/v1",
        forward_authorization_header=True,
    )
    assert br.forward_authorization_header is True


def test_backend_ref_forward_authorization_header_defaults_off_and_omitted():
    br = BackendRef(name="local", endpoint="127.0.0.1:8000")
    # Defaults off, and omitted from serialization (omitempty parity with Go).
    assert br.forward_authorization_header is None
    assert "forward_authorization_header" not in br.model_dump(exclude_none=True)


def test_model_rejects_forward_authorization_header_with_anthropic_format():
    with pytest.raises(ValueError, match="not supported with api_format: anthropic"):
        Model(
            name="claude-worker",
            api_format="anthropic",
            backend_refs=[
                # Only the second backend opts in; the flag is per-backend while
                # the format is per-model, so this must still be rejected.
                BackendRef(name="static", endpoint="127.0.0.1:8000"),
                BackendRef(
                    name="gateway",
                    base_url="https://litellm.example.com/v1",
                    forward_authorization_header=True,
                ),
            ],
        )


def test_model_allows_forward_authorization_header_for_non_anthropic_format():
    model = Model(
        name="gpt-worker",
        api_format="openai",
        backend_refs=[
            BackendRef(
                name="gateway",
                base_url="https://litellm.example.com/v1",
                forward_authorization_header=True,
            )
        ],
    )
    assert model.backend_refs[0].forward_authorization_header is True


def test_model_allows_anthropic_format_without_forwarding():
    model = Model(
        name="claude-worker",
        api_format="anthropic",
        backend_refs=[BackendRef(name="static", endpoint="127.0.0.1:8000")],
    )
    assert model.api_format == "anthropic"
