package protocolcodec

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Snowflake Cortex AI conformance fixture, observed 2026-09-20 in the Japan
// (Tokio / AP Northeast 1) region against the Cortex REST API documented for this
// provider: POST {base_url}/chat/completions with `Authorization: Bearer
// <token>` on the account-scoped base URL.
//
// A request under a trial account returned the failure envelope in
// testdata/providers/snowflake-cortex-chat-error-out.json (HTTP 403, Snowflake
// code 003001). Its bytes are the observed response with the request ID and the
// account identifier redacted, and the size is pinned here so an edited fixture
// cannot pass as conformance.
//
// Accepted-wire conformance stays uncovered: trial accounts are not allowed to
// reach this endpoint, so the accepted envelope of both declared protocols has
// no live fixture yet. That is why the provider records `fixture_verified`, not
// `live_verified`.
const (
	snowflakeFailureFixture = "snowflake-cortex-chat-error-out.json"
	snowflakeFailureBytes   = 172
)

func snowflakeFixture(t *testing.T, name string, wantBytes int) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "providers", name))
	if err != nil {
		t.Fatalf("no provider conformance fixture %s: %v", name, err)
	}
	if len(body) != wantBytes {
		t.Fatalf("%s = %d bytes, want the observed %d", name, len(body), wantBytes)
	}
	return body
}

func snowflakePolicy(vendor llmprotocol.ResponseVendor) llmprotocol.Policy {
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = vendor
	return policy
}

func TestSnowflakeCortexFailureEnvelopeSurvivesTheSnowflakeVendor(t *testing.T) {
	body := snowflakeFixture(t, snowflakeFailureFixture, snowflakeFailureBytes)

	if _, _, err := (OpenAIChatCodec{}).DecodeTransportError(body, snowflakePolicy("")); err == nil {
		t.Fatal("strict canonical decode accepted a flat Snowflake failure envelope")
	}

	transportError, _, err := (OpenAIChatCodec{}).DecodeTransportError(
		body, snowflakePolicy(llmprotocol.ResponseVendorSnowflake),
	)
	if err != nil {
		t.Fatalf("DecodeTransportError() error = %v", err)
	}
	if transportError.Error == nil {
		t.Fatal("no transport error decoded")
	}
	if transportError.Error.Code != "003001" {
		t.Fatalf("transport error code = %q, want 003001", transportError.Error.Code)
	}
	if transportError.Error.Category != llmprotocol.ErrorPermission {
		t.Fatalf("transport error category = %q, want a permission failure", transportError.Error.Category)
	}
	if strings.TrimSpace(transportError.Error.Message) == "" {
		t.Fatal("transport error message is empty; the provider message was lost")
	}
}
