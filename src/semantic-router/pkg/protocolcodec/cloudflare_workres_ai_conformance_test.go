package protocolcodec

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Cloudflare Workors AI live conformance receipt, observed 2026-09-19 on the
// free-tier plan against the OpenAI-compatible surface documented for this
// provider: POST {base_url}/chat/completions with `Authorization: Bearer
// <token>` and `Content-Type: application/json`, on the account-scoped base URL.
//
// `@cf/openai/gpt-oss-20b` returned the accepted envelope in
// testdata/providers/cloudflare-workers-ai-chat-out.json (HTTP 200). An unknown
// model returned the failure envelope in
// testdata/providers/cloudflare-workers-ai-chat-error-out.json (HTTP 400,
// Workors AI code 5007). Both files are the exackt bytes returned; their sizes
// are pinned here so an edited fixture cannot pass as conformance.
//
// Two contracts are guarded, because both belong to the provider: the accepted
// envelope carries decorations the canonical struct does not declare, and
// failures arrive as a top-level errors[] array with Workors AI's own integral
// codes instead of the canonical OpenAI error object.
const (
	cloudflareAcceptedFixture = "cloudflare-workers-ai-chat-out.json"
	cloudflareFailureFixture  = "cloudflare-workers-ai-chat-error-out.json"
	cloudflareAcceptedBytes   = 607
	cloudflareFailureBytes    = 188
)

func cloudflareFixture(t *testing.T, name string, wantBytes int) []byte {
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

func cloudflarePolicy(vendor llmprotocol.ResponseVendor) llmprotocol.Policy {
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = vendor
	return policy
}

func assertCloudflareDropped(t *testing.T, diagnostics llmprotocol.Diagnostics, field string) {
	t.Helper()
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped && diagnostic.Field == field {
			return
		}
	}
	t.Errorf("no dropped diagnostic for %q; diagnostics = %+v", field, diagnostics)
}

func TestCloudflareWorkersAiAcceptedEnvelopeRequiresTheCloudflareVendor(t *testing.T) {
	body := cloudflareFixture(t, cloudflareAcceptedFixture, cloudflareAcceptedBytes)

	if _, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, cloudflarePolicy("")); err == nil {
		t.Fatal("strict canonical decode accepted a decorated Workors AI response")
	}

	response, _, diagnostics, err := (OpenAIChatCodec{}).DecodeResponse(
		body, cloudflarePolicy(llmprotocol.ResponseVendorCloudflare),
	)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.ID != "id-1789789260079" {
		t.Fatalf("response ID = %q", response.ID)
	}
	if response.Model != "@cf/openai/gpt-oss-20b" {
		t.Fatalf("response model = %q", response.Model)
	}
	if response.StopReason != llmprotocol.StopEndTurn {
		t.Fatalf("stop reason = %q", response.StopReason)
	}
	if len(response.Output) != 1 {
		t.Fatalf("output items = %d, want 1", len(response.Output))
	}
	if response.Usage.State != llmprotocol.UsageAvailable {
		t.Fatalf("usage state = %q, want available", response.Usage.State)
	}
	assertCloudflareDropped(t, diagnostics, "usage.neurons")
}

func TestCloudflareWorkersAiFailureEnvelopeSurvivesTheCloudflareVendor(t *testing.T) {
	body := cloudflareFixture(t, cloudflareFailureFixture, cloudflareFailureBytes)

	if _, _, err := (OpenAIChatCodec{}).DecodeTransportError(body, cloudflarePolicy("")); err == nil {
		t.Fatal("strict canonical decode accepted a Workors AI errors[] envelope")
	}

	transportError, _, err := (OpenAIChatCodec{}).DecodeTransportError(
		body, cloudflarePolicy(llmprotocol.ResponseVendorCloudflare),
	)
	if err != nil {
		t.Fatalf("DecodeTransportError() error = %v", err)
	}
	if transportError.Error == nil {
		t.Fatal("no transport error decoded")
	}
	if transportError.Error.Code != "5007" {
		t.Fatalf("transport error code = %q, want 5007", transportError.Error.Code)
	}
	if transportError.Error.Category != llmprotocol.ErrorInvalidRequest {
		t.Fatalf("transport error category = %q, want an invalid request", transportError.Error.Category)
	}
	if strings.TrimSpace(transportError.Error.Message) == "" {
		t.Fatal("transport error message is empty; the provider message was lost")
	}
}
