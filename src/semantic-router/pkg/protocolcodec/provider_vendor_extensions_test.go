package protocolcodec

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// azureChatCompletion is the shape Azure OpenAI / AI Foundry actually returns:
// a canonical OpenAI chat completion decorated with Azure's own fields at four
// paths. Reported in issue #3496.
const azureChatCompletion = `{
  "id": "chatcmpl-1",
  "object": "chat.completion",
  "created": 1,
  "model": "gpt-4.1-mini",
  "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}],
  "routing": {"region": "eastus"},
  "choices": [{
    "index": 0,
    "finish_reason": "stop",
    "message": {"role": "assistant", "content": "hi"},
    "content_filter_results": {"hate": {"filtered": false, "severity": "safe"}}
  }],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 1,
    "total_tokens": 2,
    "latency_checkpoint": {"time_to_first_token_ms": 12}
  }
}`

// azureTarget is a canonical-shaped subset: the schema decides what counts as
// a decoration, so anything Azure adds beyond these fields is a drop.
type azureTarget struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func TestStripProviderVendorExtensionsRemovesAzureFields(t *testing.T) {
	stripped, dropped := stripProviderVendorExtensions([]byte(azureChatCompletion), reflect.TypeOf(azureTarget{}))

	want := []string{
		"choices[].content_filter_results",
		"prompt_filter_results",
		"routing",
		"usage.latency_checkpoint",
	}
	if !reflect.DeepEqual(dropped, want) {
		t.Errorf("dropped = %v, want %v", dropped, want)
	}

	for _, field := range []string{"prompt_filter_results", "\"routing\"", "content_filter_results", "latency_checkpoint"} {
		if strings.Contains(string(stripped), field) {
			t.Errorf("stripped body still contains %s", field)
		}
	}
	for _, kept := range []string{`"id"`, `"model"`, `"finish_reason"`, `"total_tokens"`} {
		if !strings.Contains(string(stripped), kept) {
			t.Errorf("stripped body dropped canonical field %s", kept)
		}
	}
}

func TestStripProviderVendorExtensionsLeavesCanonicalBodyUntouched(t *testing.T) {
	canonical := `{"id":"chatcmpl-1","model":"gpt-4.1-mini","choices":[{"index":0}]}`

	stripped, dropped := stripProviderVendorExtensions([]byte(canonical), reflect.TypeOf(azureTarget{}))

	if dropped != nil {
		t.Errorf("dropped = %v, want nil", dropped)
	}
	if string(stripped) != canonical {
		t.Errorf("stripped = %s, want the original body unchanged", stripped)
	}
}

// Decorations nested inside a canonical object are dropped at their real path,
// not just at the top level.
func TestStripProviderVendorExtensionsDropsNestedDecorations(t *testing.T) {
	body := `{"id":"a","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi","azure_note":"x"}}]}`

	stripped, dropped := stripProviderVendorExtensions([]byte(body), reflect.TypeOf(azureTarget{}))

	want := []string{"choices[].message.azure_note"}
	if !reflect.DeepEqual(dropped, want) {
		t.Errorf("dropped = %v, want %v", dropped, want)
	}
	if strings.Contains(string(stripped), "azure_note") {
		t.Error("stripped body still contains the nested decoration")
	}
}

func TestProviderVendorExtensionsAllowedOnlyForKnownVendors(t *testing.T) {
	for _, vendor := range []string{"", "not-a-vendor", "openai"} {
		t.Run("vendor="+vendor, func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			policy.ResponseVendor = vendor

			if providerVendorExtensionsAllowed(policy) {
				t.Errorf("providerVendorExtensionsAllowed(%q) = true, want false", vendor)
			}
		})
	}
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.VendorAzure
	if !providerVendorExtensionsAllowed(policy) {
		t.Error("providerVendorExtensionsAllowed(azure) = false, want true")
	}
}

// The point of shape 3: a decoration Azure has not shipped yet must not break
// the router the way a fixed name list would.
func TestDecodeProviderWireAcceptsUnanticipatedAzureFields(t *testing.T) {
	var target azureTarget
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.VendorAzure

	body := `{"id":"a","object":"chat.completion","created":1,"model":"m",` +
		`"a_field_azure_ships_next_year":{"nested":true},` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}]}`

	dropped, err := decodeProviderWireVendorAware([]byte(body), &target, policy)
	if err != nil {
		t.Fatalf("decodeProviderWireVendorAware() error = %v, want nil", err)
	}
	if !reflect.DeepEqual(dropped, []string{"a_field_azure_ships_next_year"}) {
		t.Errorf("dropped = %v, want the unanticipated field reported", dropped)
	}
	if target.Model != "m" {
		t.Errorf("Model = %q, want m", target.Model)
	}
}

func TestDecodeProviderWireAcceptsAzureDecoratedResponse(t *testing.T) {
	var target struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index        int    `json:"index"`
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.VendorAzure
	if err := decodeProviderWire([]byte(azureChatCompletion), &target, policy); err != nil {
		t.Fatalf("decodeProviderWire() error = %v, want nil", err)
	}
	if target.Model != "gpt-4.1-mini" {
		t.Errorf("Model = %q, want gpt-4.1-mini", target.Model)
	}
	if len(target.Choices) != 1 || target.Choices[0].Message.Content != "hi" {
		t.Errorf("Choices = %+v, want one choice carrying the assistant content", target.Choices)
	}
	if target.Usage.TotalTokens != 2 {
		t.Errorf("Usage.TotalTokens = %d, want 2", target.Usage.TotalTokens)
	}
}

// The allowlist must not become a blanket accept: an unknown field outside it
// still fails with the canonical upstream error.
func TestDecodeProviderWireStillRejectsUnknownFields(t *testing.T) {
	var target struct {
		Model string `json:"model"`
	}

	err := decodeProviderWire([]byte(`{"model":"m","surprise_field":1}`), &target, llmprotocol.DefaultPolicy())
	if err == nil {
		t.Fatal("decodeProviderWire() error = nil, want a rejection")
	}
	if !strings.Contains(err.Error(), "invalid_upstream_json") {
		t.Errorf("error = %v, want invalid_upstream_json", err)
	}
}

// Without the vendor allowance the same Azure body is still rejected: the
// backend dialect, not the field name, is what grants the exemption.
func TestDecodeProviderWireRejectsAzureFieldsWithoutVendorAllowance(t *testing.T) {
	var target struct {
		ID    string `json:"id"`
		Model string `json:"model"`
	}

	err := decodeProviderWire([]byte(azureChatCompletion), &target, llmprotocol.DefaultPolicy())
	if err == nil {
		t.Fatal("decodeProviderWire() error = nil, want rejection without a vendor allowance")
	}
	if !strings.Contains(err.Error(), "invalid_upstream_json") {
		t.Errorf("error = %v, want invalid_upstream_json", err)
	}
}

// The field name is what an operator needs and what a caller must not see.
// ProtocolError.Message is serialized into the client error body, so the name
// belongs in the cause and nowhere else.
func TestDecodeProviderWireKeepsTheOffendingFieldOutOfTheClientMessage(t *testing.T) {
	var target struct {
		Model string `json:"model"`
	}

	err := decodeProviderWire([]byte(`{"model":"m","surprise_field":1}`), &target, llmprotocol.DefaultPolicy())
	if err == nil {
		t.Fatal("decodeProviderWire() error = nil, want a rejection")
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("error = %v, want a ProtocolError", err)
	}
	if strings.Contains(protocolError.Message, "surprise_field") {
		t.Errorf("Message = %q, must not expose the upstream field name to the caller", protocolError.Message)
	}
	cause := errors.Unwrap(protocolError)
	if cause == nil || !strings.Contains(cause.Error(), "surprise_field") {
		t.Errorf("cause = %v, want it to name surprise_field for the operator log", cause)
	}
}

// The whole point of dropping rather than rejecting is that the drop is
// observable. Decoding a decorated Azure response must surface one
// DiagnosticDropped per decoration.
func TestDecodeResponseReportsVendorExtensionsAsDiagnostics(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.VendorAzure

	_, _, diagnostics, err := OpenAIChatCodec{}.DecodeResponse([]byte(azureChatCompletion), policy)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v, want nil", err)
	}

	dropped := map[string]bool{}
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped {
			dropped[diagnostic.Field] = true
		}
	}
	for _, field := range []string{
		"prompt_filter_results",
		"routing",
		"choices[].content_filter_results",
		"usage.latency_checkpoint",
	} {
		if !dropped[field] {
			t.Errorf("no dropped diagnostic for %q; diagnostics = %+v", field, diagnostics)
		}
	}
}
