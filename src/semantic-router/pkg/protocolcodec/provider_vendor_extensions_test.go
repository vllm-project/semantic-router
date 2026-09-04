package protocolcodec

import (
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

func TestStripProviderVendorExtensionsRemovesAzureFields(t *testing.T) {
	stripped, removed := stripProviderVendorExtensions([]byte(azureChatCompletion), llmprotocol.VendorAzure)

	want := []string{
		"choices[].content_filter_results",
		"prompt_filter_results",
		"routing",
		"usage.latency_checkpoint",
	}
	if !reflect.DeepEqual(removed, want) {
		t.Errorf("removed = %v, want %v", removed, want)
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

	stripped, removed := stripProviderVendorExtensions([]byte(canonical), llmprotocol.VendorAzure)

	if removed != nil {
		t.Errorf("removed = %v, want nil", removed)
	}
	if string(stripped) != canonical {
		t.Errorf("stripped = %s, want the original body unchanged", stripped)
	}
}

// A vendor name is only ignored at the path the provider emits it. The same
// token nested somewhere else is still an unknown field.
func TestStripProviderVendorExtensionsIsPathScoped(t *testing.T) {
	body := `{"choices":[{"index":0,"message":{"role":"assistant","routing":"nope"}}]}`

	stripped, removed := stripProviderVendorExtensions([]byte(body), llmprotocol.VendorAzure)

	if removed != nil {
		t.Errorf("removed = %v, want nil for a non-allowlisted path", removed)
	}
	if !strings.Contains(string(stripped), `"routing"`) {
		t.Error("stripped a routing field that is not at the allowlisted path")
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

// An unknown field must be identifiable from the error alone. Before this the
// operator saw only "contains a non-canonical field" with no way to tell which.
func TestDecodeProviderWireErrorNamesTheOffendingField(t *testing.T) {
	var target struct {
		Model string `json:"model"`
	}

	err := decodeProviderWire([]byte(`{"model":"m","surprise_field":1}`), &target, llmprotocol.DefaultPolicy())
	if err == nil {
		t.Fatal("decodeProviderWire() error = nil, want a rejection")
	}
	if !strings.Contains(err.Error(), "surprise_field") {
		t.Errorf("error = %v, want it to name surprise_field", err)
	}
}
