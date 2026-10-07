package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A missing output total beside a known reasoning count zero-fills the wire
// total while settlement keeps the bucket unknown. Both surfaces must name
// the omission instead of presenting the zero as an exact count.
func TestAnthropicPartialOutputTotalOmission(t *testing.T) {
	reasoning, cacheRead, cacheWrite, inputTotal := int64(3), int64(20), int64(10), int64(100)
	response := llmprotocol.Response{
		Generation: 1, ID: "response_1", Model: "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_1", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "done"}},
		}},
		Usage: llmprotocol.Usage{
			State:           llmprotocol.UsageAvailable,
			InputTotal:      llmprotocol.TokenCount{Value: &inputTotal, Provenance: llmprotocol.UsageAuthoritative},
			InputCacheRead:  llmprotocol.TokenCount{Value: &cacheRead, Provenance: llmprotocol.UsageAuthoritative},
			InputCacheWrite: llmprotocol.TokenCount{Value: &cacheWrite, Provenance: llmprotocol.UsageAuthoritative},
			OutputReasoning: llmprotocol.TokenCount{Value: &reasoning, Provenance: llmprotocol.UsageAuthoritative},
		},
	}
	engine := NewBuiltinEngine()
	for _, streaming := range []bool{false, true} {
		var body []byte
		var diagnostics llmprotocol.Diagnostics
		var err error
		if streaming {
			body, diagnostics, err = engine.EncodeResponseStream(
				llmprotocol.AnthropicMessagesV1,
				response,
				llmprotocol.StreamContext{
					PublicModel: response.Model,
					Options:     llmprotocol.StreamOptions{IncludeUsage: boolPointer(true)},
				},
			)
		} else {
			result, encodeErr := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, response, llmprotocol.Envelope{})
			body, diagnostics, err = result.Body, result.Diagnostics, encodeErr
		}
		if err != nil {
			t.Fatal(err)
		}
		assertDiagnosticFields(t, diagnostics, "usage.output")
		if !strings.Contains(string(body), `"output_tokens":0`) {
			t.Fatalf("streaming=%v: the zero-filled total must stay in the wire: %s", streaming, body)
		}
		if streaming && !strings.Contains(string(body), `"thinking_tokens":3`) {
			t.Fatalf("streaming=%v: the known reasoning component must stay in the wire: %s", streaming, body)
		}
	}
	if response.Usage.OutputTotal.Value != nil {
		t.Fatal("encoding fabricated settlement evidence for the unreported output total")
	}
}

// With the output total reported, the zero-fill shape does not arise and no
// output diagnostic is emitted.
func TestAnthropicReportsOutputTotalWithoutDiagnostic(t *testing.T) {
	reasoning, outputTotal := int64(3), int64(7)
	response := llmprotocol.Response{
		Generation: 1, ID: "response_1", Model: "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_1", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "done"}},
		}},
		Usage: llmprotocol.Usage{
			State:           llmprotocol.UsageAvailable,
			OutputTotal:     llmprotocol.TokenCount{Value: &outputTotal, Provenance: llmprotocol.UsageAuthoritative},
			OutputReasoning: llmprotocol.TokenCount{Value: &reasoning, Provenance: llmprotocol.UsageAuthoritative},
		},
	}
	result, err := NewBuiltinEngine().EncodeResponse(llmprotocol.AnthropicMessagesV1, response, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	for _, diagnostic := range result.Diagnostics {
		if diagnostic.Field == "usage.output" {
			t.Fatalf("reported output total must not be diagnosed: %+v", result.Diagnostics)
		}
	}
	var wire struct {
		Usage struct {
			OutputTokens int64 `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatalf("decode encoded Messages body: %v", err)
	}
	if wire.Usage.OutputTokens != outputTotal {
		t.Fatalf("reported output total must survive: %s", result.Body)
	}
}
