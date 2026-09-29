package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// OpenAI backends may omit usage on non-streaming responses. The Anthropic
// Messages projection must emit an explicit zero-valued usage object with an
// approximation diagnostic instead of failing the translation; the streaming
// path already projects the same zero-valued object for this case.
func TestAnthropicMessagesProjectsUnavailableUsageAsZeroObject(t *testing.T) {
	response := llmprotocol.Response{
		Generation: 1, ID: "response_1", Model: "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_1", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "done"}},
		}},
		Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
	}
	body, diagnostics, err := (AnthropicMessagesCodec{}).EncodeResponse(
		response, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy(),
	)
	if err != nil {
		t.Fatalf("EncodeResponse rejected unavailable usage: %v", err)
	}
	var wire struct {
		Usage *struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded Messages body: %v\n%s", err, body)
	}
	if wire.Usage == nil {
		t.Fatalf("Messages body must carry an explicit usage object: %s", body)
	}
	if wire.Usage.InputTokens != 0 || wire.Usage.OutputTokens != 0 {
		t.Fatalf("unavailable usage must project to zero-valued tokens: %s", body)
	}
	approximated := false
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "usage" && diagnostic.Action == llmprotocol.DiagnosticApproximated {
			approximated = true
		}
	}
	if !approximated {
		t.Fatalf("expected an approximation diagnostic for usage, got %+v", diagnostics)
	}
}

// Available usage keeps its authoritative projection untouched.
func TestAnthropicMessagesKeepsAvailableUsageProjection(t *testing.T) {
	input, output, total := int64(7), int64(3), int64(10)
	response := llmprotocol.Response{
		Generation: 1, ID: "response_1", Model: "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_1", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "done"}},
		}},
		Usage: llmprotocol.Usage{
			State:         llmprotocol.UsageAvailable,
			InputUncached: llmprotocol.TokenCount{Value: &input, Provenance: llmprotocol.UsageAuthoritative},
			OutputTotal:   llmprotocol.TokenCount{Value: &output, Provenance: llmprotocol.UsageAuthoritative},
			Total:         llmprotocol.TokenCount{Value: &total, Provenance: llmprotocol.UsageDerived},
		},
	}
	body, diagnostics, err := (AnthropicMessagesCodec{}).EncodeResponse(
		response, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy(),
	)
	if err != nil {
		t.Fatalf("EncodeResponse failed: %v", err)
	}
	var wire struct {
		Usage *struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded Messages body: %v\n%s", err, body)
	}
	if wire.Usage == nil || wire.Usage.InputTokens != 7 || wire.Usage.OutputTokens != 3 {
		t.Fatalf("available usage projection changed: %s", body)
	}
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "usage" {
			t.Fatalf("available usage must not emit a usage diagnostic: %+v", diagnostics)
		}
	}
}

// A known component count with both totals absent keeps its count: the
// unavailable zero projection applies only when nothing is known, because a
// known component is information the exact projection can still emit.
func TestAnthropicMessagesKeepsKnownComponentWhenTotalsAreAbsent(t *testing.T) {
	input := int64(7)
	response := llmprotocol.Response{
		Generation: 1, ID: "response_1", Model: "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_1", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "done"}},
		}},
		Usage: llmprotocol.Usage{
			State:         llmprotocol.UsageAvailable,
			InputUncached: llmprotocol.TokenCount{Value: &input, Provenance: llmprotocol.UsageAuthoritative},
		},
	}
	body, _, err := (AnthropicMessagesCodec{}).EncodeResponse(
		response, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy(),
	)
	if err != nil {
		t.Fatalf("EncodeResponse failed: %v", err)
	}
	var wire struct {
		Usage *struct {
			InputTokens int64 `json:"input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded Messages body: %v\n%s", err, body)
	}
	if wire.Usage.InputTokens != input {
		t.Fatalf("known component must survive the projection, got input_tokens=%d: %s", wire.Usage.InputTokens, body)
	}
}

// When a neutral response contains only output components, the Messages
// projection derives its required total from those components instead of
// silently replacing the known count with zero.
func TestAnthropicMessagesDerivesOutputTotalFromKnownComponents(t *testing.T) {
	three := int64(3)
	four := int64(4)
	tests := []struct {
		name             string
		reasoning        *int64
		other            *int64
		want             int64
		wantApproximated bool
	}{
		{name: "reasoning only", reasoning: &three, want: 3, wantApproximated: true},
		{name: "other only", other: &four, want: 4, wantApproximated: true},
		{name: "both components", reasoning: &three, other: &four, want: 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := llmprotocol.Response{
				Generation: 1, ID: "response_1", Model: "public-model",
				Output: []llmprotocol.OutputItem{{
					ID: "item_1", Role: llmprotocol.RoleAssistant,
					Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "done"}},
				}},
				Usage: llmprotocol.Usage{
					State:           llmprotocol.UsageAvailable,
					OutputReasoning: llmprotocol.TokenCount{Value: tt.reasoning, Provenance: llmprotocol.UsageAuthoritative},
					OutputOther:     llmprotocol.TokenCount{Value: tt.other, Provenance: llmprotocol.UsageAuthoritative},
				},
			}
			body, diagnostics, err := (AnthropicMessagesCodec{}).EncodeResponse(
				response, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy(),
			)
			if err != nil {
				t.Fatalf("EncodeResponse failed: %v", err)
			}
			var wire struct {
				Usage *struct {
					OutputTokens int64 `json:"output_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(body, &wire); err != nil {
				t.Fatalf("decode encoded Messages body: %v\n%s", err, body)
			}
			got := int64(-1)
			if wire.Usage != nil {
				got = wire.Usage.OutputTokens
			}
			if got != tt.want {
				t.Fatalf("known output components must survive the projection, got output_tokens=%d: %s", got, body)
			}
			approximated := false
			for _, diagnostic := range diagnostics {
				if diagnostic.Field == "usage" && diagnostic.Action == llmprotocol.DiagnosticApproximated {
					approximated = true
				}
			}
			if approximated != tt.wantApproximated {
				t.Fatalf("usage approximated = %v, want %v: %+v", approximated, tt.wantApproximated, diagnostics)
			}
		})
	}
}
