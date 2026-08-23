package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponseUsageFromTerminalRequiresAuthoritativeEvidence(t *testing.T) {
	tests := []struct {
		name                 string
		input, output, total llmprotocol.UsageProvenance
		invalid              bool
	}{
		{name: "authoritative", input: llmprotocol.UsageAuthoritative, output: llmprotocol.UsageAuthoritative, total: llmprotocol.UsageAuthoritative},
		{name: "provable derived total", input: llmprotocol.UsageAuthoritative, output: llmprotocol.UsageAuthoritative, total: llmprotocol.UsageDerived},
		{name: "estimated input", input: llmprotocol.UsageEstimated, output: llmprotocol.UsageAuthoritative, total: llmprotocol.UsageDerived, invalid: true},
		{name: "derived output", input: llmprotocol.UsageAuthoritative, output: llmprotocol.UsageDerived, total: llmprotocol.UsageDerived, invalid: true},
		{name: "estimated total", input: llmprotocol.UsageAuthoritative, output: llmprotocol.UsageAuthoritative, total: llmprotocol.UsageEstimated, invalid: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input, output, total := int64(7), int64(5), int64(12)
			record := backendinvoker.ResponseTerminalRecord{Terminal: backendinvoker.ResponseTerminal{
				StopReason: llmprotocol.StopEndTurn,
				Usage: llmprotocol.Usage{
					State:       llmprotocol.UsageAvailable,
					InputTotal:  llmprotocol.TokenCount{Value: &input, Provenance: test.input},
					OutputTotal: llmprotocol.TokenCount{Value: &output, Provenance: test.output},
					Total:       llmprotocol.TokenCount{Value: &total, Provenance: test.total},
				},
			}}
			usage := responseUsageFromTerminal(record)
			if usage.invalid != test.invalid {
				t.Fatalf("invalid=%v, want %v", usage.invalid, test.invalid)
			}
			if !usage.invalid && (usage.promptTokens != 7 || usage.completionTokens != 5 || usage.totalTokens != 12) {
				t.Fatalf("unexpected usage: %#v", usage)
			}
		})
	}
}

func TestResponseTerminalFailureReasonsAreClosedAndSafe(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{name: "capacity", err: backendinvoker.ErrResponseTerminalCapacity, want: "response_terminal_capacity"},
		{name: "unavailable", err: backendinvoker.ErrResponseTerminalUnavailable, want: "response_terminal_unavailable"},
		{name: "invalid", err: backendinvoker.ErrResponseTerminalInvalid, want: "response_terminal_invalid"},
		{name: "wrapped", err: errors.New("private backend detail"), want: "response_terminal_unavailable"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := responseTerminalFailureReason(test.err); got != test.want {
				t.Fatalf("reason = %q, want %q", got, test.want)
			}
		})
	}
	if reason := usageFromResponse(responseUsageMetrics{invalid: true}).Reason; reason != "authoritative_usage_missing" {
		t.Fatalf("default invalid reason = %q", reason)
	}
}

func TestResponseUsageFromTerminalRejectsEstimatedCacheBuckets(t *testing.T) {
	input, output, total, cached := int64(7), int64(5), int64(12), int64(2)
	record := backendinvoker.ResponseTerminalRecord{Terminal: backendinvoker.ResponseTerminal{
		StopReason: llmprotocol.StopEndTurn,
		Usage: llmprotocol.Usage{
			State:      llmprotocol.UsageAvailable,
			InputTotal: authoritativeCountForSettlement(&input), OutputTotal: authoritativeCountForSettlement(&output),
			Total:          authoritativeCountForSettlement(&total),
			InputCacheRead: llmprotocol.TokenCount{Value: &cached, Provenance: llmprotocol.UsageEstimated},
		},
	}}
	if usage := responseUsageFromTerminal(record); !usage.invalid {
		t.Fatalf("estimated cache evidence was charged: %#v", usage)
	}
}

func authoritativeCountForSettlement(value *int64) llmprotocol.TokenCount {
	return llmprotocol.TokenCount{Value: value, Provenance: llmprotocol.UsageAuthoritative}
}
