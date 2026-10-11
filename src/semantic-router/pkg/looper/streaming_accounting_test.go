package looper

import (
	"context"
	"testing"
)

func TestStreamingUsageSettlement(t *testing.T) {
	tests := []struct {
		name, usage string
		charged     int64
		unknown     int64
	}{
		{"null block", "null", 100, 1},
		{"null fields", `{"prompt_tokens":null,"completion_tokens":null,"total_tokens":null}`, 100, 1},
		{"missing fields", `{}`, 100, 1},
		{"partial null", `{"prompt_tokens":null,"completion_tokens":2,"total_tokens":null}`, 100, 1},
		{"zero", `{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`, 0, 0},
		{"total only", `{"total_tokens":7}`, 7, 0},
		{"pair", `{"prompt_tokens":4,"completion_tokens":2,"total_tokens":null}`, 6, 0},
		{"negative", `{"prompt_tokens":-1,"completion_tokens":-1,"total_tokens":-2}`, 100, 1},
		{"string", `{"total_tokens":"0"}`, 100, 1},
		{"fraction", `{"total_tokens":0.5}`, 100, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte("data: {\"usage\":" + tt.usage + "}\n")
			usage, presence := parseStreamingUsageWithPresence(body)
			response := &ModelResponse{Usage: usage, UsagePresent: presence}
			controller := NewBudgetController(BudgetLimits{MaxCalls: 2, MaxTotalTokens: 150})
			info := CallInfo{EstimatedTotalTokens: 100}
			reservation, err := controller.BeforeCall(context.Background(), info)
			if err != nil {
				t.Fatal(err)
			}
			controller.AfterCall(context.Background(), info, reservation, CallResult{Response: response})
			got := controller.Snapshot()
			if got.Tokens != tt.charged || got.UnknownUsageCalls != tt.unknown || got.ActiveCalls != 0 || got.ReservedTokens != 0 {
				t.Fatalf("snapshot = %+v; want tokens %d, unknown %d", got, tt.charged, tt.unknown)
			}
			if response.UsageKnown() != (tt.unknown == 0) {
				t.Fatalf("UsageKnown = %v", response.UsageKnown())
			}
			_, err = controller.BeforeCall(context.Background(), info)
			if tt.unknown == 1 && !IsBudgetExhausted(err) {
				t.Fatalf("missing usage bypassed next admission: %v", err)
			}
		})
	}
}
