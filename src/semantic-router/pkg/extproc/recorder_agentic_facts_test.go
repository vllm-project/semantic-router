package extproc

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agenticfacts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func validateAgenticFactsForReplayTest(raw string) agenticfacts.Result {
	return agenticfacts.Validate([]byte(raw), agenticfacts.Bounds{}, time.Now())
}

func agenticFactsExpiresAt(offset time.Duration) string {
	return time.Now().Add(offset).UTC().Format(time.RFC3339)
}

func TestReplayAgenticFactsOutcome(t *testing.T) {
	tests := []struct {
		name        string
		result      agenticfacts.Result
		wantStatus  string
		wantReasons []string
	}{
		{
			name:   "no envelope writes nothing",
			result: agenticfacts.Result{},
		},
		{
			name: "valid envelope with facts is accepted",
			result: validateAgenticFactsForReplayTest(fmt.Sprintf(
				`{"version":"1","delegated_role":"researcher","expires_at":%q}`,
				agenticFactsExpiresAt(30*time.Second),
			)),
			wantStatus: replayAgenticFactsStatusAccepted,
		},
		{
			name: "untrusted envelope has a bare reason",
			result: agenticfacts.Result{
				Rejections: []agenticfacts.Rejection{{Reason: agenticfacts.ReasonUntrusted}},
			},
			wantStatus:  replayAgenticFactsStatusRejected,
			wantReasons: []string{"untrusted"},
		},
		{
			name:        "malformed envelope has a bare reason",
			result:      validateAgenticFactsForReplayTest("not json at all"),
			wantStatus:  replayAgenticFactsStatusRejected,
			wantReasons: []string{"malformed"},
		},
		{
			name: "field rejection is written as field:reason",
			result: validateAgenticFactsForReplayTest(fmt.Sprintf(
				`{"version":"1","expires_at":%q}`,
				agenticFactsExpiresAt(-time.Hour),
			)),
			wantStatus:  replayAgenticFactsStatusRejected,
			wantReasons: []string{"expires_at:expired"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status, reasons := replayAgenticFactsOutcome(tt.result)
			if status != tt.wantStatus {
				t.Fatalf("status = %q, want %q", status, tt.wantStatus)
			}
			if !reflect.DeepEqual(reasons, tt.wantReasons) {
				t.Fatalf("reasons = %#v, want %#v", reasons, tt.wantReasons)
			}
		})
	}
}

// The validator records one rejection per bad capability, so the same entry
// can repeat. The Replay row must hold each entry once.
func TestReplayAgenticFactsReasonsDropsRepeats(t *testing.T) {
	rejections := []agenticfacts.Rejection{
		{Field: "expires_at", Reason: agenticfacts.ReasonExpired},
		{Field: "required_capabilities", Reason: agenticfacts.ReasonMalformed},
		{Field: "required_capabilities", Reason: agenticfacts.ReasonMalformed},
		{Field: "required_capabilities", Reason: agenticfacts.ReasonMalformed},
	}

	got := replayAgenticFactsReasons(rejections)

	want := []string{"expires_at:expired", "required_capabilities:malformed"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reasons = %#v, want %#v", got, want)
	}
}

func TestBuildReplayRouteDiagnosticsRecordsAgenticFacts(t *testing.T) {
	rejected := agenticfacts.Result{
		Rejections: []agenticfacts.Rejection{{Reason: agenticfacts.ReasonUntrusted}},
	}

	t.Run("without a learning policy", func(t *testing.T) {
		ctx := &RequestContext{AgenticFacts: rejected}

		diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "model-a", "balance", 0, 0)

		assertReplayAgenticFactsRejectedUntrusted(t, diagnostics.AgenticFactsStatus, diagnostics.AgenticFactsReasons)
	})

	// buildReplayRouteDiagnostics returns early when a protection policy is
	// active. The fields must be filled before that return, not after it.
	t.Run("with a protection learning policy", func(t *testing.T) {
		ctx := &RequestContext{
			AgenticFacts: rejected,
			VSRLearningPolicy: &routerLearningPolicy{
				Method: routerLearningMethodProtection,
				Mode:   config.DecisionAdaptationModeObserve,
				Action: "stay",
			},
		}

		diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "model-a", "balance", 0, 0)

		// Only the protection branch sets observe_only, so this proves the
		// early return was taken.
		if diagnostics.DecisionReason != "observe_only" {
			t.Fatalf("DecisionReason = %q, expected the protection policy branch to run", diagnostics.DecisionReason)
		}
		assertReplayAgenticFactsRejectedUntrusted(t, diagnostics.AgenticFactsStatus, diagnostics.AgenticFactsReasons)
	})
}

func assertReplayAgenticFactsRejectedUntrusted(t *testing.T, status string, reasons []string) {
	t.Helper()
	if status != replayAgenticFactsStatusRejected {
		t.Fatalf("AgenticFactsStatus = %q, want %q", status, replayAgenticFactsStatusRejected)
	}
	if !reflect.DeepEqual(reasons, []string{"untrusted"}) {
		t.Fatalf("AgenticFactsReasons = %#v, want [untrusted]", reasons)
	}
}

// No value the caller sent may reach the Replay row, whether the envelope was
// accepted or rejected. Only schema field names and reason codes are allowed.
func TestBuildReplayRouteDiagnosticsNeverStoresAgenticFactsValues(t *testing.T) {
	const canary = "leak-canary"
	tooLong := canary + strings.Repeat("x", 200)

	tests := []struct {
		name       string
		envelope   string
		wantStatus string
	}{
		{
			name: "accepted envelope",
			envelope: fmt.Sprintf(
				`{"version":"1","delegated_role":%q,"task_phase":%q,"required_capabilities":[%q],"trust_boundary":{"tenant":%q},"lineage":{"root_invocation_id":%q},"expires_at":%q}`,
				canary, canary, canary, canary, canary, agenticFactsExpiresAt(30*time.Second),
			),
			wantStatus: replayAgenticFactsStatusAccepted,
		},
		{
			name: "rejected envelope",
			envelope: fmt.Sprintf(
				`{"version":"1","delegated_role":%q,"required_capabilities":[%q],"trust_boundary":{"tenant":%q},"expires_at":%q}`,
				canary, tooLong, tooLong, agenticFactsExpiresAt(-time.Hour),
			),
			wantStatus: replayAgenticFactsStatusRejected,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{AgenticFacts: validateAgenticFactsForReplayTest(tt.envelope)}

			diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "model-a", "balance", 0, 0)

			if diagnostics.AgenticFactsStatus != tt.wantStatus {
				t.Fatalf("AgenticFactsStatus = %q, want %q", diagnostics.AgenticFactsStatus, tt.wantStatus)
			}
			encoded, err := json.Marshal(diagnostics)
			if err != nil {
				t.Fatalf("marshal diagnostics: %v", err)
			}
			if strings.Contains(string(encoded), canary) {
				t.Fatalf("Replay diagnostics contain a caller value: %s", encoded)
			}
		})
	}
}
