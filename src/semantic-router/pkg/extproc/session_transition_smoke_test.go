package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// TestSmokeModelTransitionLog is a smoke check that verifies the full path from
// TTFT write → CacheWarmthEstimate → maybeEmitTransitionEvent → structured log
// entry, mirroring the illustrative event from the issue spec:
//
//	{
//	  "session_id":             "sess_123",
//	  "turn_index":             4,
//	  "from_model":             "deepseek-v3",
//	  "to_model":               "deepseek-r1",
//	  "ttft_ms":                ~420,
//	  "cache_warmth_estimate":  [0,1],
//	  "previous_response_id":   "resp_abc"
//	}
func TestSmokeModelTransitionLog(t *testing.T) {
	latency.ResetTTFT()
	sessiontelemetry.ResetTransitionsForTesting()

	// ── Seed TTFT history ─────────────────────────────────────────────────
	// 50 observations from 0.2s to 1.0s to give stable percentile anchors.
	for i := 0; i < 50; i++ {
		v := 0.2 + 0.8*float64(i)/49.0
		latency.UpdateTTFT("deepseek-r1", v)
	}

	// ── Build RequestContext ──────────────────────────────────────────────
	ctx := &RequestContext{
		RequestModel:  "deepseek-r1",
		SessionID:     "sess_123",
		TurnIndex:     4,
		PreviousModel: "deepseek-v3",
		ResponseAPICtx: &ResponseAPIContext{
			PreviousResponseID: "resp_abc",
		},
	}

	// Write CacheWarmthEstimate as the TTFT path does, then emit the event.
	ctx.TTFTSeconds = 0.42
	ctx.TTFTRecorded = true
	ctx.CacheWarmthEstimate = latency.EstimateCacheProbability(latency.CacheEstimationInput{
		Model:       ctx.RequestModel,
		TTFTSeconds: ctx.TTFTSeconds,
		Now:         time.Now(),
	})
	maybeEmitTransitionEvent(ctx)

	// ── Verify: TransitionLog ─────────────────────────────────────────────
	events := sessiontelemetry.GetTransitions("sess_123")

	t.Logf("── GetTransitions(\"sess_123\") ───────────────────────")
	t.Logf("  event count      : %d  (want 1)", len(events))
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}

	e := events[0]
	t.Logf("  session_id       : %s", e.SessionID)
	t.Logf("  turn_index       : %d", e.TurnIndex)
	t.Logf("  from_model       : %s", e.FromModel)
	t.Logf("  to_model         : %s", e.ToModel)
	t.Logf("  ttft_ms          : %.2f", e.TTFTMs)
	t.Logf("  cache_warmth     : %.4f", e.CacheWarmthEstimate)
	t.Logf("  prev_response_id : %s", e.PreviousResponseID)

	checks := []struct {
		name string
		ok   bool
	}{
		{"session_id == sess_123", e.SessionID == "sess_123"},
		{"turn_index == 4", e.TurnIndex == 4},
		{"from_model == deepseek-v3", e.FromModel == "deepseek-v3"},
		{"to_model == deepseek-r1", e.ToModel == "deepseek-r1"},
		{"ttft_ms > 0", e.TTFTMs > 0},
		{"cache_warmth in [0,1]", e.CacheWarmthEstimate >= 0 && e.CacheWarmthEstimate <= 1},
		{"previous_response_id == resp_abc", e.PreviousResponseID == "resp_abc"},
	}

	t.Logf("── Assertions ──────────────────────────────────────")
	allPassed := true
	for _, c := range checks {
		status := "PASS"
		if !c.ok {
			status = "FAIL"
			allPassed = false
		}
		t.Logf("  [%s] %s", status, c.name)
	}
	if !allPassed {
		t.Fail()
	}
}

func TestMaybeEmitTransitionEvent_NoOpCases(t *testing.T) {
	sessiontelemetry.ResetTransitionsForTesting()

	tests := []struct {
		name string
		ctx  *RequestContext
	}{
		{"nil context", nil},
		{"empty session", &RequestContext{SessionID: "", RequestModel: "m", PreviousModel: "old"}},
		{"empty request model", &RequestContext{SessionID: "s", RequestModel: "", PreviousModel: "old"}},
		{"empty previous model", &RequestContext{SessionID: "s", RequestModel: "m", PreviousModel: ""}},
		{"same model", &RequestContext{SessionID: "s", RequestModel: "m", PreviousModel: "m"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			maybeEmitTransitionEvent(tc.ctx)
			got := sessiontelemetry.GetTransitions("s")
			if len(got) != 0 {
				t.Errorf("expected no transition event, got %d", len(got))
			}
		})
	}
}
