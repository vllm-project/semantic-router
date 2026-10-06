package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

func TestGuardRiskObservationContextReplay(t *testing.T) {
	for _, risk := range []float32{0, .2} {
		signals := &classification.SignalResults{JailbreakScoreAvailable: true, JailbreakConfidence: risk, SignalValues: map[string]float64{"jailbreak:limit": float64(risk)}}
		ctx := &RequestContext{}
		(&OpenAIRouter{}).applySignalResultsToContext(ctx, signals)
		if ctx.JailbreakDetected || !ctx.JailbreakScoreAvailable || ctx.JailbreakConfidence != risk {
			t.Fatalf("context dropped non-match risk: detected=%v available=%v score=%v", ctx.JailbreakDetected, ctx.JailbreakScoreAvailable, ctx.JailbreakConfidence)
		}
		signals.SignalValues["jailbreak:limit"] = .9
		if ctx.VSRSignalValues["jailbreak:limit"] != float64(risk) {
			t.Fatal("context observation aliases caller")
		}
		record := buildReplayRoutingRecord(ctx, "model", "model", "")
		raw, err := json.Marshal(record)
		if err != nil {
			t.Fatal(err)
		}
		var wire map[string]any
		if err = json.Unmarshal(raw, &wire); err != nil {
			t.Fatal(err)
		}
		if wire["jailbreak_score_available"] != true {
			t.Fatalf("replay omitted availability: %s", raw)
		}
		score, ok := wire["jailbreak_confidence"].(float64)
		if !ok || float32(score) != risk {
			t.Fatalf("replay omitted actual risk: %s", raw)
		}
	}
}
