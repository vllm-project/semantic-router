package extproc

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEndpointHallucinationHasVerdictWithoutScore(t *testing.T) {
	server, _ := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, "body")
	if !router.Classifier.IsHallucinationDetectorReady() {
		t.Fatal("endpoint detector is not prepared")
	}
	result, err := router.Classifier.DetectHallucination(context.Background(), hallucinationContext, "height?", hallucinationAnswer)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HallucinationDetected || result.ScoreAvailable {
		t.Fatalf("endpoint result=%+v", result)
	}
	evidence := &ResponseHallucinationEvidence{Detected: true, Spans: []string{"450 meters"}}
	router.publishHallucinationSignal(ctx, []config.HallucinationRule{hallucinationRule()}, evidence, "")
	router.consumeHallucinationSignal(ctx)
	if !ctx.HallucinationDetected || ctx.HallucinationScoreAvailable {
		t.Fatal("missing-score verdict lost in plugin")
	}
	if len(ctx.VSRSignalConfidences) > 0 {
		t.Fatalf("invented probabilities: %v", ctx.VSRSignalConfidences)
	}
	warning := router.buildHallucinationWarningText(ctx, true)
	if strings.Contains(warning, "confidence") || strings.Contains(warning, "%") {
		t.Fatalf("invented warning probability: %s", warning)
	}
	outcome := hallucinationReplayOutcome(ctx, hallucinationRule(), time.Now(), "body")
	if outcome.Verdict != "detected" || outcome.Metadata["score_available"] != "false" {
		t.Fatalf("replay outcome=%+v", outcome)
	}
}

func TestHallucinationRawScoreIsNotPublishedAsProbability(t *testing.T) {
	result := classification.EvaluateResponseHallucinationSignal([]config.HallucinationRule{{Name: "grounding"}}, true, .8, "", classification.HallucinationScore{Available: true, Kind: "max_hallucinated_token_score"})
	if len(result.Confidences) != 0 || result.Values["hallucination:grounding"] != float64(float32(.8)) {
		t.Fatalf("wrong score semantics: %+v", result)
	}
	warning := hallucinationWarningPrefix(true, "max_hallucinated_token_score", .8)
	if strings.Contains(warning, "80%") || !strings.Contains(warning, "score: 0.800") {
		t.Fatalf("wrong warning semantics: %s", warning)
	}
}
