package classification

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func newTestHallucinationDetector(t *testing.T) *HallucinationDetector {
	t.Helper()
	cfg := &config.HallucinationModelConfig{ModelID: "models/Vela-1.0-Encoder-307M-Halu", Threshold: 0.5, UseCPU: true}
	routerCfg := &config.RouterConfig{}
	routerCfg.HallucinationMitigation.HallucinationModel = *cfg
	models := testModelRuntime(t, routerCfg, map[string]runtimetest.Model{"@hallucination_detector": servingtest.Grounded()})
	detector, err := NewHallucinationDetector(cfg, models)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = detector.Close() })
	return detector
}

func TestHallucinationDetectorRequiresAModel(t *testing.T) {
	if detector, err := NewHallucinationDetector(nil); err == nil || detector != nil {
		t.Fatal("a nil config must be rejected")
	}
	if detector, err := NewHallucinationDetector(&config.HallucinationModelConfig{Threshold: 0.5}); err == nil || detector != nil {
		t.Fatal("an empty model_id must be rejected")
	}
}

func TestHallucinationDetectorNeedsInitializationAndContext(t *testing.T) {
	detector := newTestHallucinationDetector(t)
	if _, err := detector.Detect(context.Background(), "context", "question", "answer"); err == nil {
		t.Fatal("detection before initialization must fail")
	}
	if err := detector.Initialize(); err != nil {
		t.Fatal(err)
	}
	if _, err := detector.Detect(context.Background(), "", "What is X?", "X is Y"); err == nil {
		t.Fatal("detection without context must fail")
	}
	result, err := detector.Detect(context.Background(), "Some context", "Question?", "")
	if err != nil || result.HallucinationDetected || result.ScoreAvailable {
		t.Fatalf("an empty answer is clean and unscored: %+v %v", result, err)
	}
}

func TestHallucinationDetectorReportsUnsupportedAnswerSpans(t *testing.T) {
	detector := newTestHallucinationDetector(t)
	if err := detector.Initialize(); err != nil {
		t.Fatal(err)
	}
	contextText := "The Eiffel Tower is 330 metres tall"
	answer := "The Eiffel Tower is 330 metres tall and golden"
	result, err := detector.Detect(context.Background(), contextText, "How tall?", answer)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HallucinationDetected || len(result.Spans) != 2 || !result.ScoreAvailable || result.ScoreKind != "max_hallucinated_token_score" {
		t.Fatalf("result = %+v", result)
	}
	for _, span := range result.Spans {
		if answer[span.Start:span.End] != span.Text {
			t.Fatalf("span offsets index the answer bytes: %+v", span)
		}
	}
	explained, err := detector.DetectWithExplanations(context.Background(), contextText, "How tall?", answer)
	if err != nil || len(explained.Spans) != 2 || !strings.Contains(explained.Spans[0].Explanation, "Unsupported claim") {
		t.Fatalf("explanations = %+v %v", explained, err)
	}
	grounded, err := detector.Detect(context.Background(), contextText, "How tall?", "The Eiffel Tower is 330 metres tall")
	if err != nil || grounded.HallucinationDetected {
		t.Fatalf("a grounded answer is clean: %+v %v", grounded, err)
	}
}

// TestHallucinationDetector_JSONSerialization tests that results can be serialized
func TestHallucinationDetector_JSONSerialization(t *testing.T) {
	result := &HallucinationResult{
		HallucinationDetected: true,
		Confidence:            0.85,
		UnsupportedSpans:      []string{"claim 1", "claim 2"},
		SupportedSpans:        []string{"verified claim"},
	}

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("Failed to marshal result: %v", err)
	}

	var decoded HallucinationResult
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	if decoded.HallucinationDetected != result.HallucinationDetected {
		t.Error("HallucinationDetected mismatch after serialization")
	}
	if decoded.Confidence != result.Confidence {
		t.Error("Confidence mismatch after serialization")
	}
	if len(decoded.UnsupportedSpans) != len(result.UnsupportedSpans) {
		t.Error("UnsupportedSpans length mismatch after serialization")
	}
}
