package classification

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestHallucinationDetector_LongContextReachesAnswer guards the 512-token window:
// the context must be windowed, not the answer.
func TestHallucinationDetector_LongContextReachesAnswer(t *testing.T) {
	skipIfNoModel(t)

	toolContext := strings.Repeat("The Eiffel Tower was constructed from 1887 to 1889. It is located in Paris, France and is 330 metres tall. ", 40)
	userQuestion := "When was the Eiffel Tower built?"
	assistantAnswer := "The Eiffel Tower was built in 1950 and stands at 500 meters tall."

	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}
	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	if err := detector.Initialize(); err != nil {
		t.Fatalf("Failed to initialize detector: %v", err)
	}

	result, err := detector.Detect(toolContext, userQuestion, assistantAnswer)
	if err != nil {
		t.Fatalf("Detection failed: %v", err)
	}
	t.Logf("Long context (%d chars): detected=%v confidence=%.3f spans=%v", len(toolContext), result.HallucinationDetected, result.Confidence, result.UnsupportedSpans)
	if !result.HallucinationDetected {
		t.Errorf("Expected hallucination with %d-char context, but none detected", len(toolContext))
	}
}
