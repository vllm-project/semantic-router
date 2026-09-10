package classification

import (
	"context"
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
	if err = detector.Initialize(); err != nil {
		t.Fatalf("Failed to initialize detector: %v", err)
	}

	result, err := detector.Detect(context.Background(), toolContext, userQuestion, assistantAnswer)
	if err != nil {
		t.Fatalf("Detection failed: %v", err)
	}
	t.Logf("Long context (%d chars): detected=%v confidence=%.3f spans=%v", len(toolContext), result.HallucinationDetected, result.Confidence, result.UnsupportedSpans)
	if !result.HallucinationDetected {
		t.Errorf("Expected hallucination with %d-char context, but none detected", len(toolContext))
	}
}

// TestHallucinationDetector_LongAnswerTailIsScanned guards the other half of the
// 512-token window: an unsupported claim at the end of a long answer must be
// found rather than truncated away.
func TestHallucinationDetector_LongAnswerTailIsScanned(t *testing.T) {
	skipIfNoModel(t)

	toolContext, userQuestion, assistantAnswer := longAnswerWithUnsupportedTail(t)

	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}
	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	if err = detector.Initialize(); err != nil {
		t.Fatalf("Failed to initialize detector: %v", err)
	}

	result, err := detector.Detect(context.Background(), toolContext, userQuestion, assistantAnswer)
	if err != nil {
		t.Fatalf("Detection failed: %v", err)
	}
	t.Logf("Long answer (%d words): detected=%v confidence=%.3f spans=%v",
		len(strings.Fields(assistantAnswer)), result.HallucinationDetected, result.Confidence, result.UnsupportedSpans)
	if !result.HallucinationDetected {
		t.Errorf("Expected the unsupported claim at the end of a %d-word answer to be detected",
			len(strings.Fields(assistantAnswer)))
	}
}

// longAnswerWithUnsupportedTail returns a context, a question, and an answer whose
// only unsupported sentence is its last one, long enough that the answer alone
// passes the detector's 512-token window.
func longAnswerWithUnsupportedTail(t *testing.T) (string, string, string) {
	t.Helper()
	toolContext := "The Eiffel Tower is a wrought iron lattice tower in Paris, France. " +
		"It was designed by Gustave Eiffel and completed in 1889 for the World's Fair. " +
		"The tower is 330 metres tall."
	userQuestion := "Tell me about the Eiffel Tower."
	grounded := strings.Repeat("The tower stands in Paris. It is built from wrought iron. "+
		"Gustave Eiffel designed it. It was raised for the World's Fair. ", 30)
	answer := grounded + "The tower was completed in 1912 and it is 550 metres tall."
	if chunks := hallucinationAnswerChunks(answer); len(chunks) < 2 {
		t.Fatalf("fixture needs to chunk, got %d chunk(s)", len(chunks))
	}
	return toolContext, userQuestion, answer
}

// TestHallucinationDetector_LongAnswerTailIsScannedWithNLI covers the second entry
// point, which windows only its premise and truncated the answer the same way.
func TestHallucinationDetector_LongAnswerTailIsScannedWithNLI(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoNLIModel(t)

	toolContext, userQuestion, assistantAnswer := longAnswerWithUnsupportedTail(t)

	detector, err := NewHallucinationDetector(&config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	})
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	if err = detector.Initialize(); err != nil {
		t.Fatalf("Failed to initialize detector: %v", err)
	}
	detector.SetNLIConfig(&config.NLIModelConfig{
		ModelID:   getNLIModelPath(),
		Threshold: 0.7,
		UseCPU:    true,
	})
	if err = detector.InitializeNLI(); err != nil {
		t.Fatalf("Failed to initialize NLI: %v", err)
	}

	result, err := detector.DetectWithNLI(context.Background(), toolContext, userQuestion, assistantAnswer)
	if err != nil {
		t.Fatalf("Detection failed: %v", err)
	}
	t.Logf("Long answer with NLI (%d words): detected=%v confidence=%.3f spans=%d",
		len(strings.Fields(assistantAnswer)), result.HallucinationDetected, result.Confidence, len(result.Spans))
	if !result.HallucinationDetected {
		t.Errorf("Expected the unsupported claim at the end of a %d-word answer to be detected",
			len(strings.Fields(assistantAnswer)))
	}
}

// TestMergeChunkConfidence covers the fold that turns per chunk verdicts into one
// answer verdict, which runs whether or not the model is present.
func TestMergeChunkConfidence(t *testing.T) {
	tests := []struct {
		name             string
		merged           bool
		mergedConfidence float32
		chunk            bool
		chunkConfidence  float32
		want             float32
	}{
		{"first detection replaces the clean confidence", false, 1.0, true, 0.8, 0.8},
		{"a stronger detection wins", true, 0.8, true, 0.9, 0.9},
		{"a weaker detection is kept out", true, 0.9, true, 0.6, 0.9},
		{"a clean chunk cannot dilute a detection", true, 0.9, false, 0.2, 0.9},
		{"the least confident clean chunk is reported", false, 1.0, false, 0.7, 0.7},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := mergeChunkConfidence(tc.merged, tc.mergedConfidence, tc.chunk, tc.chunkConfidence)
			if got != tc.want {
				t.Fatalf("mergeChunkConfidence = %.2f, want %.2f", got, tc.want)
			}
		})
	}
}
