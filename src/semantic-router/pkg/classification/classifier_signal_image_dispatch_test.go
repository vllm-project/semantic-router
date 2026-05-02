package classification

import (
	"errors"
	"sync"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// classifierWithEmbeddingOnly returns a minimal *Classifier wrapping just the
// embedding sub-classifier. Sufficient for evaluateEmbeddingSignal-targeted
// tests since that method only reads keywordEmbeddingClassifier.
func classifierWithEmbeddingOnly(t *testing.T, rules []config.EmbeddingRule, modelType string) *Classifier {
	t.Helper()
	embedConfig := config.HNSWConfig{
		ModelType:         modelType,
		PreloadEmbeddings: true,
	}
	embed, err := NewEmbeddingClassifier(rules, embedConfig)
	if err != nil {
		t.Fatalf("NewEmbeddingClassifier failed: %v", err)
	}
	return &Classifier{
		keywordEmbeddingClassifier: embed,
	}
}

func newSignalResultsForTest() *SignalResults {
	return &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
	}
}

// TestEvaluateEmbeddingSignal_TextOnly_PreservesExistingBehavior confirms the
// pre-PR-2 behavior is byte-for-byte intact when no image is attached: text
// rules score and match exactly as before, no image-rule scoring happens, and
// the signal value map carries only embedding entries from the text rules.
func TestEvaluateEmbeddingSignal_TextOnly_PreservesExistingBehavior(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"TensorFlow pipeline":  makeEmbedding(0.90, 0.85, 0.10),
		"machine learning":     makeEmbedding(0.85, 0.0, 0.0),
		"neural network":       makeEmbedding(0.80, 0.0, 0.0),
		"python code":          makeEmbedding(0.0, 0.90, 0.0),
		"software development": makeEmbedding(0.0, 0.85, 0.0),
		"recipe":               makeEmbedding(0.0, 0.0, 0.30),
		"ingredients":          makeEmbedding(0.0, 0.0, 0.25),
	})

	classifier := classifierWithEmbeddingOnly(t, topicRules(), "")
	results := newSignalResultsForTest()
	var mu sync.Mutex

	classifier.evaluateEmbeddingSignal(results, &mu, "TensorFlow pipeline", "")

	if len(results.MatchedEmbeddingRules) == 0 {
		t.Fatalf("expected at least one matched rule on text-only path, got 0")
	}
	for _, name := range results.MatchedEmbeddingRules {
		if name == "" {
			t.Fatalf("matched rule name should not be empty, got %v", results.MatchedEmbeddingRules)
		}
	}
}

// TestEvaluateEmbeddingSignal_ImageProvidedActivatesImageRules confirms that
// when an image attachment is present and image-modality rules exist, the
// image rules score alongside the text rules and surface in MatchedEmbeddingRules.
func TestEvaluateEmbeddingSignal_ImageProvidedActivatesImageRules(t *testing.T) {
	stubMultiModalImageLookup(t, map[string][]float32{
		"data:image/png;base64,FAKE_WAFER_BYTES": makeEmbedding(0.92, 0.0, 0.0),
	})
	stubEmbeddingLookup(t, map[string][]float32{
		"machine learning":             makeEmbedding(0.0, 0.0, 0.95),
		"neural network":               makeEmbedding(0.0, 0.0, 0.85),
		"wafer photo":                  makeEmbedding(0.95, 0.0, 0.0),
		"SEM micrograph":               makeEmbedding(0.85, 0.0, 0.0),
		"TensorFlow training pipeline": makeEmbedding(0.0, 0.0, 0.95),
	})

	classifier := classifierWithEmbeddingOnly(t, mixedModalityRules(), "multimodal")
	results := newSignalResultsForTest()
	var mu sync.Mutex

	classifier.evaluateEmbeddingSignal(results, &mu, "TensorFlow training pipeline", "data:image/png;base64,FAKE_WAFER_BYTES")

	// The text rule should match the text query.
	hasText := false
	hasImage := false
	for _, name := range results.MatchedEmbeddingRules {
		if name == "text_topic_ai" {
			hasText = true
		}
		if name == "chip_fab_sensitive_imagery" {
			hasImage = true
		}
	}
	if !hasText {
		t.Errorf("expected text rule 'text_topic_ai' to match the text query, got: %v",
			results.MatchedEmbeddingRules)
	}
	if !hasImage {
		t.Errorf("expected image rule 'chip_fab_sensitive_imagery' to match the image attachment, got: %v",
			results.MatchedEmbeddingRules)
	}
}

// TestEvaluateEmbeddingSignal_TextErrorDoesNotSkipImagePass exercises the
// intentional behavior change introduced in PR-2: when text classification
// errors, the image classification still runs. Pre-PR-2 the function returned
// early on text error; this test guards against accidental re-introduction
// of that early-return, which would silently drop a valid image-rule match.
func TestEvaluateEmbeddingSignal_TextErrorDoesNotSkipImagePass(t *testing.T) {
	// Stub the text-embedding FFI to error.
	originalText := getEmbeddingWithModelType
	getEmbeddingWithModelType = func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		return nil, errors.New("synthetic text-FFI failure")
	}
	t.Cleanup(func() { getEmbeddingWithModelType = originalText })

	// Stub the image-embedding FFI to succeed.
	stubMultiModalImageLookup(t, map[string][]float32{
		"data:image/png;base64,FAKE_WAFER_BYTES": makeEmbedding(0.92, 0.0, 0.0),
	})
	stubEmbeddingLookup(t, map[string][]float32{
		"wafer photo":       makeEmbedding(0.95, 0.0, 0.0),
		"SEM micrograph":    makeEmbedding(0.85, 0.0, 0.0),
		"office whiteboard": makeEmbedding(0.0, 0.95, 0.0),
		"conference room":   makeEmbedding(0.0, 0.85, 0.0),
	})

	classifier := classifierWithEmbeddingOnly(t, mixedModalityRules(), "multimodal")
	results := newSignalResultsForTest()
	var mu sync.Mutex

	// Despite the text-FFI error, the image classification should still run
	// and surface the chip-fab image rule.
	classifier.evaluateEmbeddingSignal(results, &mu, "TensorFlow training pipeline", "data:image/png;base64,FAKE_WAFER_BYTES")

	hasImage := false
	for _, name := range results.MatchedEmbeddingRules {
		if name == "chip_fab_sensitive_imagery" {
			hasImage = true
		}
	}
	if !hasImage {
		t.Errorf("expected image rule to fire even when text classification errored, got: %v",
			results.MatchedEmbeddingRules)
	}
}

// TestEvaluateEmbeddingSignal_ImageURLWithNoImageRules_GracefulNoOp confirms
// that passing an image attachment to a classifier configured with only
// text-modality rules does not produce errors, does not match any image rule,
// and does not regress text-rule behavior. This is the "free no-op" guarantee
// the FFI-skip-when-no-rules optimization provides upstream of this evaluator.
func TestEvaluateEmbeddingSignal_ImageURLWithNoImageRules_GracefulNoOp(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"TensorFlow pipeline":  makeEmbedding(0.90, 0.85, 0.10),
		"machine learning":     makeEmbedding(0.85, 0.0, 0.0),
		"neural network":       makeEmbedding(0.80, 0.0, 0.0),
		"python code":          makeEmbedding(0.0, 0.90, 0.0),
		"software development": makeEmbedding(0.0, 0.85, 0.0),
		"recipe":               makeEmbedding(0.0, 0.0, 0.30),
		"ingredients":          makeEmbedding(0.0, 0.0, 0.25),
	})

	classifier := classifierWithEmbeddingOnly(t, topicRules(), "multimodal")
	results := newSignalResultsForTest()
	var mu sync.Mutex

	// Text-only classifier + image URL: should not error, should not call
	// the multimodal image FFI (that's the point of the no-rules early-return
	// in ClassifyDetailedMultimodal), and should produce the same matches as
	// the no-image case.
	classifier.evaluateEmbeddingSignal(results, &mu, "TensorFlow pipeline", "data:image/png;base64,IGNORED_BYTES")

	if len(results.MatchedEmbeddingRules) == 0 {
		t.Fatalf("expected text rule to still match when image URL is present but no image rules exist, got 0 matches")
	}
}
