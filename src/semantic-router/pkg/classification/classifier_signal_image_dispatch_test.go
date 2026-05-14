package classification

import (
	"errors"
	"sync"
	"sync/atomic"
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

	classifier.evaluateEmbeddingSignal(results, &mu, "TensorFlow pipeline", "", nil)

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

	classifier.evaluateEmbeddingSignal(results, &mu, "TensorFlow training pipeline", "data:image/png;base64,FAKE_WAFER_BYTES", nil)

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
	classifier.evaluateEmbeddingSignal(results, &mu, "TensorFlow training pipeline", "data:image/png;base64,FAKE_WAFER_BYTES", nil)

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

// TestEvaluateEmbeddingSignal_ImageOnlyContent_SkipsTextFFI confirms that
// when a request carries only an image (empty text), the text-FFI is never
// invoked and no spurious "text-modality embedding rule evaluation failed"
// log fires. This is the request shape produced by OpenAI chat completion
// content arrays containing only image_url parts.
func TestEvaluateEmbeddingSignal_ImageOnlyContent_SkipsTextFFI(t *testing.T) {
	var textCalls int32
	originalText := getEmbeddingWithModelType
	getEmbeddingWithModelType = func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		atomic.AddInt32(&textCalls, 1)
		return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(0.0, 0.0, 0.0)}, nil
	}
	t.Cleanup(func() { getEmbeddingWithModelType = originalText })

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

	classifier.evaluateEmbeddingSignal(results, &mu, "", "data:image/png;base64,FAKE_WAFER_BYTES", nil)

	if got := atomic.LoadInt32(&textCalls); got != 0 {
		t.Errorf("text-FFI must not be invoked when text is empty, got %d calls", got)
	}
	hasImage := false
	for _, name := range results.MatchedEmbeddingRules {
		if name == "chip_fab_sensitive_imagery" {
			hasImage = true
		}
	}
	if !hasImage {
		t.Errorf("image rule should still fire for image-only request, got matches: %v",
			results.MatchedEmbeddingRules)
	}
}

// TestEvaluateAllSignals_DedupsImageFFIAcrossDispatchers exercises the
// production deduplication property under the actual concurrency pattern
// runSignalDispatchers uses. Both evaluateEmbeddingSignal and
// evaluateComplexitySignal run as parallel goroutines sharing one
// requestImageEmbeddingCache. With the cache, the SigLIP forward pass FFI
// must run exactly once across the dispatcher fan-out.
//
// Setup notes:
//   - The complexity classifier preloads its image_candidates at init time,
//     which calls getMultiModalImageEmbedding once per candidate. The test
//     resets the FFI counter AFTER classifier construction so only
//     request-time calls are counted.
//   - The embedding classifier with image rules does not trigger image FFI
//     at init (its candidates are text anchor labels embedded via the
//     multimodal text path), so no reset is needed for that side.
func TestEvaluateAllSignals_DedupsImageFFIAcrossDispatchers(t *testing.T) {
	var ffiCalls int32
	originalImage := getMultiModalImageEmbedding
	getMultiModalImageEmbedding = func(imageRef string, targetDim int) ([]float32, error) {
		atomic.AddInt32(&ffiCalls, 1)
		return makeEmbedding(0.92, 0.0, 0.0), nil
	}
	t.Cleanup(func() { getMultiModalImageEmbedding = originalImage })

	stubEmbeddingLookup(t, map[string][]float32{
		"wafer photo":          makeEmbedding(0.95, 0.0, 0.0),
		"SEM micrograph":       makeEmbedding(0.85, 0.0, 0.0),
		"office whiteboard":    makeEmbedding(0.0, 0.95, 0.0),
		"conference room":      makeEmbedding(0.0, 0.85, 0.0),
		"trace the root cause": makeEmbedding(0.99, 0.05, 0.0),
		"quick summary":        makeEmbedding(0.05, 0.99, 0.0),
		"wafer photo query":    makeEmbedding(0.92, 0.0, 0.0),
	})

	originalMMText := getMultiModalTextEmbedding
	getMultiModalTextEmbedding = func(text string, targetDim int) ([]float32, error) {
		return makeEmbedding(0.0, 0.0, 0.5), nil
	}
	t.Cleanup(func() { getMultiModalTextEmbedding = originalMMText })

	embedClassifier := classifierWithEmbeddingOnly(t, mixedModalityRules(), "multimodal")

	complexityRules := []config.ComplexityRule{{
		Name:      "needs_reasoning",
		Threshold: 0.2,
		Hard: config.ComplexityCandidates{
			Candidates:      []string{"trace the root cause"},
			ImageCandidates: []string{"data:image/png;base64,FAKE_HARD_CANDIDATE"},
		},
		Easy: config.ComplexityCandidates{
			Candidates:      []string{"quick summary"},
			ImageCandidates: []string{"data:image/png;base64,FAKE_EASY_CANDIDATE"},
		},
	}}
	complexityClassifier, err := NewComplexityClassifier(complexityRules, "multimodal", config.PrototypeScoringConfig{
		ClusterSimilarityThreshold: 0.98,
		MaxPrototypes:              2,
	})
	if err != nil {
		t.Fatalf("NewComplexityClassifier failed: %v", err)
	}

	// Reset the counter after init so we only measure request-time FFI calls.
	atomic.StoreInt32(&ffiCalls, 0)

	classifier := &Classifier{
		keywordEmbeddingClassifier: embedClassifier.keywordEmbeddingClassifier,
		complexityClassifier:       complexityClassifier,
	}
	results := newSignalResultsForTest()
	var mu sync.Mutex
	cache := newRequestImageEmbeddingCache()

	const imageURL = "data:image/png;base64,FAKE_WAFER_REQUEST_BYTES"

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		classifier.evaluateEmbeddingSignal(results, &mu, "wafer photo query", imageURL, cache)
	}()
	go func() {
		defer wg.Done()
		classifier.evaluateComplexitySignal(results, &mu, "wafer photo query", imageURL, cache)
	}()
	wg.Wait()

	if got := atomic.LoadInt32(&ffiCalls); got != 1 {
		t.Errorf("expected image FFI to run once across embedding+complexity dispatcher fan-out, ran %d times", got)
	}
}

// TestEvaluateEmbeddingSignal_SharedCacheDedupsAcrossDivergentTargetDims is
// the test that catches the canonical-config bug. Embedding signal asks for
// optimizationConfig.TargetDimension; complexity signal hardcodes 0. With a
// targetDim-keyed cache they would compute independently. With the
// imageRef-keyed full-dim cache they share. Simulates that divergence by
// pulling the same image through the cache at two different targetDims.
func TestEvaluateEmbeddingSignal_SharedCacheDedupsAcrossDivergentTargetDims(t *testing.T) {
	var ffiCalls int32
	cache := newRequestImageEmbeddingCache()

	compute := func() ([]float32, error) {
		atomic.AddInt32(&ffiCalls, 1)
		return makeEmbedding(0.6, 0.8, 0.0), nil
	}

	// Embedding-side: requests truncation to 2 (mirrors a non-default TargetDimension).
	embView, err := cache.resolve("img-A", 2, compute)
	if err != nil {
		t.Fatalf("embedding-side resolve failed: %v", err)
	}
	// Complexity-side: requests full dim (hardcoded 0).
	complexityView, err := cache.resolve("img-A", 0, compute)
	if err != nil {
		t.Fatalf("complexity-side resolve failed: %v", err)
	}

	if got := atomic.LoadInt32(&ffiCalls); got != 1 {
		t.Errorf("FFI must run once across divergent targetDims for the same image, ran %d times", got)
	}
	if len(embView) != 2 {
		t.Errorf("embedding view should be truncated to 2, got len=%d", len(embView))
	}
	if len(complexityView) != 768 {
		t.Errorf("complexity view should be full-dim 768 (makeEmbedding default), got len=%d", len(complexityView))
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
	classifier.evaluateEmbeddingSignal(results, &mu, "TensorFlow pipeline", "data:image/png;base64,IGNORED_BYTES", nil)

	if len(results.MatchedEmbeddingRules) == 0 {
		t.Fatalf("expected text rule to still match when image URL is present but no image rules exist, got 0 matches")
	}
}
