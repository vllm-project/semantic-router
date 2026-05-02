package classification

// Integration test for the multimodal embedding classifier path.
//
// This test exercises the real candle-binding FFI (no stubbed embedding
// functions) to prove the end-to-end image-query path works against the
// production model: base64 image -> getMultiModalImageEmbedding -> 384-dim
// embedding -> cosine match against preloaded text anchors. It mirrors the
// env-var-skip pattern used by the multimodal tests in
// candle-binding/semantic-router_test.go.
//
// Run with:
//   MULTIMODAL_MODEL_PATH=/path/to/multi-modal-embed-small \
//     go test ./pkg/classification/ -run TestEmbeddingClassifier_Integration -v

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// generateSyntheticPNGBase64 returns a base64-encoded 32x32 PNG with a single
// solid color. The exact pixel content is unimportant; the test only needs a
// valid PNG that the multi-modal-embed-small image branch can decode and embed.
// Using a procedurally generated image keeps the fixture in-source so the test
// has no external dependencies.
func generateSyntheticPNGBase64(t *testing.T, c color.RGBA) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("failed to encode synthetic PNG: %v", err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// requireMultiModalModel skips the test when the model path is not available,
// matching the convention in candle-binding/semantic-router_test.go for
// MultiModal tests.
func requireMultiModalModel(t *testing.T) {
	t.Helper()
	modelPath := os.Getenv("MULTIMODAL_MODEL_PATH")
	if modelPath == "" {
		t.Skip("Skipping integration test: MULTIMODAL_MODEL_PATH not set; export it to point at the multi-modal-embed-small model directory to run this test")
	}
	// Init is idempotent at the candle-binding layer; ignoring already-initialized
	// is consistent with how candle-binding/semantic-router_test.go handles it.
	_ = candle_binding.InitMultiModalEmbeddingModel(modelPath, true)
}

// TestEmbeddingClassifier_IntegrationImageQueryEndToEnd proves the full image
// classification path works against the real model: it constructs a classifier
// with image-modality rules, calls ClassifyDetailedMultimodal with a real
// base64-encoded PNG payload, and asserts that the FFI returned an embedding
// and the result populates Scores for the image rules (not the absent text
// rules). This is the integration coverage missing from the unit-test suite,
// which stubs getMultiModalImageEmbedding and never exercises the FFI path.
func TestEmbeddingClassifier_IntegrationImageQueryEndToEnd(t *testing.T) {
	requireMultiModalModel(t)

	imagePayload := generateSyntheticPNGBase64(t, color.RGBA{R: 200, G: 50, B: 50, A: 255})

	// Use the same fixture the unit tests use so the integration test exercises
	// real preload semantics for the same anchor pack shape that ships with the
	// upstream PR's reference example tutorial.
	classifier, err := NewEmbeddingClassifier(chipFabImageRules(), multimodalHNSWConfig(true))
	if err != nil {
		t.Fatalf("NewEmbeddingClassifier failed: %v", err)
	}

	result, err := classifier.ClassifyDetailedMultimodal(config.QueryModalityImage, imagePayload)
	if err != nil {
		t.Fatalf("ClassifyDetailedMultimodal failed against real FFI: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result, got nil")
	}
	// Both image-modality rules should appear in Scores even if neither passes
	// the threshold; the score distribution is what callers reason about.
	if len(result.Scores) != 2 {
		t.Errorf("expected 2 scored image rules from chipFabImageRules(), got %d: %+v",
			len(result.Scores), result.Scores)
	}
	for _, score := range result.Scores {
		if score.Score < -1.001 || score.Score > 1.001 {
			t.Errorf("rule %q score %.4f outside valid cosine range [-1, 1]", score.Name, score.Score)
		}
	}
}

// TestEmbeddingClassifier_IntegrationTextRulesIgnoredOnImagePath confirms the
// modality-filter cache works against the real model: a classifier with
// mixed text + image rules should only score the image rules when called via
// ClassifyDetailedMultimodal.
func TestEmbeddingClassifier_IntegrationTextRulesIgnoredOnImagePath(t *testing.T) {
	requireMultiModalModel(t)

	imagePayload := generateSyntheticPNGBase64(t, color.RGBA{R: 100, G: 100, B: 200, A: 255})

	classifier, err := NewEmbeddingClassifier(mixedModalityRules(), multimodalHNSWConfig(true))
	if err != nil {
		t.Fatalf("NewEmbeddingClassifier failed: %v", err)
	}

	result, err := classifier.ClassifyDetailedMultimodal(config.QueryModalityImage, imagePayload)
	if err != nil {
		t.Fatalf("ClassifyDetailedMultimodal failed against real FFI: %v", err)
	}
	for _, score := range result.Scores {
		if score.Name == "text_topic_ai" {
			t.Errorf("image classification path should NOT score text-modality rule %q via real FFI, got %+v",
				score.Name, score)
		}
	}
}
