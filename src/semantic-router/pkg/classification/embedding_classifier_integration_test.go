package classification

// Real model integration uses an explicitly prepared, owned Omni artifact.
// Run with VELA_OMNI_ARTIFACT=/path/to/vela-1.0-omni-nano and ORT_DYLIB_PATH
// pointing to the installed ORT library. An explicitly selected missing/broken
// artifact fails; only an unselected model-backed suite is skipped.

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// generateSyntheticPNGBase64 returns a base64-encoded 32x32 PNG with a single
// solid color. The exact pixel content is unimportant; the test only needs a
// valid PNG that the selected Omni image branch can decode and embed.
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

func requireOmniProvider(t *testing.T) embedding.Provider {
	t.Helper()
	artifact := os.Getenv("VELA_OMNI_ARTIFACT")
	if artifact == "" {
		if os.Getenv("REQUIRE_OMNI_TESTS") == "1" {
			t.Fatal("VELA_OMNI_ARTIFACT must select a prepared artifact")
		}
		t.Skip("set VELA_OMNI_ARTIFACT to select owned Omni integration")
	}
	provider, err := native.New(nil).Embedding(context.Background(), config.ResolvedModelBinding{
		Recipe: "image-test", Name: "embedding",
		Binding:    config.ModelBinding{Deployment: "omni", Contract: "embedding.v1", Adapter: "vela_omni"},
		Deployment: config.ModelDeployment{Artifact: artifact, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{Overflow: "reject"}},
	}, 0, 0)
	if err != nil {
		t.Fatalf("prepare selected Omni artifact: %v", err)
	}
	t.Cleanup(func() {
		if err := provider.Close(); err != nil {
			t.Error(err)
		}
	})
	return provider
}

func newOwnedImageClassifier(t *testing.T, rules []config.EmbeddingRule) *EmbeddingClassifier {
	t.Helper()
	provider := requireOmniProvider(t)
	options := multimodalHNSWConfig(true)
	options.TargetDimension = provider.Dimension()
	classifier, err := NewEmbeddingClassifierWithProvider(rules, options, provider)
	if err != nil {
		t.Fatal(err)
	}
	if err := classifier.WarmupCandidateEmbeddings(); err != nil {
		t.Fatalf("warm owned image anchors: %v", err)
	}
	return classifier
}

// TestEmbeddingClassifier_IntegrationImageQueryEndToEnd proves the full image
// classification path works against the real model: it constructs a classifier
// with image-modality rules, calls ClassifyDetailedMultimodal with a real
// base64-encoded PNG payload, and asserts that the FFI returned an embedding
// and the result populates Scores for the image rules (not the absent text
// rules). This is the integration coverage missing from the unit-test suite,
// which uses owned typed fakes instead of a real native artifact.
func TestEmbeddingClassifier_IntegrationImageQueryEndToEnd(t *testing.T) {
	imagePayload := generateSyntheticPNGBase64(t, color.RGBA{R: 200, G: 50, B: 50, A: 255})

	// Use the same fixture the unit tests use so the integration test exercises
	// real preload semantics for the same anchor pack shape that ships with the
	// upstream PR's reference example tutorial.
	classifier := newOwnedImageClassifier(t, chipFabImageRules())

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
	imagePayload := generateSyntheticPNGBase64(t, color.RGBA{R: 100, G: 100, B: 200, A: 255})

	classifier := newOwnedImageClassifier(t, mixedModalityRules())

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
