//go:build !windows && cgo

package apiserver

import (
	"strings"
	"testing"
)

// target_layer must be validated against the layers the loaded model actually
// advertises, not a hardcoded list. For the official
// mmbert-embed-32k-2d-matryoshka (available_layers [6, 11, 16, 22]), layer 16
// must be accepted (it ships and is loadable) and layer 3 must be rejected
// (it is not on disk and previously fell back silently to the full model).
func TestValidateEmbeddingRequestTargetLayerFollowsModelManifest(t *testing.T) {
	available := []int{6, 11, 16, 22}

	if _, _, ok := validateEmbeddingRequest(EmbeddingRequest{
		Model:       "mmbert",
		Texts:       []string{"hello"},
		Dimension:   defaultEmbeddingDimension,
		TargetLayer: 16,
	}, available); !ok {
		t.Fatalf("expected target_layer=16 to be valid for %v", available)
	}

	code, message, ok := validateEmbeddingRequest(EmbeddingRequest{
		Model:       "mmbert",
		Texts:       []string{"hello"},
		Dimension:   defaultEmbeddingDimension,
		TargetLayer: 3,
	}, available)
	if ok {
		t.Fatalf("expected target_layer=3 to be rejected for %v", available)
	}
	if code != "INVALID_LAYER" {
		t.Fatalf("expected INVALID_LAYER, got %q", code)
	}
	if !strings.Contains(message, "6, 11, 16, 22") {
		t.Fatalf("error message should list the model's real layers, got %q", message)
	}
}

// When a model ships without a manifest the validator falls back to the legacy
// layer set, so target_layer=3 stays valid for that set.
func TestValidateEmbeddingRequestTargetLayerLegacyFallback(t *testing.T) {
	if _, _, ok := validateEmbeddingRequest(EmbeddingRequest{
		Model:       "mmbert",
		Texts:       []string{"hello"},
		Dimension:   defaultEmbeddingDimension,
		TargetLayer: 3,
	}, []int{3, 6, 11, 22}); !ok {
		t.Fatalf("expected target_layer=3 to be valid for the legacy fallback set")
	}
}

// target_layer is only meaningful for mmbert; other models must reject it.
func TestValidateEmbeddingRequestTargetLayerRejectedForNonMmbert(t *testing.T) {
	code, _, ok := validateEmbeddingRequest(EmbeddingRequest{
		Model:       "qwen3",
		Texts:       []string{"hello"},
		Dimension:   defaultEmbeddingDimension,
		TargetLayer: 6,
	}, []int{6, 11, 16, 22})
	if ok {
		t.Fatalf("expected target_layer on non-mmbert model to be rejected")
	}
	wantCode := "INVALID_PARAMETER"
	if _, supported := nativeEmbeddingBackendCapabilities().models["qwen3"]; !supported {
		wantCode = "INVALID_MODEL"
	}
	if code != wantCode {
		t.Fatalf("expected %s, got %q", wantCode, code)
	}
}

func TestApplyEmbeddingDefaultsUsesCompatibleDimensionByModality(t *testing.T) {
	tests := []struct {
		name string
		req  EmbeddingRequest
		want int
	}{
		{name: "text only", req: EmbeddingRequest{Texts: []string{"hello"}}, want: defaultEmbeddingDimension},
		{name: "image only", req: EmbeddingRequest{Images: []string{"placeholder"}}, want: defaultImageEmbeddingDimension},
		{name: "mixed", req: EmbeddingRequest{Texts: []string{"hello"}, Images: []string{"placeholder"}}, want: defaultMixedEmbeddingDimension},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			applyEmbeddingDefaults(&test.req)
			if test.req.Dimension != test.want {
				t.Fatalf("default dimension = %d, want %d", test.req.Dimension, test.want)
			}
		})
	}
}

func TestValidateEmbeddingRequestAcceptsDefaultMixedDimension(t *testing.T) {
	req := EmbeddingRequest{
		Texts:  []string{"hello"},
		Images: []string{mustEmbeddingImageDataURI(t, "image/png")},
	}
	applyEmbeddingDefaults(&req)

	if code, message, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("mixed defaults rejected: code=%q message=%q", code, message)
	}
}
