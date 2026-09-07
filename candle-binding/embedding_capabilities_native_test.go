//go:build !windows && cgo && (amd64 || arm64)

package candle_binding

import "testing"

func assertCapabilityDimensions(t *testing.T, got EmbeddingCapabilities) {
	t.Helper()
	switch got.DimensionState {
	case DimensionStateNotLoaded:
		if got.NativeDimension != 0 || len(got.SupportedDimensions) != 0 {
			t.Fatalf("unloaded model reports dimension data: %#v", got)
		}
	case DimensionStateAvailable:
		if err := validateObservedDimensions(got.NativeDimension, got.SupportedDimensions); err != nil {
			t.Fatal(err)
		}
	default:
		t.Fatalf("unknown dimension state: %q", got.DimensionState)
	}
}

// Called by the existing model-gated CI initialization test, after model load.
func assertLoadedMultimodalCapabilities(t *testing.T) {
	t.Helper()
	got, err := EmbeddingCapabilitiesFor("multimodal")
	if err != nil {
		t.Fatal(err)
	}
	assertCapabilityDimensions(t, got)
	if got.DimensionState != DimensionStateAvailable {
		t.Fatalf("loaded model has dimension state %q", got.DimensionState)
	}
	output, err := MultiModalEncodeText("model dimension contract", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.NativeDimension != len(output.Embedding) {
		t.Fatalf("native width %d does not match produced width %d", got.NativeDimension, len(output.Embedding))
	}
	native := got.NativeDimension
	got.SupportedDimensions[0] = -1
	again, err := EmbeddingCapabilitiesFor("multimodal")
	if err != nil {
		t.Fatal(err)
	}
	assertCapabilityDimensions(t, again)
	if again.NativeDimension != native {
		t.Fatalf("native model metadata changed after modifying Go copy: %#v", again)
	}
}
