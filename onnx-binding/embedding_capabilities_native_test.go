//go:build !windows && cgo && (amd64 || arm64)

package onnx_binding

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"
)

func TestNativeEmbeddingCapabilitiesCanonicalizeMmBert(t *testing.T) {
	got, err := EmbeddingCapabilitiesFor("  MMBERT  ")
	if err != nil {
		t.Fatalf("EmbeddingCapabilitiesFor(mmbert) error = %v", err)
	}
	if got.ModelType != ModelTypeMmBert || got.Backend != BackendONNX || got.SupportsBatching {
		t.Fatalf("EmbeddingCapabilitiesFor(mmbert) = %#v, want canonical non-batched ONNX mmbert", got)
	}
	if got.DimensionState != DimensionStateNotLoaded && got.DimensionState != DimensionStateAvailable {
		t.Fatalf("unexpected dimension state: %q", got.DimensionState)
	}
}

func TestGetEmbeddingWithModelTypeRejectsUnsupportedTypes(t *testing.T) {
	for _, modelType := range []string{"", "qwen3", "gemma", "unknown"} {
		t.Run(modelType, func(t *testing.T) {
			_, err := GetEmbeddingWithModelType("test", modelType, 0)
			if !errors.Is(err, ErrUnsupportedModelType) {
				t.Fatalf("GetEmbeddingWithModelType(%q) error = %v, want ErrUnsupportedModelType", modelType, err)
			}
		})
	}
}

func TestNativeEmbeddingCapabilitiesLoadedDimensions(t *testing.T) {
	// The legacy native model is process-global. Keep the fixture isolated
	// from tests that initialize a caller-supplied checkpoint.
	const childEnv = "ONNX_CAPABILITIES_LOADED_CHILD"
	if os.Getenv(childEnv) != "1" {
		executable, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, executable, "-test.run=^TestNativeEmbeddingCapabilitiesLoadedDimensions$", "-test.v")
		cmd.Env = append(os.Environ(), childEnv+"=1")
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("loaded capabilities subprocess: %v\n%s", err, output)
		}
		return
	}

	before, err := EmbeddingCapabilitiesFor("mmbert")
	if err != nil {
		t.Fatal(err)
	}
	assertCapabilityDimensions(t, before)
	if before.DimensionState != DimensionStateNotLoaded {
		t.Fatalf("fresh process reports a loaded mmBERT model: %#v", before)
	}
	if err = InitMmBertEmbeddingModel(filepath.Join("instance", "testdata", "embedding"), true); err != nil {
		t.Fatal(err)
	}
	assertLoadedMmBertCapabilities(t)
	got, err := EmbeddingCapabilitiesFor("mmbert")
	if err != nil {
		t.Fatal(err)
	}
	if got.NativeDimension != 3 || !slices.Equal(got.SupportedDimensions, []int{3}) {
		t.Fatalf("3-wide fixture dimensions = %#v, want native 3 and supported [3]", got)
	}
}

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

func assertLoadedMmBertCapabilities(t *testing.T) {
	t.Helper()
	got, err := EmbeddingCapabilitiesFor("mmbert")
	if err != nil {
		t.Fatal(err)
	}
	assertCapabilityDimensions(t, got)
	if got.DimensionState != DimensionStateAvailable {
		t.Fatalf("loaded model has dimension state %q", got.DimensionState)
	}
	output, err := GetEmbeddingWithModelType("model dimension contract", "mmbert", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.NativeDimension != len(output.Embedding) {
		t.Fatalf("native width %d does not match produced width %d", got.NativeDimension, len(output.Embedding))
	}
	for _, dimension := range got.SupportedDimensions {
		t.Run(strconv.Itoa(dimension), func(t *testing.T) {
			output, err := GetEmbeddingWithModelType("model dimension contract", "mmbert", dimension)
			if err != nil {
				t.Fatal(err)
			}
			if len(output.Embedding) != dimension {
				t.Fatalf("advertised width %d produced %d values", dimension, len(output.Embedding))
			}
		})
	}
}
