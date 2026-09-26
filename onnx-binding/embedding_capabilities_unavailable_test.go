//go:build windows || !cgo || (!amd64 && !arm64)

package onnx_binding

import (
	"errors"
	"testing"
)

func TestEmbeddingCapabilitiesUnavailableWithoutNativeRuntime(t *testing.T) {
	_, err := EmbeddingCapabilitiesFor("mmbert")
	if !errors.Is(err, ErrBackendUnavailable) {
		t.Fatalf("EmbeddingCapabilitiesFor() error = %v, want ErrBackendUnavailable", err)
	}
}
