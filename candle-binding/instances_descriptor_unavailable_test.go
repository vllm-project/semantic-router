//go:build windows || !cgo || (!amd64 && !arm64 && !riscv64)

package candle_binding

import (
	"errors"
	"testing"
)

func TestOwnedEmbeddingDescriptorUnavailable(t *testing.T) {
	var nilModel *EmbeddingModel
	if _, err := nilModel.RuntimeDescriptor(0, 0); !errors.Is(err, ErrInstanceClosed) {
		t.Fatalf("nil model: %v", err)
	}
	model := &EmbeddingModel{instance: &instance{handle: 1}}
	if _, err := model.RuntimeDescriptor(0, 0); !errors.Is(err, ErrBackendUnavailable) {
		t.Fatalf("unavailable build claimed identity: %v", err)
	}
}
