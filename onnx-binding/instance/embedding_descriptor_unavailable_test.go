//go:build windows || !cgo || (!amd64 && !arm64)

package instance

import "testing"

func TestOwnedEmbeddingDescriptorUnavailable(t *testing.T) {
	if _, err := (&EmbeddingModel{}).RuntimeDescriptor(0, 0); err != unavailable {
		t.Fatalf("unavailable build claimed identity: %v", err)
	}
}
