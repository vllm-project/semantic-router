//go:build !windows && cgo

package classification

import "testing"

func TestUnifiedClassifierNativeResultArrayBoundSupportsLargeBatches(t *testing.T) {
	if unifiedCResultArrayMax <= 1000 {
		t.Fatalf("expected native result array bound above legacy 1000 cap, got %d", unifiedCResultArrayMax)
	}
}
