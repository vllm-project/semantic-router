package configprojection

import (
	"strings"
	"testing"
)

func TestNewActivationVersionUsesNanosecondSuffix(t *testing.T) {
	t.Parallel()

	version := NewActivationVersion()
	parts := strings.Split(version, ".")
	if len(parts) != 2 {
		t.Fatalf("expected version with nanosecond suffix, got %q", version)
	}
	if len(parts[0]) != len("20060102-150405") {
		t.Fatalf("unexpected timestamp prefix in %q", version)
	}
	if len(parts[1]) != 9 {
		t.Fatalf("expected 9-digit nanosecond suffix, got %q", parts[1])
	}
}

func TestNewActivationVersionDiffersWithinSameSecond(t *testing.T) {
	t.Parallel()

	first := NewActivationVersion()
	second := NewActivationVersion()
	if first == second {
		t.Fatalf("expected unique activation versions, both %q", first)
	}
}
