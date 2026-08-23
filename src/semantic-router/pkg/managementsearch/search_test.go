package managementsearch

import (
	"strings"
	"testing"
)

func TestNormalizeAndPrefixPattern(t *testing.T) {
	value, err := Normalize("  Production_100%  ")
	if err != nil || value != "production_100%" {
		t.Fatalf("Normalize() = %q, %v", value, err)
	}
	if pattern := PrefixPattern(value); pattern != `production\_100\%%` {
		t.Fatalf("PrefixPattern() = %q", pattern)
	}
	if pattern := PrefixPattern(""); pattern != "" {
		t.Fatalf("PrefixPattern(empty) = %q", pattern)
	}
}

func TestNormalizeRejectsUnboundedOrControlInput(t *testing.T) {
	for _, value := range []string{"bad\nquery", strings.Repeat("a", MaximumRunes+1)} {
		if _, err := Normalize(value); err == nil {
			t.Fatalf("Normalize(%q) unexpectedly succeeded", value)
		}
	}
}
