package testmatrix

import "testing"

func TestBaselineInventoryHasNoDuplicates(t *testing.T) {
	full := BaselineCases()
	if len(full) != 35 {
		t.Fatalf("inventory changed: %d cases", len(full))
	}
	seen := map[string]bool{}
	for _, name := range full {
		if seen[name] {
			t.Fatal("duplicate", name)
		}
		seen[name] = true
	}
	if !seen["router-config-version-gate"] {
		t.Fatal("router-config-version-gate must run in the CI baseline")
	}
}
