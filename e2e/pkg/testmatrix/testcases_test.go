package testmatrix

import "testing"

func TestBaselineInventoryHasNoDuplicates(t *testing.T) {
	full := BaselineCases()
	if len(full) != 34 {
		t.Fatalf("inventory changed: %d cases", len(full))
	}
	seen := map[string]bool{}
	for _, name := range full {
		if seen[name] {
			t.Fatal("duplicate", name)
		}
		seen[name] = true
	}
}
