package testmatrix

import "testing"

func TestBaselineScopesPreserveEveryFunctionalCase(t *testing.T) {
	standard, err := BaselineCases("standard")
	if err != nil {
		t.Fatal(err)
	}
	full, err := BaselineCases("full")
	if err != nil {
		t.Fatal(err)
	}
	if len(full) != 39 || len(standard) != 37 {
		t.Fatalf("inventory changed: standard=%d full=%d", len(standard), len(full))
	}
	selected := map[string]bool{}
	for _, name := range standard {
		if selected[name] {
			t.Fatal("duplicate", name)
		}
		selected[name] = true
	}
	stress := map[string]bool{}
	for _, name := range BaselineStress {
		stress[name] = true
	}
	for _, name := range full {
		if selected[name] == stress[name] {
			t.Fatalf("case %s is missing, duplicated or misclassified", name)
		}
	}
	if _, err := BaselineCases("stnadard"); err == nil {
		t.Fatal("unknown scope accepted")
	}
}

// BaselineSuites is the enumerable form of the scopes BaselineCases accepts.
// BaselineCases does not read the list, so both directions are checked:
// every listed suite must be accepted, the empty suite must resolve to
// "standard", and a suite outside the list must be rejected.
func TestBaselineSuitesEnumerateTheAcceptedScopes(t *testing.T) {
	for _, suite := range BaselineSuites {
		if _, err := BaselineCases(suite); err != nil {
			t.Fatalf("listed suite %q rejected by BaselineCases: %v", suite, err)
		}
	}
	standard, err := BaselineCases("standard")
	if err != nil {
		t.Fatal(err)
	}
	empty, err := BaselineCases("")
	if err != nil {
		t.Fatal(err)
	}
	if len(empty) != len(standard) {
		t.Fatalf("empty suite selects %d cases, standard selects %d; the empty suite must resolve to standard", len(empty), len(standard))
	}
	for i := range standard {
		if empty[i] != standard[i] {
			t.Fatalf("empty suite differs from standard at %d: %q vs %q", i, empty[i], standard[i])
		}
	}
	listed := map[string]bool{}
	for _, suite := range BaselineSuites {
		listed[suite] = true
	}
	if listed["nightly-only"] {
		t.Fatal("test precondition: the probe suite name must not be listed")
	}
	if _, err := BaselineCases("nightly-only"); err == nil {
		t.Fatal("unlisted suite accepted; add it to BaselineSuites or reject it in BaselineCases")
	}
}
