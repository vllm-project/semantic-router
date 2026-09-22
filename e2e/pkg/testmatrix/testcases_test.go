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
