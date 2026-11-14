package store

import "testing"

func TestRecordIDAllowsOnlyOpaqueIDSyntax(t *testing.T) {
	accepted := []string{
		"replay-025",
		"3f2b8c1e-9d4a-4f1b-8c2e-1a2b3c4d5e6f",
		"abc_DEF.1",
	}
	for _, id := range accepted {
		if !validRecordID(id) {
			t.Fatalf("validRecordID(%q) = false, want true", id)
		}
	}

	rejected := []string{
		"",
		"x' or timestamp > 0 or id == 'x",
		`a"b`,
		`a\b`,
		"a b",
		"a\nb",
	}
	for _, id := range rejected {
		if validRecordID(id) {
			t.Fatalf("validRecordID(%q) = true, want false", id)
		}
	}
}
