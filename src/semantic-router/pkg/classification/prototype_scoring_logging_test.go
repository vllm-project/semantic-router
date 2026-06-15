package classification

import "testing"

func TestPrototypeBankRepresentativesReturnsCopy(t *testing.T) {
	bank := &prototypeBank{
		prototypes: []prototypeRepresentative{
			{Text: "alpha", ClusterSize: 3},
		},
	}

	representatives := bank.representatives()
	if len(representatives) != 1 {
		t.Fatalf("expected one representative, got %+v", representatives)
	}

	representatives[0].Text = "mutated"
	if bank.prototypes[0].Text != "alpha" {
		t.Fatalf("expected bank prototypes to remain unchanged, got %+v", bank.prototypes)
	}
}

func TestTruncatePrototypePreview(t *testing.T) {
	short := "short preview"
	if got := truncatePrototypePreview(short); got != short {
		t.Fatalf("expected short preview unchanged, got %q", got)
	}

	long := "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890"
	got := truncatePrototypePreview(long)
	if len(got) != prototypeMedoidPreviewLimit {
		t.Fatalf("expected truncated preview length %d, got %d (%q)", prototypeMedoidPreviewLimit, len(got), got)
	}
	if got[len(got)-3:] != "..." {
		t.Fatalf("expected ellipsis suffix, got %q", got)
	}
}
