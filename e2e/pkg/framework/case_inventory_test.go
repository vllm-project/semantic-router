package framework

import "testing"

func TestCaseInventoryRejectsZeroDuplicateAndEmptyNames(t *testing.T) {
	for _, names := range [][]string{nil, {"case", "case"}, {""}} {
		if validateCaseInventory(names) == nil {
			t.Fatalf("invalid inventory accepted: %v", names)
		}
	}
	if err := validateCaseInventory([]string{"a", "b"}); err != nil {
		t.Fatal(err)
	}
}
