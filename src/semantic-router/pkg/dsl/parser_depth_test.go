package dsl

import (
	"strings"
	"testing"
)

// Regression: 200k nested parentheses used to end in an unrecoverable
// "fatal error: stack overflow", killing the process that served the request.
func TestParseRejectsExcessiveNesting(t *testing.T) {
	src := "ROUTE r { WHEN " + strings.Repeat("(", 200000) + `keyword("urgent")` + strings.Repeat(")", 200000) + ` MODEL "m:1b" }`
	prog, errs := Parse(src)
	if prog != nil {
		t.Fatal("expected no program for over-deep input")
	}
	if len(errs) == 0 || !strings.Contains(errs[0].Error(), "nest deeper") {
		t.Fatalf("expected a nesting error, got %v", errs)
	}
}

func TestParseKeepsReasonableNesting(t *testing.T) {
	body := strings.Repeat("(", 40) + `keyword("urgent")` + strings.Repeat(")", 40)
	src := "ROUTE r { WHEN " + body + ` MODEL "m:1b" }`
	if _, errs := Parse(src); len(errs) > 0 {
		t.Fatalf("40 levels should parse: %v", errs)
	}
}

// Parentheses inside strings and comments are text, not nesting.
func TestNestingScannerSkipsStringsAndComments(t *testing.T) {
	input := `ROUTE r { WHEN keyword("` + strings.Repeat("(", 1000) + `") # ` + strings.Repeat("(", 1000) + "\n}"
	if exceedsNestingDepth(input, maxNestingDepth) {
		t.Fatal("string and comment contents must not count as nesting")
	}
}
