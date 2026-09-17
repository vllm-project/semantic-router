package dsl

import (
	"fmt"
	"strings"
	"testing"
)

func TestValidateProjectionPartitionUnsupportedMemberType(t *testing.T) {
	input := `
SIGNAL keyword urgent { operator: "any" patterns: ["urgent"] }

PROJECTION partition test_group {
  semantics: "exclusive"
  members: ["urgent"]
  default: "urgent"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if strings.Contains(d.Message, "supported runtime signal type") &&
			strings.Contains(d.Message, "keyword") {
			found = true
		}
	}
	if !found {
		t.Error("expected constraint about unsupported projection partition member type")
	}
}

func TestValidateProjectionPartitionMixedMemberTypes(t *testing.T) {
	input := `
SIGNAL domain math { mmlu_categories: ["math"] }
SIGNAL embedding ai {
  examples: ["transformer", "neural network"]
}

PROJECTION partition test_group {
  semantics: "softmax_exclusive"
  temperature: 0.1
  members: ["math", "ai"]
  default: "math"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if strings.Contains(d.Message, "share one supported runtime signal type") &&
			strings.Contains(d.Message, "domain") &&
			strings.Contains(d.Message, "embedding") {
			found = true
		}
	}
	if !found {
		t.Error("expected constraint about mixed projection partition member types")
	}
}

func TestValidateProjectionPartitionImpossibleAndCondition(t *testing.T) {
	input := `
SIGNAL domain math { mmlu_categories: ["math"] }
SIGNAL domain science { mmlu_categories: ["physics"] }

PROJECTION partition domain_taxonomy {
  semantics: "exclusive"
  members: ["math", "science"]
  default: "science"
}

ROUTE impossible_route {
  PRIORITY 100
  WHEN domain("math") AND domain("science")
  MODEL "m1"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if strings.Contains(d.Message, "mutually exclusive") &&
			strings.Contains(d.Message, "domain_taxonomy") &&
			strings.Contains(d.Message, "math") &&
			strings.Contains(d.Message, "science") {
			found = true
		}
	}
	if !found {
		t.Error("expected constraint about impossible AND between partition members")
	}
}

func TestProjectionPartitionChecksWholeBooleanCondition(t *testing.T) {
	for _, test := range []struct {
		name, condition string
		impossible      bool
	}{
		{"repeated alternatives", `(domain("math") OR domain("science")) AND (domain("math") OR domain("science"))`, false},
		{"viable fallback", `(domain("math") AND domain("science")) OR domain("history")`, false},
		{"all alternatives conflict", `(domain("math") OR domain("science")) AND domain("history")`, true},
		{"nested conditional alternatives", `(domain("math") OR domain("science")) AND (keyword("explain") OR (keyword("details") AND (domain("math") OR domain("science"))))`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			input := fmt.Sprintf(`
SIGNAL domain math { mmlu_categories: ["math"] }
SIGNAL domain science { mmlu_categories: ["physics"] }
SIGNAL domain history { mmlu_categories: ["history"] }
SIGNAL keyword explain { operator: "any" patterns: ["explain"] }
SIGNAL keyword details { operator: "any" patterns: ["details"] }
PROJECTION partition topics { semantics: "exclusive" members: ["math", "science", "history"] default: "history" }
ROUTE condition { PRIORITY 100 WHEN %s MODEL "m1" }
`, test.condition)
			diagnostics, errs := Validate(input)
			if len(errs) > 0 {
				t.Fatal(errs)
			}
			conflict := false
			for _, diag := range diagnostics {
				if strings.Contains(diag.Message, "mutually exclusive") {
					conflict = true
				}
			}
			if conflict != test.impossible {
				t.Fatalf("constraint=%v want=%v: %+v", conflict, test.impossible, diagnostics)
			}
		})
	}
}
