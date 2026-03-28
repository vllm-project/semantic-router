package dsl

import (
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
