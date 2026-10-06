package main

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

func TestBuildNativeTestBlockRunnerEvaluatesKeywordRoute(t *testing.T) {
	prog, errs := dsl.Parse(`
SIGNAL keyword urgent { operator: "OR" keywords: ["urgent"] }
ROUTE urgent_route { PRIORITY 100 WHEN keyword("urgent") MODEL "m:1b" }

TEST routing_intent {
  "urgent help" -> urgent_route
}
`)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}

	runner, err := buildNativeTestBlockRunner(prog)
	if err != nil {
		t.Fatalf("buildNativeTestBlockRunner error: %v", err)
	}

	result, err := runner.EvaluateTestBlockQuery("urgent help")
	if err != nil {
		t.Fatalf("EvaluateTestBlockQuery error: %v", err)
	}
	if result == nil {
		t.Fatal("expected routing result, got nil")
	}
	if result.DecisionName != "urgent_route" {
		t.Fatalf("decision = %q, want urgent_route", result.DecisionName)
	}
}
