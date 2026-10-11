package dsl

import (
	"reflect"
	"strings"
	"testing"
)

const nativeCascadeDSL = `
ROUTE native {
 MODEL "kai", "vega"
 ALGORITHM cascade {
  budget: { deadline: "3s", max_calls: 2 }
  quality: { type: "uncalibrated", acceptance: { rules: [{question_type: "noul", field: "top_probability", predicate: {gte: 0}}] } }
  stages: [
   {name: "fast", kind: "native", model: "kai", accept: {rules: [{question_type: "noul", field: "top_probability", predicate: {gte: 0.6}}]}},
   {name: "upgrade", kind: "native", model: "vega", timeout: "2s"}
  ]
 }
}`

func TestNativeCascadeDSLRetainsAlgorithmBudgetAndStages(t *testing.T) {
	cfg := mustCompilePolicyDSL(t, nativeCascadeDSL)
	first := cfg.Decisions[0].Algorithm
	if first.Budget == nil || first.Budget.MaxCalls != 2 || len(first.Stages) != 2 {
		t.Fatalf("native contract lost: %+v", first)
	}
	text, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip := mustCompilePolicyDSL(t, text)
	if !reflect.DeepEqual(first, roundTrip.Decisions[0].Algorithm) {
		t.Fatalf("native fields changed during round trip:\n%s\n%#v", text, roundTrip.Decisions[0].Algorithm)
	}
}

func TestNativeCascadeDSLRejectsRemovedOrMisplacedFields(t *testing.T) {
	for _, input := range []string{
		strings.Replace(nativeCascadeDSL, "ALGORITHM cascade", "ALGORITHM policy", 1),
		`ROUTE chat { MODEL "chat" ALGORITHM static { budget: { deadline: "3s", max_calls: 2 } } }`,
		strings.Replace(nativeCascadeDSL, `budget: { deadline: "3s", max_calls: 2 }`, `policy: {source: "policy.json"}`, 1),
		strings.Replace(nativeCascadeDSL, "max_calls: 2", "max_calls: 0", 1),
	} {
		_, compileErrors := Compile(input)
		if len(compileErrors) == 0 {
			t.Fatalf("invalid native DSL accepted: %s", input)
		}
	}
}
