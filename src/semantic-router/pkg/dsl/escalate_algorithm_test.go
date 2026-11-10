package dsl

import (
	"reflect"
	"strings"
	"testing"
)

const systemOneEscalateDSL = `
ROUTE native {
 MODEL "kai", "vega"
 ALGORITHM escalate {
  budget: { deadline: "3s", max_calls: 2 }
  quality: { type: "uncalibrated", accept: { native: [{question_type: "noul", field: "top_probability", gte: 0}] } }
  stages: [
   {name: "fast", model: "kai", accept: {native: [{question_type: "noul", field: "top_probability", gte: 0.6}]}},
   {name: "upgrade", model: "vega", timeout: "2s"}
  ]
 }
}`

const chatEscalateDSL = `
ROUTE chat {
 MODEL "small", "large"
 ALGORITHM escalate {
  on_error: "fail"
  budget: { deadline: "20s", max_calls: 2 }
  order: "cost"
  accept: { confidence: { method: "hybrid", gte: 0.72, hybrid_weights: { logprob_weight: 0.65, margin_weight: 0.35 }, token_filter: "tool_call_args" } }
 }
}`

func TestEscalateDSLRoundTripsBothPaths(t *testing.T) {
	for name, input := range map[string]string{"system one": systemOneEscalateDSL, "chat": chatEscalateDSL} {
		t.Run(name, func(t *testing.T) {
			cfg := mustCompilePolicyDSL(t, input)
			first := cfg.Decisions[0].Algorithm
			if first.Budget == nil || first.Budget.MaxCalls != 2 || first.Escalate == nil {
				t.Fatalf("escalate contract lost: %+v", first)
			}
			text, err := Decompile(cfg)
			if err != nil {
				t.Fatal(err)
			}
			roundTrip := mustCompilePolicyDSL(t, text)
			if !reflect.DeepEqual(first, roundTrip.Decisions[0].Algorithm) {
				t.Fatalf("escalate fields changed during round trip:\n%s\n%#v", text, roundTrip.Decisions[0].Algorithm)
			}
		})
	}
	native := mustCompilePolicyDSL(t, systemOneEscalateDSL).Decisions[0].Algorithm
	if stages := native.NativeStages(); len(stages) != 2 || stages[0].Accept == nil || *stages[0].Accept.Native[0].GTE != 0.6 {
		t.Fatalf("stage accept lost: %+v", stages)
	}
	chat := mustCompilePolicyDSL(t, chatEscalateDSL).Decisions[0].Algorithm
	if chat.OnError != "fail" || chat.Escalate.Order != "cost" || chat.Escalate.Accept.Confidence.Method != "hybrid" ||
		chat.Escalate.Accept.Confidence.HybridWeights.MarginWeight != 0.35 {
		t.Fatalf("Chat escalate settings lost: %+v", chat)
	}
}

func TestEscalateDSLRejectsRemovedOrMisplacedFields(t *testing.T) {
	for _, input := range []string{
		strings.Replace(systemOneEscalateDSL, "ALGORITHM escalate", "ALGORITHM cascade", 1),
		strings.Replace(chatEscalateDSL, "ALGORITHM escalate", "ALGORITHM confidence", 1),
		`ROUTE chat { MODEL "chat" ALGORITHM static { budget: { deadline: "3s", max_calls: 2 } } }`,
		strings.Replace(systemOneEscalateDSL, `budget: { deadline: "3s", max_calls: 2 }`, `policy: {source: "policy.json"}`, 1),
		strings.Replace(systemOneEscalateDSL, "max_calls: 2", "max_calls: 0", 1),
		strings.Replace(chatEscalateDSL, `order: "cost"`, `escalation_order: "cost"`, 1),
		strings.Replace(chatEscalateDSL, `gte: 0.72`, `threshold: 0.72`, 1),
	} {
		_, compileErrors := Compile(input)
		if len(compileErrors) == 0 {
			t.Fatalf("invalid escalate DSL accepted: %s", input)
		}
	}
}
