package dsl

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const applicationSignalsSource = `
SIGNAL domain sciences { labels: ["physics", "chemistry"] predicate: { gte: 0.6 } deployment: "kai" model_scores: [{ model: "m:1b", score: 0.9, use_reasoning: true }] }
SIGNAL fact_check needs_check { labels: ["FACT_CHECK_NEEDED"] timeout_ms: 300 }
SIGNAL user_feedback unhappy { labels: ["WRONG_ANSWER", "WANT_DIFFERENT"] }
SIGNAL turn_relation asked_again { labels: ["reask"] input: { previous_turns: 6 } }
SIGNAL complexity difficulty { labels: ["hard"] description: "Hard requests" }
SIGNAL modality image_out { labels: ["image", "text_image"] predicate: { gte: 0.5 } }
SIGNAL jailbreak attack { predicate: { gte: 0.7 } stage: "request" input: { history: true } }
SIGNAL safety harmful { labels: ["Harassment"] stage: "response" }
SIGNAL pii personal { predicate: { gte: 0.8 } ignore_labels: ["EMAIL_ADDRESS"] input: { history: true, tool_results: true } }
SIGNAL pii tool_output { input: { tool_results: true, turn: false } }
SIGNAL hallucination unsupported { predicate: { gte: 0.6 } }
SIGNAL refusal refused { predicate: { gte: 0.5 } }
ROUTE guard { PRIORITY 1 WHEN jailbreak("attack") MODEL "m:1b" }
`

type applicationRuleSets struct {
	Domains       []config.Category
	FactCheck     []config.FactCheckRule
	UserFeedback  []config.UserFeedbackRule
	TurnRelation  []config.TurnRelationRule
	Complexity    []config.ComplexityRule
	Modality      []config.ModalityRule
	Jailbreak     []config.JailbreakRule
	Safety        []config.SafetyRule
	PII           []config.PIIRule
	Hallucination []config.HallucinationRule
	Refusal       []config.RefusalRule
}

func applicationRulesOf(cfg *config.RouterConfig) applicationRuleSets {
	return applicationRuleSets{
		cfg.Categories, cfg.FactCheckRules, cfg.UserFeedbackRules, cfg.TurnRelationRules, cfg.ComplexityRules,
		cfg.ModalityRules, cfg.JailbreakRules, cfg.SafetyRules, cfg.PIIRules, cfg.HallucinationRules, cfg.RefusalRules,
	}
}

func TestApplicationSignalsRoundTrip(t *testing.T) {
	cfg, errs := Compile(applicationSignalsSource)
	if len(errs) > 0 {
		t.Fatalf("compile: %v", errs)
	}
	compiled := applicationRulesOf(cfg)
	if got := cfg.PIIRules[0]; !got.Input.History || !got.Input.ToolResults || got.IgnoreLabels[0] != "EMAIL_ADDRESS" || *got.Predicate.GTE != 0.8 {
		t.Fatalf("a block's fields are its rule's YAML fields: %+v", got)
	}
	if got := cfg.Categories[0].ModelScores[0]; got.Model != "m:1b" || got.UseReasoning == nil || !*got.UseReasoning {
		t.Fatalf("model_scores = %+v", cfg.Categories[0].ModelScores)
	}

	rendered, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	again, errs := Compile(rendered)
	if len(errs) > 0 {
		t.Fatalf("round trip compile: %v\n%s", errs, rendered)
	}
	if !reflect.DeepEqual(compiled, applicationRulesOf(again)) {
		t.Fatalf("text round trip changed the rules:\n%#v\n%#v", compiled, applicationRulesOf(again))
	}

	compiler := &Compiler{config: &config.RouterConfig{}}
	for _, signal := range DecompileToAST(cfg).Signals {
		applicationSignals[signal.SignalType].compile(compiler, signal)
	}
	if len(compiler.errors) > 0 || !reflect.DeepEqual(compiled, applicationRulesOf(compiler.config)) {
		t.Fatalf("AST round trip changed the rules (%v):\n%#v\n%#v", compiler.errors, compiled, applicationRulesOf(compiler.config))
	}
}

func TestApplicationSignalRefusesAFieldItsRuleDoesNotHave(t *testing.T) {
	for _, source := range []string{
		`SIGNAL jailbreak attack { threshold: 0.7 }`,
		`SIGNAL domain math { mmlu_categories: ["math"] }`,
		`SIGNAL pii personal { pii_types_allowed: ["EMAIL_ADDRESS"] }`,
	} {
		_, errs := Compile(source + "\nROUTE guard { PRIORITY 1 WHEN keyword(\"x\") MODEL \"m:1b\" }\nSIGNAL keyword x { keywords: [\"x\"] }")
		if len(errs) == 0 || !strings.Contains(fmt.Sprint(errs), "not found") {
			t.Fatalf("%s: errors = %v", source, errs)
		}
	}
}
