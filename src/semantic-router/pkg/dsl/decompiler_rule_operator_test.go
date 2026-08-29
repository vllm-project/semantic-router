package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// unnormalizedORDecision builds a decision whose rule tree uses a
// lowercase, unnormalized "or" operator — the shape a config.RouterConfig
// can carry when it was never routed through config.NormalizeRuleOperator
// (e.g. a Kubernetes CR merged directly for a dashboard preview). The
// decompiler must not assume normalization has already happened.
func unnormalizedORDecision() *config.RouterConfig {
	return &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:     "vision_or_urgent",
				Priority: 100,
				Rules: config.RuleCombination{
					Operator: "or",
					Conditions: []config.RuleNode{
						{Type: "domain", Name: "business"},
						{Type: "keyword", Name: "urgent"},
					},
				},
				ModelRefs: []config.ModelRef{{Model: "m:1b"}},
			}},
		},
	}
}

// TestDecompileToAST_RecognizesUnnormalizedOperatorCase locks in the fix for
// issue #2937: decompileRuleNodeToExpr previously switched on the raw
// node.Operator value, so an unnormalized "or" matched no case and the
// route's WHEN clause silently became nil — turning a scoped OR route into
// one that matches every request.
func TestDecompileToAST_RecognizesUnnormalizedOperatorCase(t *testing.T) {
	cfg := unnormalizedORDecision()

	prog := DecompileToAST(cfg)
	if len(prog.Routes) != 1 {
		t.Fatalf("expected 1 route, got %d", len(prog.Routes))
	}

	route := prog.Routes[0]
	if route.When == nil {
		t.Fatal("expected a WHEN clause, got nil — the route now matches every request")
	}
	if _, ok := route.When.(*BoolOr); !ok {
		t.Fatalf("expected *BoolOr, got %T (%+v)", route.When, route.When)
	}
}

// TestDecompile_RecognizesUnnormalizedOperatorCase locks in the text
// decompiler side of the same bug: decompileRuleNode fell through to
// decompileRuleFallback for an unnormalized "or", which joined children with
// the literal lowercase operator string. That text does not parse as valid
// DSL (AND/OR/NOT are reserved uppercase keywords), so the emitted source
// failed to recompile.
func TestDecompile_RecognizesUnnormalizedOperatorCase(t *testing.T) {
	cfg := unnormalizedORDecision()

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("unexpected decompile error: %v", err)
	}
	if strings.Contains(dslText, " or ") {
		t.Fatalf("decompiled DSL contains the invalid lowercase keyword %q:\n%s", " or ", dslText)
	}
	if !strings.Contains(dslText, "OR") {
		t.Fatalf("expected decompiled DSL to use the OR keyword:\n%s", dslText)
	}

	recompiled, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("decompiled DSL failed to recompile: %v\n%s", errs, dslText)
	}
	if len(recompiled.Decisions) != 1 {
		t.Fatalf("expected 1 recompiled decision, got %d", len(recompiled.Decisions))
	}
	rules := recompiled.Decisions[0].Rules
	if rules.Operator != "OR" {
		t.Fatalf("expected recompiled operator OR, got %q", rules.Operator)
	}
	if len(rules.Conditions) != 2 {
		t.Fatalf("expected 2 conditions to survive the round trip, got %d: %+v", len(rules.Conditions), rules.Conditions)
	}
}

// TestCompileComplexitySignal_RejectsUnsupportedComposerOperator locks in the
// DSL-compiler half of issue #2937: compileComposerObj used to copy an
// invalid operator into the config verbatim, so a typo in DSL source (e.g.
// "XOR" instead of "OR") only failed much later at config load, far from the
// line that caused it. The compiler must reject it immediately.
func TestCompileComplexitySignal_RejectsUnsupportedComposerOperator(t *testing.T) {
	input := `SIGNAL complexity needs_reasoning {
  threshold: 0.7
  hard: { candidates: ["a"] }
  easy: { candidates: ["b"] }
  composer: { operator: "XOR", conditions: [{ type: "keyword", name: "kw" }] }
}`

	_, errs := Compile(input)
	if len(errs) == 0 {
		t.Fatal("expected a compile error for an unsupported composer operator")
	}
	found := false
	for _, e := range errs {
		if strings.Contains(e.Error(), "XOR") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected an error naming the unsupported operator XOR, got: %v", errs)
	}
}
