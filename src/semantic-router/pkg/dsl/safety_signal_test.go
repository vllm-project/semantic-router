package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSafetySignalRoundTrip(t *testing.T) {
	source := `
SIGNAL safety content { model: "binary" threshold: 0.55 labels: ["benign", "harmful"] unsafe_labels: ["harmful"] description: "Content hazards" hazard: { model: "hazards" labels: ["privacy", "violence", "other"] categories: ["privacy", "violence"] threshold: 0.6 } }
ROUTE guard { PRIORITY 1 WHEN safety("content") MODEL "m:1b" }
`
	cfg, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatalf("compile: %v", errs)
	}
	rendered, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rendered, "SIGNAL safety") {
		t.Fatalf("safety omitted: %s", rendered)
	}
	again, errs := Compile(rendered)
	if len(errs) > 0 {
		t.Fatalf("round trip compile: %v", errs)
	}
	if !reflect.DeepEqual(cfg.SafetyRules, again.SafetyRules) {
		t.Fatalf("safety changed: %#v -> %#v", cfg.SafetyRules, again.SafetyRules)
	}
	ast := DecompileToAST(cfg)
	if len(ast.Signals) != 1 || ast.Signals[0].SignalType != "safety" {
		t.Fatalf("AST dropped safety: %+v", ast.Signals)
	}
	compiler := &Compiler{config: &config.RouterConfig{}}
	compiler.compileSafetySignal(ast.Signals[0])
	if !reflect.DeepEqual(cfg.SafetyRules, compiler.config.SafetyRules) {
		t.Fatal("AST dropped safety fields")
	}
}
