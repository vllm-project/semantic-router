package dsl

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionRuleLimitsStandaloneValidation(t *testing.T) {
	source := `SIGNAL keyword urgent {operator: "OR" keywords: ["urgent"]}
ROUTE bounded {PRIORITY 1 WHEN ` + strings.Repeat("NOT ", 16) + `keyword("urgent") MODEL "test"}`
	for name, validate := range map[string]func() []Diagnostic{
		"source":  func() []Diagnostic { diags, _ := Validate(source); return diags },
		"symbols": func() []Diagnostic { diags, _, _ := ValidateWithSymbols(source); return diags },
		"ast":     func() []Diagnostic { prog, _ := Parse(source); return ValidateAST(prog) },
	} {
		t.Run(name, func(t *testing.T) {
			diags := validate()
			if !hasBlockingDiagnostics(diags) || !strings.Contains(diags[0].Message, "depth 17 exceeds max_depth=16") {
				t.Fatalf("standalone validation missed default budget: %+v", diags)
			}
		})
	}
	input := filepath.Join(t.TempDir(), "oversized.dsl")
	if err := os.WriteFile(input, []byte(source), 0o600); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if CLIValidate(input, &output) == 0 || !strings.Contains(output.String(), "max_depth=16") {
		t.Fatalf("CLI validation accepted oversized input: %s", output.String())
	}
}

func TestDecisionRuleLimitsHelmPreservesEnclosingBudget(t *testing.T) {
	source := `SIGNAL keyword urgent {operator: "OR" keywords: ["urgent"]}
ROUTE bounded {PRIORITY 1 WHEN ` + strings.Repeat("NOT ", 16) + `keyword("urgent") MODEL "test"}`
	dir := t.TempDir()
	input, base, output := filepath.Join(dir, "rules.dsl"), filepath.Join(dir, "base.yaml"), filepath.Join(dir, "values.yaml")
	for path, data := range map[string]string{
		input: source,
		base:  "global:\n  router:\n    decision_rule_limits: {max_depth: 32, max_nodes: 512}\n",
	} {
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := CLICompile(input, output, "helm", "", "", base); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var values map[string]interface{}
	if err = yaml.Unmarshal(data, &values); err != nil {
		t.Fatal(err)
	}
	document, err := yaml.Marshal(values["config"])
	if err != nil {
		t.Fatal(err)
	}
	limits, err := config.DecisionRuleLimitsFromYAML(document)
	if err != nil {
		t.Fatalf("emitted Helm config cannot reload: %v", err)
	}
	depth, nodes, err := limits.Effective()
	if err != nil || depth != 32 || nodes != 512 {
		t.Fatalf("Helm lost overrides: depth=%d nodes=%d err=%v", depth, nodes, err)
	}
}

func TestDecisionRuleLimitsCompileBeforeRecursiveLowering(t *testing.T) {
	var expression BoolExpr = &SignalRefExpr{SignalType: "keyword", SignalName: "urgent"}
	for i := 1; i < 17; i++ {
		expression = &BoolNot{Expr: expression}
	}
	program := &Program{Routes: []*RouteDecl{{Name: "bounded", When: expression}}}
	if _, errs := CompileAST(program); len(errs) != 1 || !strings.Contains(errs[0].Error(), "depth 17 exceeds max_depth=16") {
		t.Fatalf("expected default depth rejection, got %v", errs)
	}
	maxDepth := 32
	cfg, errs := CompileASTWithLimits(program, config.DecisionRuleLimits{MaxDepth: &maxDepth})
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	if _, err := Decompile(cfg); err != nil {
		t.Fatal(err)
	}
	crd, err := EmitCRD(cfg, "router", "default")
	if err != nil || !strings.Contains(string(crd), "max_depth: 32") {
		t.Fatalf("CRD must carry the enclosing rule budget: %s, %v", crd, err)
	}
	// In-memory callers get the same protection as YAML and AST producers.
	cfg.DecisionRuleLimits = config.DecisionRuleLimits{}
	if _, err := Decompile(cfg); err == nil || !strings.Contains(err.Error(), "max_depth=16") {
		t.Fatalf("expected decompilation to reject before recursion, got %v", err)
	}
	if _, err := EmitCRD(cfg, "router", "default"); err == nil {
		t.Fatal("CRD emitter must validate in-memory input before serialization")
	}
	if DecompileRoutingToAST(cfg) != nil {
		t.Fatal("unchecked AST decompilation must not recurse over an oversized tree")
	}
}

func TestDecisionRuleLimitsFlattenedASTAndRecipes(t *testing.T) {
	var expression BoolExpr = &SignalRefExpr{SignalType: "keyword", SignalName: "urgent"}
	for i := 1; i < 255; i++ {
		expression = &BoolAnd{Left: expression, Right: &SignalRefExpr{SignalType: "keyword", SignalName: "urgent"}}
	}
	program := &Program{Routes: []*RouteDecl{{Name: "bounded", When: expression}}}
	if _, errs := CompileAST(program); len(errs) > 0 {
		t.Fatalf("255 flattened leaves plus root must pass, got %v", errs)
	}
	expression = &BoolAnd{Left: expression, Right: &SignalRefExpr{SignalType: "keyword", SignalName: "urgent"}}
	program.Routes[0].When = expression
	program = &Program{Recipes: []*RecipeDecl{{Name: "support", Program: program}}}
	if _, errs := CompileAST(program); len(errs) != 1 || !strings.Contains(errs[0].Error(), `routing recipe "support": decision "bounded": rules.conditions[255]: node count 257`) {
		t.Fatalf("expected recipe node rejection, got %v", errs)
	}
}

func TestDecisionRuleLimitsMergeUsesEnclosingBudget(t *testing.T) {
	program := &Program{Routes: []*RouteDecl{{Name: "bounded", When: &BoolNot{Expr: &SignalRefExpr{SignalType: "keyword", SignalName: "urgent"}}}}}
	cfg, errs := CompileAST(program)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	base := []byte("version: v0.3\nglobal:\n  router:\n    decision_rule_limits:\n      max_nodes: 1\n")
	if _, err := MergeRoutingIntoBase(cfg, base); err == nil || !strings.Contains(err.Error(), "node count 2 exceeds max_nodes=1") {
		t.Fatalf("expected merge to use base limits, got %v", err)
	}
	base = []byte("version: v0.3\nglobal:\n  router:\n    decision_rule_limits:\n      max_nodes: 512\n")
	merged, err := MergeRoutingIntoBase(cfg, base)
	if err != nil || !strings.Contains(string(merged), "max_nodes: 512") {
		t.Fatalf("merge must preserve configured limits: %s, %v", merged, err)
	}
}
