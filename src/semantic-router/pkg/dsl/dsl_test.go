package dsl

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ---------- Lexer Tests ----------

func TestLexBasicTokens(t *testing.T) {
	input := `SIGNAL keyword urgent { operator: "any" }`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("unexpected lex errors: %v", errs)
	}

	expected := []TokenType{
		TOKEN_SIGNAL, TOKEN_IDENT, TOKEN_IDENT, TOKEN_LBRACE,
		TOKEN_IDENT, TOKEN_COLON, TOKEN_STRING,
		TOKEN_RBRACE, TOKEN_EOF,
	}
	if len(tokens) != len(expected) {
		t.Fatalf("expected %d tokens, got %d: %v", len(expected), len(tokens), tokens)
	}
	for i, exp := range expected {
		if tokens[i].Type != exp {
			t.Errorf("token[%d]: expected %s, got %s (%q)", i, exp, tokens[i].Type, tokens[i].Literal)
		}
	}
}

func TestLexNumbers(t *testing.T) {
	input := `threshold: 0.75 port: 8080`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	// IDENT COLON FLOAT IDENT COLON INT EOF
	if tokens[2].Type != TOKEN_FLOAT || tokens[2].Literal != "0.75" {
		t.Errorf("expected FLOAT 0.75, got %s %q", tokens[2].Type, tokens[2].Literal)
	}
	if tokens[5].Type != TOKEN_INT || tokens[5].Literal != "8080" {
		t.Errorf("expected INT 8080, got %s %q", tokens[5].Type, tokens[5].Literal)
	}
}

func TestLexBooleans(t *testing.T) {
	input := `enabled: true disabled: false`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	if tokens[2].Type != TOKEN_BOOL || tokens[2].Literal != "true" {
		t.Errorf("expected BOOL true, got %s %q", tokens[2].Type, tokens[2].Literal)
	}
	if tokens[5].Type != TOKEN_BOOL || tokens[5].Literal != "false" {
		t.Errorf("expected BOOL false, got %s %q", tokens[5].Type, tokens[5].Literal)
	}
}

func TestLexStringEscape(t *testing.T) {
	input := `"hello \"world\""`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	if tokens[0].Type != TOKEN_STRING || tokens[0].Literal != `hello "world"` {
		t.Errorf("expected STRING with escaped quotes, got %q", tokens[0].Literal)
	}
}

func TestLexComments(t *testing.T) {
	input := `# this is a comment
SIGNAL keyword test {
  # another comment
}`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	// Comments are skipped, so we should get: SIGNAL IDENT IDENT LBRACE RBRACE EOF
	if tokens[0].Type != TOKEN_SIGNAL {
		t.Errorf("expected SIGNAL, got %s", tokens[0].Type)
	}
}

func TestLexKeywords(t *testing.T) {
	input := `SIGNAL ROUTE PLUGIN BACKEND GLOBAL PRIORITY WHEN MODEL ALGORITHM AND OR NOT`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	expected := []TokenType{
		TOKEN_SIGNAL, TOKEN_ROUTE, TOKEN_PLUGIN, TOKEN_BACKEND, TOKEN_GLOBAL,
		TOKEN_PRIORITY, TOKEN_WHEN, TOKEN_MODEL, TOKEN_ALGORITHM,
		TOKEN_AND, TOKEN_OR, TOKEN_NOT, TOKEN_EOF,
	}
	for i, exp := range expected {
		if tokens[i].Type != exp {
			t.Errorf("token[%d]: expected %s, got %s", i, exp, tokens[i].Type)
		}
	}
}

func TestLexPositionTracking(t *testing.T) {
	input := "SIGNAL\nROUTE"
	tokens, _ := Lex(input)
	if tokens[0].Pos.Line != 1 || tokens[0].Pos.Column != 1 {
		t.Errorf("SIGNAL pos: expected (1,1), got (%d,%d)", tokens[0].Pos.Line, tokens[0].Pos.Column)
	}
	if tokens[1].Pos.Line != 2 || tokens[1].Pos.Column != 1 {
		t.Errorf("ROUTE pos: expected (2,1), got (%d,%d)", tokens[1].Pos.Line, tokens[1].Pos.Column)
	}
}

// ---------- Parser Tests ----------

func TestParseMinimalSignal(t *testing.T) {
	input := `SIGNAL domain math {
  description: "Math domain"
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	if len(prog.Signals) != 1 {
		t.Fatalf("expected 1 signal, got %d", len(prog.Signals))
	}
	s := prog.Signals[0]
	if s.SignalType != "domain" || s.Name != "math" {
		t.Errorf("expected domain/math, got %s/%s", s.SignalType, s.Name)
	}
	if desc, ok := s.Fields["description"]; ok {
		if sv, ok := desc.(StringValue); !ok || sv.V != "Math domain" {
			t.Errorf("unexpected description: %v", desc)
		}
	} else {
		t.Error("missing description field")
	}
}

func TestParseKeywordSignal(t *testing.T) {
	input := `SIGNAL keyword urgent_request {
  operator: "any"
  keywords: ["urgent", "asap", "emergency"]
  case_sensitive: false
  method: "regex"
  fuzzy_match: true
  fuzzy_threshold: 2
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	s := prog.Signals[0]
	if s.SignalType != "keyword" {
		t.Errorf("expected keyword signal type, got %s", s.SignalType)
	}
	if kw, ok := s.Fields["keywords"]; ok {
		if av, ok := kw.(ArrayValue); !ok || len(av.Items) != 3 {
			t.Errorf("expected 3 keywords, got %v", kw)
		}
	}
}

func TestParseRoute(t *testing.T) {
	input := `SIGNAL domain math { description: "math" }

ROUTE math_decision (description = "Math route") {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5:3b" (reasoning = true, effort = "high")
  PLUGIN system_prompt {
    system_prompt: "You are a math expert."
  }
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	if len(prog.Routes) != 1 {
		t.Fatalf("expected 1 route, got %d", len(prog.Routes))
	}

	r := prog.Routes[0]
	if r.Name != "math_decision" {
		t.Errorf("expected route name math_decision, got %s", r.Name)
	}
	if r.Description != "Math route" {
		t.Errorf("expected description 'Math route', got %q", r.Description)
	}
	if r.Priority != 100 {
		t.Errorf("expected priority 100, got %d", r.Priority)
	}

	// Check WHEN
	ref, ok := r.When.(*SignalRefExpr)
	if !ok {
		t.Fatalf("expected SignalRefExpr, got %T", r.When)
	}
	if ref.SignalType != "domain" || ref.SignalName != "math" {
		t.Errorf("expected domain(math), got %s(%s)", ref.SignalType, ref.SignalName)
	}

	// Check MODEL
	if len(r.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(r.Models))
	}
	m := r.Models[0]
	if m.Model != "qwen2.5:3b" {
		t.Errorf("expected model qwen2.5:3b, got %s", m.Model)
	}
	if m.Reasoning == nil || *m.Reasoning != true {
		t.Error("expected reasoning = true")
	}
	if m.Effort != "high" {
		t.Errorf("expected effort high, got %s", m.Effort)
	}

	// Check PLUGIN
	if len(r.Plugins) != 1 {
		t.Fatalf("expected 1 plugin, got %d", len(r.Plugins))
	}
}

func TestParseBoolExprComplex(t *testing.T) {
	input := `ROUTE test {
  PRIORITY 1
  WHEN keyword("urgent") AND (domain("math") OR embedding("ai")) AND NOT domain("other")
  MODEL "test:1b"
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}

	r := prog.Routes[0]
	// Should be: AND(AND(keyword, OR(domain, embedding)), NOT(domain))
	topAnd, ok := r.When.(*BoolAnd)
	if !ok {
		t.Fatalf("expected top-level AND, got %T", r.When)
	}

	// Right should be NOT
	notExpr, ok := topAnd.Right.(*BoolNot)
	if !ok {
		t.Fatalf("expected NOT on right, got %T", topAnd.Right)
	}
	notRef, ok := notExpr.Expr.(*SignalRefExpr)
	if !ok || notRef.SignalType != "domain" || notRef.SignalName != "other" {
		t.Errorf("expected NOT domain(other), got %v", notExpr.Expr)
	}

	// Left should be AND(keyword, OR(domain, embedding))
	leftAnd, ok := topAnd.Left.(*BoolAnd)
	if !ok {
		t.Fatalf("expected inner AND, got %T", topAnd.Left)
	}
	kwRef, ok := leftAnd.Left.(*SignalRefExpr)
	if !ok || kwRef.SignalType != "keyword" || kwRef.SignalName != "urgent" {
		t.Errorf("expected keyword(urgent), got %v", leftAnd.Left)
	}
	orExpr, ok := leftAnd.Right.(*BoolOr)
	if !ok {
		t.Fatalf("expected OR, got %T", leftAnd.Right)
	}
	_ = orExpr // structure validated
}

func TestParseMultiModel(t *testing.T) {
	input := `ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "qwen3:70b" (reasoning = true, effort = "high", param_size = "70b"),
        "qwen2.5:3b" (reasoning = false, param_size = "3b")
  ALGORITHM confidence {
    confidence_method: "hybrid"
    threshold: 0.5
  }
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	r := prog.Routes[0]
	if len(r.Models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(r.Models))
	}
	if r.Models[0].ParamSize != "70b" || r.Models[1].ParamSize != "3b" {
		t.Errorf("param_size mismatch: %s, %s", r.Models[0].ParamSize, r.Models[1].ParamSize)
	}
	if r.Algorithm == nil || r.Algorithm.AlgoType != "confidence" {
		t.Error("expected confidence algorithm")
	}
}

func TestParsePluginTemplate(t *testing.T) {
	input := `PLUGIN safe_pii pii {
  enabled: true
  pii_types_allowed: []
}

ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN safe_pii
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	if len(prog.Plugins) != 1 {
		t.Fatalf("expected 1 plugin decl, got %d", len(prog.Plugins))
	}
	pd := prog.Plugins[0]
	if pd.Name != "safe_pii" || pd.PluginType != "pii" {
		t.Errorf("expected safe_pii/pii, got %s/%s", pd.Name, pd.PluginType)
	}

	r := prog.Routes[0]
	if len(r.Plugins) != 1 || r.Plugins[0].Name != "safe_pii" {
		t.Error("expected plugin ref to safe_pii")
	}
}

func TestParseBackend(t *testing.T) {
	input := `BACKEND vllm_endpoint ollama {
  address: "127.0.0.1"
  port: 11434
  weight: 1
  type: "ollama"
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	if len(prog.Backends) != 1 {
		t.Fatalf("expected 1 backend, got %d", len(prog.Backends))
	}
	b := prog.Backends[0]
	if b.BackendType != "vllm_endpoint" || b.Name != "ollama" {
		t.Errorf("expected vllm_endpoint/ollama, got %s/%s", b.BackendType, b.Name)
	}
}

func TestParseGlobal(t *testing.T) {
	input := `GLOBAL {
  default_model: "qwen2.5:3b"
  strategy: "priority"
  observability: {
    metrics: { enabled: true }
    tracing: {
      enabled: true
      provider: "opentelemetry"
    }
  }
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	if prog.Global == nil {
		t.Fatal("expected GLOBAL block")
	}
	if dm, ok := prog.Global.Fields["default_model"]; ok {
		if sv, ok := dm.(StringValue); !ok || sv.V != "qwen2.5:3b" {
			t.Errorf("unexpected default_model: %v", dm)
		}
	}
}

func TestParseErrorRecovery(t *testing.T) {
	input := `SIGNAL domain math {
  description "missing colon"
}

SIGNAL domain physics {
  description: "Physics"
}`
	prog, errs := Parse(input)
	// Should have errors but still parse the second signal
	if len(errs) == 0 {
		t.Error("expected parse errors")
	}
	// We should recover and get at least the second signal
	if prog == nil {
		t.Fatal("expected non-nil program even with errors")
	}
}

// ---------- Compiler Tests ----------

func TestCompileMinimal(t *testing.T) {
	input := `SIGNAL domain math {
  description: "Mathematics"
  mmlu_categories: ["math"]
}

ROUTE math_route {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5:3b" (reasoning = true, effort = "high")
  PLUGIN system_prompt {
    system_prompt: "You are a math expert."
  }
}

BACKEND vllm_endpoint ollama {
  address: "127.0.0.1"
  port: 11434
  weight: 1
}

GLOBAL {
  default_model: "qwen2.5:3b"
  strategy: "priority"
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	// Check signal
	if len(cfg.Categories) != 1 {
		t.Fatalf("expected 1 category, got %d", len(cfg.Categories))
	}
	if cfg.Categories[0].Name != "math" {
		t.Errorf("expected category name 'math', got %q", cfg.Categories[0].Name)
	}
	if len(cfg.Categories[0].MMLUCategories) != 1 || cfg.Categories[0].MMLUCategories[0] != "math" {
		t.Errorf("unexpected mmlu_categories: %v", cfg.Categories[0].MMLUCategories)
	}

	// Check decision
	if len(cfg.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(cfg.Decisions))
	}
	d := cfg.Decisions[0]
	if d.Name != "math_route" {
		t.Errorf("expected decision name 'math_route', got %q", d.Name)
	}
	if d.Priority != 100 {
		t.Errorf("expected priority 100, got %d", d.Priority)
	}
	if d.Rules.Type != "domain" || d.Rules.Name != "math" {
		t.Errorf("expected rules type=domain name=math, got type=%s name=%s", d.Rules.Type, d.Rules.Name)
	}
	if len(d.ModelRefs) != 1 {
		t.Fatalf("expected 1 model ref, got %d", len(d.ModelRefs))
	}
	if d.ModelRefs[0].Model != "qwen2.5:3b" {
		t.Errorf("expected model qwen2.5:3b, got %s", d.ModelRefs[0].Model)
	}
	if d.ModelRefs[0].UseReasoning == nil || *d.ModelRefs[0].UseReasoning != true {
		t.Error("expected use_reasoning = true")
	}
	if d.ModelRefs[0].ReasoningEffort != "high" {
		t.Errorf("expected reasoning_effort high, got %s", d.ModelRefs[0].ReasoningEffort)
	}

	// Check plugins
	if len(d.Plugins) != 1 || d.Plugins[0].Type != "system_prompt" {
		t.Errorf("expected 1 system_prompt plugin, got %v", d.Plugins)
	}

	// Check backend
	if len(cfg.VLLMEndpoints) != 1 {
		t.Fatalf("expected 1 endpoint, got %d", len(cfg.VLLMEndpoints))
	}
	ep := cfg.VLLMEndpoints[0]
	if ep.Name != "ollama" || ep.Address != "127.0.0.1" || ep.Port != 11434 {
		t.Errorf("unexpected endpoint: %+v", ep)
	}

	// Check global
	if cfg.DefaultModel != "qwen2.5:3b" {
		t.Errorf("expected default_model qwen2.5:3b, got %s", cfg.DefaultModel)
	}
	if cfg.Strategy != "priority" {
		t.Errorf("expected strategy priority, got %s", cfg.Strategy)
	}
}

func TestCompilePluginTemplate(t *testing.T) {
	input := `PLUGIN safe_pii pii {
  enabled: true
  pii_types_allowed: []
}

ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN safe_pii
}

SIGNAL domain test { description: "test" }`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(cfg.Decisions))
	}
	if len(cfg.Decisions[0].Plugins) != 1 {
		t.Fatalf("expected 1 plugin, got %d", len(cfg.Decisions[0].Plugins))
	}
	p := cfg.Decisions[0].Plugins[0]
	if p.Type != "pii" {
		t.Errorf("expected plugin type pii, got %s", p.Type)
	}
}

func TestCompilePluginTemplateWithOverride(t *testing.T) {
	input := `PLUGIN default_cache semantic_cache {
  enabled: true
  similarity_threshold: 0.80
}

SIGNAL domain test { description: "test" }

ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN default_cache {
    similarity_threshold: 0.95
  }
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	p := cfg.Decisions[0].Plugins[0]
	if p.Type != "semantic-cache" { // normalized
		t.Errorf("expected plugin type semantic-cache, got %s", p.Type)
	}
}

func TestCompileBoolExprToRuleNode(t *testing.T) {
	input := `SIGNAL keyword urgent { operator: "any" keywords: ["urgent"] }
SIGNAL domain math { description: "math" }
SIGNAL embedding ai { threshold: 0.7 candidates: ["AI"] }
SIGNAL domain other { description: "other" }

ROUTE test {
  PRIORITY 1
  WHEN keyword("urgent") AND (domain("math") OR embedding("ai")) AND NOT domain("other")
  MODEL "m:1b"
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	rules := cfg.Decisions[0].Rules
	if rules.Operator != "AND" {
		t.Fatalf("expected top-level AND, got %s", rules.Operator)
	}
	// After N-ary flattening, the AND has 3 direct children:
	// keyword("urgent"), (domain("math") OR embedding("ai")), NOT domain("other")
	if len(rules.Conditions) != 3 {
		t.Fatalf("expected 3 top conditions (N-ary flattened), got %d", len(rules.Conditions))
	}
}

func TestCompileAlgorithm(t *testing.T) {
	input := `SIGNAL domain test { description: "test" }

ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m1:7b", "m2:3b"
  ALGORITHM confidence {
    confidence_method: "hybrid"
    threshold: 0.5
    hybrid_weights: { logprob_weight: 0.6, margin_weight: 0.4 }
    on_error: "skip"
  }
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	algo := cfg.Decisions[0].Algorithm
	if algo == nil {
		t.Fatal("expected algorithm")
	}
	if algo.Type != "confidence" {
		t.Errorf("expected confidence, got %s", algo.Type)
	}
	if algo.Confidence == nil {
		t.Fatal("expected confidence config")
	}
	if algo.Confidence.ConfidenceMethod != "hybrid" {
		t.Errorf("expected hybrid method, got %s", algo.Confidence.ConfidenceMethod)
	}
	if algo.Confidence.Threshold != 0.5 {
		t.Errorf("expected threshold 0.5, got %f", algo.Confidence.Threshold)
	}
	if algo.Confidence.HybridWeights == nil {
		t.Fatal("expected hybrid weights")
	}
	if algo.Confidence.HybridWeights.LogprobWeight != 0.6 {
		t.Errorf("expected logprob_weight 0.6, got %f", algo.Confidence.HybridWeights.LogprobWeight)
	}
}

func TestCompileReMoMAlgorithm(t *testing.T) {
	input := `SIGNAL domain test { description: "test" }

ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m1:7b", "m2:3b"
  ALGORITHM remom {
    breadth_schedule: [8, 2]
    model_distribution: "weighted"
    temperature: 1.0
    include_reasoning: true
    on_error: "skip"
  }
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	algo := cfg.Decisions[0].Algorithm
	if algo.ReMoM == nil {
		t.Fatal("expected remom config")
	}
	if len(algo.ReMoM.BreadthSchedule) != 2 || algo.ReMoM.BreadthSchedule[0] != 8 {
		t.Errorf("unexpected breadth_schedule: %v", algo.ReMoM.BreadthSchedule)
	}
}

func TestCompileAllSignalTypes(t *testing.T) {
	input := `
SIGNAL keyword kw { operator: "any" keywords: ["test"] }
SIGNAL embedding emb { threshold: 0.75 candidates: ["test"] }
SIGNAL domain dom { description: "test" mmlu_categories: ["math"] }
SIGNAL fact_check fc { description: "fact check" }
SIGNAL user_feedback uf { description: "feedback" }
SIGNAL preference pref { description: "preference" }
SIGNAL language lang { description: "English" }
SIGNAL context ctx { min_tokens: "1K" max_tokens: "32K" }
SIGNAL complexity comp { threshold: 0.1 hard: { candidates: ["hard task"] } easy: { candidates: ["easy task"] } }
SIGNAL modality mod { description: "image" }
SIGNAL authz auth { role: "admin" subjects: [{ kind: "User", name: "admin" }] }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	if len(cfg.KeywordRules) != 1 {
		t.Errorf("expected 1 keyword rule, got %d", len(cfg.KeywordRules))
	}
	if len(cfg.EmbeddingRules) != 1 {
		t.Errorf("expected 1 embedding rule, got %d", len(cfg.EmbeddingRules))
	}
	if len(cfg.Categories) != 1 {
		t.Errorf("expected 1 category, got %d", len(cfg.Categories))
	}
	if len(cfg.FactCheckRules) != 1 {
		t.Errorf("expected 1 fact_check rule, got %d", len(cfg.FactCheckRules))
	}
	if len(cfg.UserFeedbackRules) != 1 {
		t.Errorf("expected 1 user_feedback rule, got %d", len(cfg.UserFeedbackRules))
	}
	if len(cfg.PreferenceRules) != 1 {
		t.Errorf("expected 1 preference rule, got %d", len(cfg.PreferenceRules))
	}
	if len(cfg.LanguageRules) != 1 {
		t.Errorf("expected 1 language rule, got %d", len(cfg.LanguageRules))
	}
	if len(cfg.ContextRules) != 1 {
		t.Errorf("expected 1 context rule, got %d", len(cfg.ContextRules))
	}
	if cfg.ContextRules[0].MinTokens != "1K" || cfg.ContextRules[0].MaxTokens != "32K" {
		t.Errorf("unexpected context tokens: %s - %s", cfg.ContextRules[0].MinTokens, cfg.ContextRules[0].MaxTokens)
	}
	if len(cfg.ComplexityRules) != 1 {
		t.Errorf("expected 1 complexity rule, got %d", len(cfg.ComplexityRules))
	}
	if len(cfg.ModalityRules) != 1 {
		t.Errorf("expected 1 modality rule, got %d", len(cfg.ModalityRules))
	}
	if len(cfg.RoleBindings) != 1 {
		t.Errorf("expected 1 role binding, got %d", len(cfg.RoleBindings))
	}
	if cfg.RoleBindings[0].Role != "admin" {
		t.Errorf("expected role admin, got %s", cfg.RoleBindings[0].Role)
	}
	if len(cfg.RoleBindings[0].Subjects) != 1 || cfg.RoleBindings[0].Subjects[0].Kind != "User" {
		t.Errorf("unexpected subjects: %v", cfg.RoleBindings[0].Subjects)
	}
}

func TestCompileEmitYAML(t *testing.T) {
	input := `SIGNAL domain math {
  description: "Mathematics"
  mmlu_categories: ["math"]
}

ROUTE math_route {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5:3b" (reasoning = true)
}

GLOBAL {
  default_model: "qwen2.5:3b"
  strategy: "priority"
}`

	yamlBytes, errs := EmitYAML(input)
	if len(errs) > 0 {
		t.Fatalf("emit errors: %v", errs)
	}

	yamlStr := string(yamlBytes)
	// Basic sanity checks on the YAML output
	if !strings.Contains(yamlStr, "default_model: qwen2.5:3b") {
		t.Error("YAML should contain default_model")
	}
	if !strings.Contains(yamlStr, "strategy: priority") {
		t.Error("YAML should contain strategy")
	}
	if !strings.Contains(yamlStr, "name: math") {
		t.Error("YAML should contain category name")
	}
}

// ---------- Full Example Test ----------

const fullDSLExample = `
# Signals
SIGNAL domain math {
  description: "Mathematics and quantitative reasoning"
  mmlu_categories: ["math"]
}

SIGNAL domain physics {
  description: "Physics and physical sciences"
  mmlu_categories: ["physics"]
}

SIGNAL domain other {
  description: "General knowledge"
  mmlu_categories: ["other"]
}

SIGNAL embedding ai_topics {
  threshold: 0.75
  candidates: ["machine learning", "neural network", "deep learning"]
  aggregation_method: "max"
}

SIGNAL keyword urgent_request {
  operator: "any"
  keywords: ["urgent", "asap", "emergency"]
  method: "regex"
  case_sensitive: false
}

SIGNAL context long_context {
  min_tokens: "4K"
  max_tokens: "32K"
}

SIGNAL complexity code_complexity {
  threshold: 0.1
  hard: { candidates: ["implement distributed system"] }
  easy: { candidates: ["print hello world"] }
}

# Plugins
PLUGIN safe_pii pii {
  enabled: true
  pii_types_allowed: []
}

PLUGIN default_cache semantic_cache {
  enabled: true
  similarity_threshold: 0.80
}

# Routes
ROUTE math_decision (description = "Math route") {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5:3b" (reasoning = true, effort = "high")
  PLUGIN system_prompt {
    system_prompt: "You are a math expert."
  }
  PLUGIN safe_pii
}

ROUTE physics_decision {
  PRIORITY 100
  WHEN domain("physics")
  MODEL "qwen2.5:3b" (reasoning = true)
  PLUGIN safe_pii
}

ROUTE urgent_ai_route {
  PRIORITY 200
  WHEN keyword("urgent_request") AND embedding("ai_topics") AND NOT domain("other")
  MODEL "qwen3:70b" (reasoning = true, effort = "high"),
        "qwen2.5:3b" (reasoning = false)
  ALGORITHM confidence {
    confidence_method: "hybrid"
    threshold: 0.5
    hybrid_weights: { logprob_weight: 0.6, margin_weight: 0.4 }
    on_error: "skip"
  }
  PLUGIN safe_pii
  PLUGIN default_cache
}

ROUTE general_decision {
  PRIORITY 50
  WHEN domain("other")
  MODEL "qwen2.5:3b" (reasoning = false)
  PLUGIN safe_pii
  PLUGIN default_cache
}

# Backends
BACKEND vllm_endpoint ollama {
  address: "127.0.0.1"
  port: 11434
  weight: 1
  type: "ollama"
}

BACKEND provider_profile openai_prod {
  type: "openai"
  base_url: "https://api.openai.com/v1"
}

BACKEND embedding_model ultra {
  mmbert_model_path: "models/mom-embedding-ultra"
  use_cpu: true
  hnsw_config: {
    model_type: "mmbert"
    preload_embeddings: true
    target_dimension: 768
  }
}

BACKEND semantic_cache main_cache {
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.8
  max_entries: 1000
  ttl_seconds: 3600
  use_hnsw: true
}

# Global
GLOBAL {
  default_model: "qwen2.5:3b"
  strategy: "priority"
  default_reasoning_effort: "low"

  prompt_guard: {
    enabled: true
    threshold: 0.7
  }

  observability: {
    metrics: { enabled: true }
    tracing: {
      enabled: true
      provider: "opentelemetry"
      exporter: {
        type: "otlp"
        endpoint: "jaeger:4317"
        insecure: true
      }
      sampling: { type: "always_on", rate: 1.0 }
    }
  }
}
`

func TestCompileFullExample(t *testing.T) {
	cfg, errs := Compile(fullDSLExample)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	// Signals
	if len(cfg.Categories) != 3 {
		t.Errorf("expected 3 categories, got %d", len(cfg.Categories))
	}
	if len(cfg.EmbeddingRules) != 1 {
		t.Errorf("expected 1 embedding rule, got %d", len(cfg.EmbeddingRules))
	}
	if len(cfg.KeywordRules) != 1 {
		t.Errorf("expected 1 keyword rule, got %d", len(cfg.KeywordRules))
	}
	if len(cfg.ContextRules) != 1 {
		t.Errorf("expected 1 context rule, got %d", len(cfg.ContextRules))
	}
	if len(cfg.ComplexityRules) != 1 {
		t.Errorf("expected 1 complexity rule, got %d", len(cfg.ComplexityRules))
	}

	// Decisions
	if len(cfg.Decisions) != 4 {
		t.Errorf("expected 4 decisions, got %d", len(cfg.Decisions))
	}

	// Check urgent_ai_route has complex bool expr
	var urgentRoute *struct {
		name  string
		rules interface{}
	}
	for _, d := range cfg.Decisions {
		if d.Name == "urgent_ai_route" {
			if d.Rules.Operator != "AND" {
				t.Errorf("expected AND operator for urgent route rules, got %s", d.Rules.Operator)
			}
			if len(d.ModelRefs) != 2 {
				t.Errorf("expected 2 model refs, got %d", len(d.ModelRefs))
			}
			if d.Algorithm == nil || d.Algorithm.Type != "confidence" {
				t.Error("expected confidence algorithm")
			}
			break
		}
	}
	_ = urgentRoute

	// Backends
	if len(cfg.VLLMEndpoints) != 1 {
		t.Errorf("expected 1 vllm endpoint, got %d", len(cfg.VLLMEndpoints))
	}
	if len(cfg.ProviderProfiles) != 1 {
		t.Errorf("expected 1 provider profile, got %d", len(cfg.ProviderProfiles))
	}

	// Global
	if cfg.DefaultModel != "qwen2.5:3b" {
		t.Errorf("expected default_model qwen2.5:3b, got %s", cfg.DefaultModel)
	}
	if cfg.Strategy != "priority" {
		t.Errorf("expected strategy priority, got %s", cfg.Strategy)
	}
	if cfg.DefaultReasoningEffort != "low" {
		t.Errorf("expected default_reasoning_effort low, got %s", cfg.DefaultReasoningEffort)
	}

	// Semantic cache backend
	if !cfg.SemanticCache.Enabled {
		t.Error("expected semantic cache enabled")
	}
	if cfg.SemanticCache.MaxEntries != 1000 {
		t.Errorf("expected max_entries 1000, got %d", cfg.SemanticCache.MaxEntries)
	}

	// Observability
	if !cfg.Observability.Tracing.Enabled {
		t.Error("expected tracing enabled")
	}
	if cfg.Observability.Tracing.Exporter.Endpoint != "jaeger:4317" {
		t.Errorf("expected exporter endpoint jaeger:4317, got %s", cfg.Observability.Tracing.Exporter.Endpoint)
	}

	// Emit YAML to verify it works
	yamlBytes, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("YAML emit error: %v", err)
	}
	yamlStr := string(yamlBytes)
	if !strings.Contains(yamlStr, "default_model: qwen2.5:3b") {
		t.Error("YAML missing default_model")
	}
	if !strings.Contains(yamlStr, "name: math") {
		t.Error("YAML missing math category")
	}
}

// ==================== P0 Tests ====================

// ---------- P0-1: YAML Round-Trip Test ----------
// DSL → Compile → RouterConfig → EmitYAML → yaml.Unmarshal → verify all fields survive the round trip.

func TestYAMLRoundTrip(t *testing.T) {
	input := `
SIGNAL domain math {
  description: "Mathematics"
  mmlu_categories: ["math"]
}

SIGNAL keyword urgent {
  operator: "any"
  keywords: ["urgent", "asap"]
  case_sensitive: false
}

SIGNAL embedding ai_topics {
  threshold: 0.75
  candidates: ["machine learning", "deep learning"]
  aggregation_method: "max"
}

ROUTE math_route {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5:3b" (reasoning = true, effort = "high")
  PLUGIN system_prompt {
    system_prompt: "You are a math expert."
  }
}

ROUTE urgent_ai_route {
  PRIORITY 200
  WHEN keyword("urgent") AND embedding("ai_topics")
  MODEL "qwen3:70b" (reasoning = true, effort = "high"),
        "qwen2.5:3b" (reasoning = false)
  ALGORITHM confidence {
    confidence_method: "hybrid"
    threshold: 0.5
    hybrid_weights: { logprob_weight: 0.6, margin_weight: 0.4 }
    on_error: "skip"
  }
}

BACKEND vllm_endpoint ollama {
  address: "127.0.0.1"
  port: 11434
  weight: 1
  type: "ollama"
}

GLOBAL {
  default_model: "qwen2.5:3b"
  strategy: "priority"
  default_reasoning_effort: "low"
  observability: {
    metrics: { enabled: true }
    tracing: {
      enabled: true
      provider: "opentelemetry"
      exporter: {
        type: "otlp"
        endpoint: "jaeger:4317"
        insecure: true
      }
    }
  }
}
`

	// Step 1: DSL → RouterConfig
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	// Step 2: RouterConfig → YAML bytes
	yamlBytes, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("emit YAML error: %v", err)
	}

	// Step 3: YAML bytes → RouterConfig (using yaml.v2 like the real loader)
	var roundTripped config.RouterConfig
	if err := yaml.Unmarshal(yamlBytes, &roundTripped); err != nil {
		t.Fatalf("yaml.Unmarshal failed: %v\nYAML content:\n%s", err, string(yamlBytes))
	}

	// Step 4: Verify key fields survived the round trip
	// -- Global settings
	if roundTripped.DefaultModel != "qwen2.5:3b" {
		t.Errorf("round-trip: default_model = %q, want %q", roundTripped.DefaultModel, "qwen2.5:3b")
	}
	if roundTripped.Strategy != "priority" {
		t.Errorf("round-trip: strategy = %q, want %q", roundTripped.Strategy, "priority")
	}
	if roundTripped.DefaultReasoningEffort != "low" {
		t.Errorf("round-trip: default_reasoning_effort = %q, want %q", roundTripped.DefaultReasoningEffort, "low")
	}

	// -- Categories
	if len(roundTripped.Categories) != 1 {
		t.Fatalf("round-trip: expected 1 category, got %d", len(roundTripped.Categories))
	}
	if roundTripped.Categories[0].Name != "math" {
		t.Errorf("round-trip: category name = %q, want %q", roundTripped.Categories[0].Name, "math")
	}
	if len(roundTripped.Categories[0].MMLUCategories) != 1 || roundTripped.Categories[0].MMLUCategories[0] != "math" {
		t.Errorf("round-trip: mmlu_categories = %v, want [math]", roundTripped.Categories[0].MMLUCategories)
	}

	// -- Keyword rules
	if len(roundTripped.KeywordRules) != 1 {
		t.Fatalf("round-trip: expected 1 keyword rule, got %d", len(roundTripped.KeywordRules))
	}
	kw := roundTripped.KeywordRules[0]
	if kw.Name != "urgent" {
		t.Errorf("round-trip: keyword rule name = %q, want %q", kw.Name, "urgent")
	}
	if kw.Operator != "any" {
		t.Errorf("round-trip: keyword operator = %q, want %q", kw.Operator, "any")
	}
	if len(kw.Keywords) != 2 {
		t.Errorf("round-trip: keyword count = %d, want 2", len(kw.Keywords))
	}

	// -- Embedding rules
	if len(roundTripped.EmbeddingRules) != 1 {
		t.Fatalf("round-trip: expected 1 embedding rule, got %d", len(roundTripped.EmbeddingRules))
	}
	emb := roundTripped.EmbeddingRules[0]
	if emb.Name != "ai_topics" {
		t.Errorf("round-trip: embedding rule name = %q, want %q", emb.Name, "ai_topics")
	}
	if emb.SimilarityThreshold != 0.75 {
		t.Errorf("round-trip: embedding threshold = %v, want 0.75", emb.SimilarityThreshold)
	}

	// -- Decisions
	if len(roundTripped.Decisions) != 2 {
		t.Fatalf("round-trip: expected 2 decisions, got %d", len(roundTripped.Decisions))
	}

	// Find math_route decision
	var mathDec, urgentDec *config.Decision
	for i := range roundTripped.Decisions {
		switch roundTripped.Decisions[i].Name {
		case "math_route":
			mathDec = &roundTripped.Decisions[i]
		case "urgent_ai_route":
			urgentDec = &roundTripped.Decisions[i]
		}
	}

	if mathDec == nil {
		t.Fatal("round-trip: math_route decision not found")
	}
	if mathDec.Priority != 100 {
		t.Errorf("round-trip: math_route priority = %d, want 100", mathDec.Priority)
	}
	if mathDec.Rules.Type != "domain" || mathDec.Rules.Name != "math" {
		t.Errorf("round-trip: math_route rules = {type: %q, name: %q}, want {type: domain, name: math}",
			mathDec.Rules.Type, mathDec.Rules.Name)
	}
	if len(mathDec.ModelRefs) != 1 {
		t.Fatalf("round-trip: math_route expected 1 model ref, got %d", len(mathDec.ModelRefs))
	}
	if mathDec.ModelRefs[0].Model != "qwen2.5:3b" {
		t.Errorf("round-trip: math_route model = %q, want %q", mathDec.ModelRefs[0].Model, "qwen2.5:3b")
	}
	if mathDec.ModelRefs[0].UseReasoning == nil || *mathDec.ModelRefs[0].UseReasoning != true {
		t.Error("round-trip: math_route model use_reasoning should be true")
	}
	if mathDec.ModelRefs[0].ReasoningEffort != "high" {
		t.Errorf("round-trip: math_route reasoning_effort = %q, want %q", mathDec.ModelRefs[0].ReasoningEffort, "high")
	}

	// Verify plugins survived
	if len(mathDec.Plugins) != 1 {
		t.Fatalf("round-trip: math_route expected 1 plugin, got %d", len(mathDec.Plugins))
	}
	if mathDec.Plugins[0].Type != "system_prompt" {
		t.Errorf("round-trip: math_route plugin type = %q, want %q", mathDec.Plugins[0].Type, "system_prompt")
	}

	// Verify urgent_ai_route
	if urgentDec == nil {
		t.Fatal("round-trip: urgent_ai_route decision not found")
	}
	if urgentDec.Priority != 200 {
		t.Errorf("round-trip: urgent_ai_route priority = %d, want 200", urgentDec.Priority)
	}
	if urgentDec.Rules.Operator != "AND" {
		t.Errorf("round-trip: urgent_ai_route rules operator = %q, want AND", urgentDec.Rules.Operator)
	}
	if len(urgentDec.ModelRefs) != 2 {
		t.Fatalf("round-trip: urgent_ai_route expected 2 model refs, got %d", len(urgentDec.ModelRefs))
	}

	// Verify algorithm
	if urgentDec.Algorithm == nil {
		t.Fatal("round-trip: urgent_ai_route algorithm is nil")
	}
	if urgentDec.Algorithm.Type != "confidence" {
		t.Errorf("round-trip: algorithm type = %q, want %q", urgentDec.Algorithm.Type, "confidence")
	}
	if urgentDec.Algorithm.Confidence == nil {
		t.Fatal("round-trip: algorithm.confidence is nil")
	}
	if urgentDec.Algorithm.Confidence.ConfidenceMethod != "hybrid" {
		t.Errorf("round-trip: confidence_method = %q, want %q", urgentDec.Algorithm.Confidence.ConfidenceMethod, "hybrid")
	}
	if urgentDec.Algorithm.Confidence.Threshold != 0.5 {
		t.Errorf("round-trip: confidence threshold = %v, want 0.5", urgentDec.Algorithm.Confidence.Threshold)
	}
	if urgentDec.Algorithm.Confidence.HybridWeights == nil {
		t.Fatal("round-trip: hybrid_weights is nil")
	}
	if urgentDec.Algorithm.Confidence.HybridWeights.LogprobWeight != 0.6 {
		t.Errorf("round-trip: logprob_weight = %v, want 0.6", urgentDec.Algorithm.Confidence.HybridWeights.LogprobWeight)
	}

	// -- Backend
	if len(roundTripped.VLLMEndpoints) != 1 {
		t.Fatalf("round-trip: expected 1 vllm endpoint, got %d", len(roundTripped.VLLMEndpoints))
	}
	ep := roundTripped.VLLMEndpoints[0]
	if ep.Name != "ollama" {
		t.Errorf("round-trip: endpoint name = %q, want %q", ep.Name, "ollama")
	}
	if ep.Address != "127.0.0.1" || ep.Port != 11434 {
		t.Errorf("round-trip: endpoint = %s:%d, want 127.0.0.1:11434", ep.Address, ep.Port)
	}

	// -- Observability
	if !roundTripped.Observability.Tracing.Enabled {
		t.Error("round-trip: tracing should be enabled")
	}
	if roundTripped.Observability.Tracing.Provider != "opentelemetry" {
		t.Errorf("round-trip: tracing provider = %q, want %q", roundTripped.Observability.Tracing.Provider, "opentelemetry")
	}
	if roundTripped.Observability.Tracing.Exporter.Endpoint != "jaeger:4317" {
		t.Errorf("round-trip: exporter endpoint = %q, want %q", roundTripped.Observability.Tracing.Exporter.Endpoint, "jaeger:4317")
	}
}

// ---------- P0-2: Single-condition WHEN engine compatibility ----------
// Verify that a single-signal WHEN compiles to a leaf RuleNode,
// and that this is compatible with the decision engine's evalNode.

func TestSingleConditionWHENCompatibility(t *testing.T) {
	input := `
SIGNAL domain math { description: "Math" }
ROUTE math_route {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5:3b" (reasoning = true)
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	if len(cfg.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(cfg.Decisions))
	}
	rules := cfg.Decisions[0].Rules

	// Single WHEN domain("math") should produce a leaf node
	if !rules.IsLeaf() {
		t.Fatalf("single WHEN should produce a leaf node, got operator=%q with %d conditions",
			rules.Operator, len(rules.Conditions))
	}
	if rules.Type != "domain" || rules.Name != "math" {
		t.Errorf("leaf node = {type: %q, name: %q}, want {type: domain, name: math}", rules.Type, rules.Name)
	}

	// Verify this survives YAML round-trip
	yamlBytes, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("emit YAML error: %v", err)
	}
	var rt config.RouterConfig
	if err := yaml.Unmarshal(yamlBytes, &rt); err != nil {
		t.Fatalf("yaml.Unmarshal failed: %v\nYAML:\n%s", err, string(yamlBytes))
	}

	rtRules := rt.Decisions[0].Rules
	if !rtRules.IsLeaf() {
		t.Fatalf("after round-trip: expected leaf node, got operator=%q", rtRules.Operator)
	}
	if rtRules.Type != "domain" || rtRules.Name != "math" {
		t.Errorf("after round-trip: leaf = {type: %q, name: %q}, want {type: domain, name: math}",
			rtRules.Type, rtRules.Name)
	}
}

func TestMultiConditionWHENProducesCompositeNode(t *testing.T) {
	input := `
SIGNAL keyword urgent { operator: "any" keywords: ["urgent"] }
SIGNAL domain math { description: "Math" }

ROUTE test_route {
  PRIORITY 1
  WHEN keyword("urgent") AND domain("math")
  MODEL "m:1b" (reasoning = false)
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	rules := cfg.Decisions[0].Rules
	if rules.IsLeaf() {
		t.Fatal("AND expression should produce a composite node, got leaf")
	}
	if rules.Operator != "AND" {
		t.Errorf("expected AND operator, got %q", rules.Operator)
	}
	if len(rules.Conditions) != 2 {
		t.Fatalf("expected 2 conditions, got %d", len(rules.Conditions))
	}

	// Left = keyword("urgent"), Right = domain("math")
	left := rules.Conditions[0]
	right := rules.Conditions[1]
	if left.Type != "keyword" || left.Name != "urgent" {
		t.Errorf("left condition = {type: %q, name: %q}, want keyword/urgent", left.Type, left.Name)
	}
	if right.Type != "domain" || right.Name != "math" {
		t.Errorf("right condition = {type: %q, name: %q}, want domain/math", right.Type, right.Name)
	}
}

// ---------- P0-3: Negative Tests ----------
// Verify that invalid inputs produce errors (not panics) and error messages are accurate.

func TestNegativeInvalidDSLDoesNotPanic(t *testing.T) {
	invalidInputs := []struct {
		name  string
		input string
	}{
		{
			name:  "empty input",
			input: "",
		},
		{
			name:  "garbage tokens",
			input: "@@@ $$$ !!!",
		},
		{
			name:  "missing signal body",
			input: "SIGNAL domain math",
		},
		{
			name:  "missing route body",
			input: "ROUTE test_route",
		},
		{
			name:  "missing WHEN in route",
			input: `ROUTE test { PRIORITY 1 MODEL "m:1b" }`,
		},
		{
			name:  "unclosed string",
			input: `SIGNAL domain math { description: "unclosed }`,
		},
		{
			name:  "unclosed brace",
			input: `SIGNAL domain math { description: "test"`,
		},
		{
			name:  "invalid signal type in WHEN",
			input: `ROUTE test { PRIORITY 1 WHEN 123 MODEL "m:1b" }`,
		},
		{
			name:  "missing colon in field",
			input: `SIGNAL domain math { description "no colon" }`,
		},
		{
			name:  "duplicate GLOBAL blocks",
			input: `GLOBAL { strategy: "priority" } GLOBAL { strategy: "confidence" }`,
		},
	}

	for _, tc := range invalidInputs {
		t.Run(tc.name, func(t *testing.T) {
			// Should not panic regardless of input
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("panicked on input %q: %v", tc.name, r)
					}
				}()
				_, _ = Compile(tc.input)
			}()
		})
	}
}

func TestNegativeCompileUndefinedSignalType(t *testing.T) {
	input := `
SIGNAL nonexistent_type test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN nonexistent_type("test")
  MODEL "m:1b" (reasoning = false)
}
`
	_, errs := Compile(input)
	if len(errs) == 0 {
		t.Error("expected compile error for unknown signal type, got none")
	}
	// Verify error message mentions the unknown type
	found := false
	for _, e := range errs {
		if strings.Contains(e.Error(), "nonexistent_type") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("error should mention 'nonexistent_type', got: %v", errs)
	}
}

func TestNegativeCompileUnknownPluginType(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b" (reasoning = false)
  PLUGIN totally_unknown_plugin {
    foo: "bar"
  }
}
`
	_, errs := Compile(input)
	if len(errs) == 0 {
		t.Error("expected compile error for unknown plugin type, got none")
	}
}

func TestNegativeCompileUnknownAlgorithmType(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b" (reasoning = false)
  ALGORITHM nonexistent_algo {
    foo: "bar"
  }
}
`
	_, errs := Compile(input)
	if len(errs) == 0 {
		t.Error("expected compile error for unknown algorithm type, got none")
	}
}

func TestNegativeCompileUnknownBackendType(t *testing.T) {
	input := `
BACKEND unknown_backend_type myname {
  address: "127.0.0.1"
}
`
	_, errs := Compile(input)
	if len(errs) == 0 {
		t.Error("expected compile error for unknown backend type, got none")
	}
}

func TestNegativeParseErrors(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantError string // substring expected in error
	}{
		{
			name:      "missing field colon",
			input:     `SIGNAL domain math { description "no colon" }`,
			wantError: "unexpected token",
		},
		{
			name:      "unexpected token in route",
			input:     `ROUTE test { FOO 123 }`,
			wantError: "unexpected token",
		},
		{
			name:      "missing model string",
			input:     `ROUTE test { PRIORITY 1 WHEN domain("x") MODEL 123 }`,
			wantError: "unexpected token",
		},
		{
			name: "bad priority type",
			input: `ROUTE test {
  PRIORITY "not_a_number"
  WHEN domain("x")
  MODEL "m:1b"
}`,
			wantError: "unexpected token",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, errs := Parse(tc.input)
			if len(errs) == 0 {
				t.Fatalf("expected parse error containing %q, got none", tc.wantError)
			}
			found := false
			for _, e := range errs {
				if strings.Contains(e.Error(), tc.wantError) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("expected error containing %q, got: %v", tc.wantError, errs)
			}
		})
	}
}

func TestNegativeLexerErrors(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "unclosed string literal",
			input: `"hello world`,
		},
		{
			name:  "unexpected character @",
			input: `SIGNAL @ test {}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, errs := Lex(tc.input)
			if len(errs) == 0 {
				t.Error("expected lexer error, got none")
			}
		})
	}
}

// ==================== P1 Tests ====================
// Edge cases, deeper coverage of each DSL component, and all code paths.

// ---------- P1-1: All Algorithm Types ----------

func TestCompileAllAlgorithmTypes(t *testing.T) {
	algoDSLs := []struct {
		name     string
		algoType string
		body     string
		verify   func(t *testing.T, algo *config.AlgorithmConfig)
	}{
		{
			name:     "ratings",
			algoType: "ratings",
			body:     `max_concurrent: 4 on_error: "skip"`,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.Ratings == nil {
					t.Fatal("expected ratings config")
				}
				if algo.Ratings.MaxConcurrent != 4 {
					t.Errorf("max_concurrent = %d, want 4", algo.Ratings.MaxConcurrent)
				}
			},
		},
		{
			name:     "elo",
			algoType: "elo",
			body:     `initial_rating: 1500.0 k_factor: 32.0 category_weighted: true decay_factor: 0.95 min_comparisons: 10 cost_scaling_factor: 0.5 storage_path: "/tmp/elo"`,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.Elo == nil {
					t.Fatal("expected elo config")
				}
				if algo.Elo.InitialRating != 1500.0 {
					t.Errorf("initial_rating = %v, want 1500", algo.Elo.InitialRating)
				}
				if algo.Elo.KFactor != 32.0 {
					t.Errorf("k_factor = %v, want 32", algo.Elo.KFactor)
				}
				if !algo.Elo.CategoryWeighted {
					t.Error("expected category_weighted = true")
				}
				if algo.Elo.StoragePath != "/tmp/elo" {
					t.Errorf("storage_path = %q, want /tmp/elo", algo.Elo.StoragePath)
				}
			},
		},
		{
			name:     "router_dc",
			algoType: "router_dc",
			body:     `temperature: 0.8 dimension_size: 128 min_similarity: 0.3 use_query_contrastive: true use_model_contrastive: false`,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.RouterDC == nil {
					t.Fatal("expected router_dc config")
				}
				if algo.RouterDC.Temperature != 0.8 {
					t.Errorf("temperature = %v, want 0.8", algo.RouterDC.Temperature)
				}
				if algo.RouterDC.DimensionSize != 128 {
					t.Errorf("dimension_size = %d, want 128", algo.RouterDC.DimensionSize)
				}
				if !algo.RouterDC.UseQueryContrastive {
					t.Error("expected use_query_contrastive = true")
				}
			},
		},
		{
			name:     "automix",
			algoType: "automix",
			body:     `verification_threshold: 0.9 max_escalations: 3 cost_aware_routing: true`,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.AutoMix == nil {
					t.Fatal("expected automix config")
				}
				if algo.AutoMix.VerificationThreshold != 0.9 {
					t.Errorf("verification_threshold = %v, want 0.9", algo.AutoMix.VerificationThreshold)
				}
				if algo.AutoMix.MaxEscalations != 3 {
					t.Errorf("max_escalations = %d, want 3", algo.AutoMix.MaxEscalations)
				}
				if !algo.AutoMix.CostAwareRouting {
					t.Error("expected cost_aware_routing = true")
				}
			},
		},
		{
			name:     "hybrid",
			algoType: "hybrid",
			body:     `elo_weight: 0.3 router_dc_weight: 0.3 automix_weight: 0.2 cost_weight: 0.2`,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.Hybrid == nil {
					t.Fatal("expected hybrid config")
				}
				if algo.Hybrid.EloWeight != 0.3 {
					t.Errorf("elo_weight = %v, want 0.3", algo.Hybrid.EloWeight)
				}
				if algo.Hybrid.CostWeight != 0.2 {
					t.Errorf("cost_weight = %v, want 0.2", algo.Hybrid.CostWeight)
				}
			},
		},
		{
			name:     "rl_driven",
			algoType: "rl_driven",
			body:     `exploration_rate: 0.1 use_thompson_sampling: true enable_personalization: false`,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.RLDriven == nil {
					t.Fatal("expected rl_driven config")
				}
				if algo.RLDriven.ExplorationRate != 0.1 {
					t.Errorf("exploration_rate = %v, want 0.1", algo.RLDriven.ExplorationRate)
				}
				if !algo.RLDriven.UseThompsonSampling {
					t.Error("expected use_thompson_sampling = true")
				}
			},
		},
		{
			name:     "gmtrouter",
			algoType: "gmtrouter",
			body:     `enable_personalization: true history_sample_size: 50 model_path: "/models/gmt"`,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.GMTRouter == nil {
					t.Fatal("expected gmtrouter config")
				}
				if !algo.GMTRouter.EnablePersonalization {
					t.Error("expected enable_personalization = true")
				}
				if algo.GMTRouter.HistorySampleSize != 50 {
					t.Errorf("history_sample_size = %d, want 50", algo.GMTRouter.HistorySampleSize)
				}
				if algo.GMTRouter.ModelPath != "/models/gmt" {
					t.Errorf("model_path = %q, want /models/gmt", algo.GMTRouter.ModelPath)
				}
			},
		},
		{
			name:     "latency_aware",
			algoType: "latency_aware",
			body:     `tpot_percentile: 95 ttft_percentile: 90`,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.LatencyAware == nil {
					t.Fatal("expected latency_aware config")
				}
				if algo.LatencyAware.TPOTPercentile != 95 {
					t.Errorf("tpot_percentile = %d, want 95", algo.LatencyAware.TPOTPercentile)
				}
				if algo.LatencyAware.TTFTPercentile != 90 {
					t.Errorf("ttft_percentile = %d, want 90", algo.LatencyAware.TTFTPercentile)
				}
			},
		},
		{
			name:     "static",
			algoType: "static",
			body:     ``,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.Type != "static" {
					t.Errorf("type = %q, want static", algo.Type)
				}
			},
		},
		{
			name:     "knn",
			algoType: "knn",
			body:     ``,
			verify: func(t *testing.T, algo *config.AlgorithmConfig) {
				if algo.Type != "knn" {
					t.Errorf("type = %q, want knn", algo.Type)
				}
			},
		},
	}

	for _, tc := range algoDSLs {
		t.Run(tc.name, func(t *testing.T) {
			body := ""
			if tc.body != "" {
				body = fmt.Sprintf("{ %s }", tc.body)
			}
			input := fmt.Sprintf(`
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m1:7b", "m2:3b"
  ALGORITHM %s %s
}`, tc.algoType, body)

			cfg, errs := Compile(input)
			if len(errs) > 0 {
				t.Fatalf("compile errors: %v", errs)
			}
			algo := cfg.Decisions[0].Algorithm
			if algo == nil {
				t.Fatal("expected algorithm")
			}
			if algo.Type != tc.algoType {
				t.Errorf("algo type = %q, want %q", algo.Type, tc.algoType)
			}
			tc.verify(t, algo)
		})
	}
}

// ---------- P1-2: All Plugin Types ----------

func TestCompileAllPluginTypes(t *testing.T) {
	pluginTests := []struct {
		name       string
		pluginType string
		body       string
		verifyType string // expected type after normalization
	}{
		{
			name:       "jailbreak",
			pluginType: "jailbreak",
			body:       `enabled: true threshold: 0.8`,
			verifyType: "jailbreak",
		},
		{
			name:       "hallucination",
			pluginType: "hallucination",
			body:       `enabled: true use_nli: true hallucination_action: "warn"`,
			verifyType: "hallucination",
		},
		{
			name:       "memory",
			pluginType: "memory",
			body:       `enabled: true retrieval_limit: 5 similarity_threshold: 0.7 auto_store: true`,
			verifyType: "memory",
		},
		{
			name:       "rag",
			pluginType: "rag",
			body:       `enabled: true backend: "chromadb" top_k: 10 similarity_threshold: 0.6 injection_mode: "prepend" on_failure: "skip" backend_config: { collection_name: "docs" }`,
			verifyType: "rag",
		},
		{
			name:       "header_mutation",
			pluginType: "header_mutation",
			body:       ``,
			verifyType: "header_mutation",
		},
		{
			name:       "router_replay",
			pluginType: "router_replay",
			body:       `enabled: true`,
			verifyType: "router_replay",
		},
		{
			name:       "image_gen",
			pluginType: "image_gen",
			body:       `enabled: true backend: "dall-e-3"`,
			verifyType: "image_gen",
		},
	}

	for _, tc := range pluginTests {
		t.Run(tc.name, func(t *testing.T) {
			body := "{}"
			if tc.body != "" {
				body = fmt.Sprintf("{ %s }", tc.body)
			}
			input := fmt.Sprintf(`
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b" (reasoning = false)
  PLUGIN %s %s
}`, tc.pluginType, body)

			cfg, errs := Compile(input)
			if len(errs) > 0 {
				t.Fatalf("compile errors: %v", errs)
			}
			if len(cfg.Decisions[0].Plugins) != 1 {
				t.Fatalf("expected 1 plugin, got %d", len(cfg.Decisions[0].Plugins))
			}
			p := cfg.Decisions[0].Plugins[0]
			if p.Type != tc.verifyType {
				t.Errorf("plugin type = %q, want %q", p.Type, tc.verifyType)
			}
		})
	}
}

// ---------- P1-3: All Backend Types ----------

func TestCompileAllBackendTypes(t *testing.T) {
	t.Run("memory_backend", func(t *testing.T) {
		input := `
BACKEND memory mem_store {
  enabled: true
  auto_store: true
  default_retrieval_limit: 20
  default_similarity_threshold: 0.75
}`
		cfg, errs := Compile(input)
		if len(errs) > 0 {
			t.Fatalf("compile errors: %v", errs)
		}
		if !cfg.Memory.Enabled {
			t.Error("expected memory enabled")
		}
		if !cfg.Memory.AutoStore {
			t.Error("expected memory auto_store")
		}
		if cfg.Memory.DefaultRetrievalLimit != 20 {
			t.Errorf("retrieval_limit = %d, want 20", cfg.Memory.DefaultRetrievalLimit)
		}
	})

	t.Run("response_api_backend", func(t *testing.T) {
		input := `
BACKEND response_api resp {
  enabled: true
  store_backend: "redis"
  ttl_seconds: 7200
  max_responses: 500
}`
		cfg, errs := Compile(input)
		if len(errs) > 0 {
			t.Fatalf("compile errors: %v", errs)
		}
		if !cfg.ResponseAPI.Enabled {
			t.Error("expected response_api enabled")
		}
		if cfg.ResponseAPI.StoreBackend != "redis" {
			t.Errorf("store_backend = %q, want redis", cfg.ResponseAPI.StoreBackend)
		}
		if cfg.ResponseAPI.TTLSeconds != 7200 {
			t.Errorf("ttl_seconds = %d, want 7200", cfg.ResponseAPI.TTLSeconds)
		}
		if cfg.ResponseAPI.MaxResponses != 500 {
			t.Errorf("max_responses = %d, want 500", cfg.ResponseAPI.MaxResponses)
		}
	})
}

// ---------- P1-4: EmitCRD Test ----------

func TestEmitCRD(t *testing.T) {
	input := `
SIGNAL domain math { description: "Math" mmlu_categories: ["math"] }
ROUTE math_route {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5:3b" (reasoning = true)
}
GLOBAL { default_model: "qwen2.5:3b" strategy: "priority" }`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	t.Run("with_namespace", func(t *testing.T) {
		crdBytes, err := EmitCRD(cfg, "my-router", "production")
		if err != nil {
			t.Fatalf("EmitCRD error: %v", err)
		}
		crdStr := string(crdBytes)
		if !strings.Contains(crdStr, "apiVersion: vllm.ai/v1alpha1") {
			t.Error("CRD missing apiVersion")
		}
		if !strings.Contains(crdStr, "kind: SemanticRouter") {
			t.Error("CRD missing kind")
		}
		if !strings.Contains(crdStr, "name: my-router") {
			t.Error("CRD missing name")
		}
		if !strings.Contains(crdStr, "namespace: production") {
			t.Error("CRD missing namespace")
		}
		// default_model should be inside spec.config
		if !strings.Contains(crdStr, "default_model: qwen2.5:3b") {
			t.Error("CRD missing spec.config.default_model")
		}
		// decisions should be inside spec.config
		if !strings.Contains(crdStr, "decisions:") {
			t.Error("CRD missing spec.config.decisions")
		}
		// categories (signal rules) should be inside spec.config
		if !strings.Contains(crdStr, "categories:") {
			t.Error("CRD missing spec.config.categories for domain signals")
		}
	})

	t.Run("default_namespace", func(t *testing.T) {
		crdBytes, err := EmitCRD(cfg, "test-router", "")
		if err != nil {
			t.Fatalf("EmitCRD error: %v", err)
		}
		if !strings.Contains(string(crdBytes), "namespace: default") {
			t.Error("expected default namespace")
		}
	})

	t.Run("vllm_endpoints_as_k8s_service", func(t *testing.T) {
		inputWithBackend := `
SIGNAL domain math { description: "Math" }
BACKEND vllm_endpoint vllm_qwen {
  address: "vllm-qwen-svc"
  port: 8000
}
ROUTE test {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5:3b" (reasoning = true)
}
GLOBAL { default_model: "qwen2.5:3b" strategy: "priority" }`

		cfgP, errs := Compile(inputWithBackend)
		if len(errs) > 0 {
			t.Fatalf("compile errors: %v", errs)
		}
		crdBytes, err := EmitCRD(cfgP, "test", "ns")
		if err != nil {
			t.Fatalf("EmitCRD error: %v", err)
		}
		crdStr := string(crdBytes)
		// Should have vllmEndpoints with backend.type: service
		if !strings.Contains(crdStr, "vllmEndpoints:") {
			t.Error("CRD missing spec.vllmEndpoints")
		}
		if !strings.Contains(crdStr, "type: service") {
			t.Error("CRD vllmEndpoints should use backend type: service")
		}
		if !strings.Contains(crdStr, "name: vllm-qwen-svc") {
			t.Error("CRD vllmEndpoints should have service name from address")
		}
		// Should NOT have flat vllm_endpoints or model_config at top level
		if strings.Contains(crdStr, "vllm_endpoints:") {
			t.Error("CRD should not contain flat vllm_endpoints")
		}
		if strings.Contains(crdStr, "model_config:") {
			t.Error("CRD should not contain flat model_config")
		}
	})
}

// ---------- P1-5: Complex Boolean Expression Variants ----------

func TestCompileORExpression(t *testing.T) {
	input := `
SIGNAL domain math { description: "Math" }
SIGNAL domain physics { description: "Physics" }
ROUTE test {
  PRIORITY 1
  WHEN domain("math") OR domain("physics")
  MODEL "m:1b" (reasoning = false)
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	rules := cfg.Decisions[0].Rules
	if rules.Operator != "OR" {
		t.Errorf("expected OR operator, got %q", rules.Operator)
	}
	if len(rules.Conditions) != 2 {
		t.Fatalf("expected 2 conditions, got %d", len(rules.Conditions))
	}
}

func TestCompileNOTExpression(t *testing.T) {
	input := `
SIGNAL domain other { description: "Other" }
ROUTE test {
  PRIORITY 1
  WHEN NOT domain("other")
  MODEL "m:1b" (reasoning = false)
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	rules := cfg.Decisions[0].Rules
	if rules.Operator != "NOT" {
		t.Errorf("expected NOT operator, got %q", rules.Operator)
	}
	if len(rules.Conditions) != 1 {
		t.Fatalf("expected 1 condition, got %d", len(rules.Conditions))
	}
}

func TestCompileDeeplyNestedBoolExpr(t *testing.T) {
	input := `
SIGNAL domain a { description: "A" }
SIGNAL domain b { description: "B" }
SIGNAL domain c { description: "C" }
SIGNAL domain d { description: "D" }
ROUTE test {
  PRIORITY 1
  WHEN (domain("a") OR domain("b")) AND (NOT domain("c") OR domain("d"))
  MODEL "m:1b" (reasoning = false)
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	rules := cfg.Decisions[0].Rules
	if rules.Operator != "AND" {
		t.Errorf("expected top-level AND, got %q", rules.Operator)
	}
	// Left = OR(a, b)
	if rules.Conditions[0].Operator != "OR" {
		t.Errorf("left expected OR, got %q", rules.Conditions[0].Operator)
	}
	// Right = OR(NOT(c), d)
	if rules.Conditions[1].Operator != "OR" {
		t.Errorf("right expected OR, got %q", rules.Conditions[1].Operator)
	}
}

// ---------- P1-6: Global Settings Full Coverage ----------

func TestCompileGlobalFullCoverage(t *testing.T) {
	input := `
GLOBAL {
  default_model: "qwen2.5:3b"
  strategy: "priority"
  default_reasoning_effort: "medium"
  prompt_guard: {
    enabled: true
    threshold: 0.85
  }
  model_selection: {
    enabled: true
    method: "elo"
  }
  looper: {
    endpoint: "http://looper:8080"
    timeout_seconds: 30
  }
  authz: {
    fail_open: true
  }
  ratelimit: {
    fail_open: false
  }
  hallucination_mitigation: {
    enabled: true
    on_hallucination_detected: "warn"
    fact_check_model: { threshold: 0.7 }
    hallucination_model: { threshold: 0.5 }
    nli_model: { threshold: 0.65 }
  }
  reasoning_families: {
    deepseek: { type: "chat_template_kwargs", parameter: "thinking" }
    qwen3: { type: "chat_template_kwargs", parameter: "enable_thinking" }
  }
  observability: {
    metrics: { enabled: true }
    tracing: {
      enabled: true
      provider: "opentelemetry"
      exporter: {
        type: "otlp"
        endpoint: "otel:4317"
        insecure: false
      }
      sampling: { type: "ratio", rate: 0.5 }
      resource: {
        service_name: "my-router"
        service_version: "1.0.0"
        deployment_environment: "prod"
      }
    }
  }
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	if cfg.DefaultModel != "qwen2.5:3b" {
		t.Errorf("default_model = %q", cfg.DefaultModel)
	}
	if cfg.DefaultReasoningEffort != "medium" {
		t.Errorf("default_reasoning_effort = %q", cfg.DefaultReasoningEffort)
	}
	if !cfg.PromptGuard.Enabled {
		t.Error("prompt_guard should be enabled")
	}
	if cfg.PromptGuard.Threshold != 0.85 {
		t.Errorf("prompt_guard threshold = %v, want 0.85", cfg.PromptGuard.Threshold)
	}
	if !cfg.ModelSelection.Enabled {
		t.Error("model_selection should be enabled")
	}
	if cfg.ModelSelection.Method != "elo" {
		t.Errorf("model_selection method = %q, want elo", cfg.ModelSelection.Method)
	}
	if cfg.Looper.Endpoint != "http://looper:8080" {
		t.Errorf("looper endpoint = %q", cfg.Looper.Endpoint)
	}
	if cfg.Looper.TimeoutSeconds != 30 {
		t.Errorf("looper timeout = %d, want 30", cfg.Looper.TimeoutSeconds)
	}
	if !cfg.Authz.FailOpen {
		t.Error("authz fail_open should be true")
	}
	if cfg.RateLimit.FailOpen {
		t.Error("ratelimit fail_open should be false")
	}

	// Hallucination mitigation
	if !cfg.HallucinationMitigation.Enabled {
		t.Error("hallucination_mitigation should be enabled")
	}
	if cfg.HallucinationMitigation.OnHallucinationDetected != "warn" {
		t.Errorf("on_hallucination_detected = %q, want warn", cfg.HallucinationMitigation.OnHallucinationDetected)
	}
	if cfg.HallucinationMitigation.FactCheckModel.Threshold != 0.7 {
		t.Errorf("fact_check threshold = %v, want 0.7", cfg.HallucinationMitigation.FactCheckModel.Threshold)
	}
	if cfg.HallucinationMitigation.HallucinationModel.Threshold != 0.5 {
		t.Errorf("hallucination threshold = %v, want 0.5", cfg.HallucinationMitigation.HallucinationModel.Threshold)
	}
	if cfg.HallucinationMitigation.NLIModel.Threshold != 0.65 {
		t.Errorf("nli threshold = %v, want 0.65", cfg.HallucinationMitigation.NLIModel.Threshold)
	}

	// Reasoning families
	if len(cfg.ReasoningFamilies) != 2 {
		t.Errorf("reasoning_families count = %d, want 2", len(cfg.ReasoningFamilies))
	}
	if ds, ok := cfg.ReasoningFamilies["deepseek"]; !ok {
		t.Error("missing reasoning_family deepseek")
	} else {
		if ds.Type != "chat_template_kwargs" {
			t.Errorf("deepseek type = %q", ds.Type)
		}
		if ds.Parameter != "thinking" {
			t.Errorf("deepseek parameter = %q", ds.Parameter)
		}
	}

	// Observability deep fields
	if cfg.Observability.Tracing.Sampling.Type != "ratio" {
		t.Errorf("sampling type = %q, want ratio", cfg.Observability.Tracing.Sampling.Type)
	}
	if cfg.Observability.Tracing.Sampling.Rate != 0.5 {
		t.Errorf("sampling rate = %v, want 0.5", cfg.Observability.Tracing.Sampling.Rate)
	}
	if cfg.Observability.Tracing.Resource.ServiceName != "my-router" {
		t.Errorf("service_name = %q", cfg.Observability.Tracing.Resource.ServiceName)
	}
	if cfg.Observability.Tracing.Resource.ServiceVersion != "1.0.0" {
		t.Errorf("service_version = %q", cfg.Observability.Tracing.Resource.ServiceVersion)
	}
	if cfg.Observability.Tracing.Resource.DeploymentEnvironment != "prod" {
		t.Errorf("deployment_environment = %q", cfg.Observability.Tracing.Resource.DeploymentEnvironment)
	}
	if cfg.Observability.Tracing.Exporter.Insecure {
		t.Error("exporter insecure should be false")
	}
}

// ---------- P1-7: Lexer Edge Cases ----------

func TestLexNegativeNumber(t *testing.T) {
	input := `port: -1`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	// IDENT COLON INT EOF
	if tokens[2].Type != TOKEN_INT || tokens[2].Literal != "-1" {
		t.Errorf("expected INT -1, got %s %q", tokens[2].Type, tokens[2].Literal)
	}
}

func TestLexIdentWithDotAndDash(t *testing.T) {
	input := `qwen2.5 semantic-cache`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	// qwen2.5 IDENT(semantic-cache) EOF
	if tokens[0].Type != TOKEN_IDENT || tokens[0].Literal != "qwen2.5" {
		t.Errorf("expected IDENT qwen2.5, got %s %q", tokens[0].Type, tokens[0].Literal)
	}
	if tokens[1].Type != TOKEN_IDENT || tokens[1].Literal != "semantic-cache" {
		t.Errorf("expected IDENT semantic-cache, got %s %q", tokens[1].Type, tokens[1].Literal)
	}
}

func TestLexStringEscapeSequences(t *testing.T) {
	input := `"line1\nline2\ttab\\backslash"`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	expected := "line1\nline2\ttab\\backslash"
	if tokens[0].Literal != expected {
		t.Errorf("escape result = %q, want %q", tokens[0].Literal, expected)
	}
}

func TestLexAllPunctuation(t *testing.T) {
	input := `{}()[],:=`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	expected := []TokenType{
		TOKEN_LBRACE, TOKEN_RBRACE, TOKEN_LPAREN, TOKEN_RPAREN,
		TOKEN_LBRACKET, TOKEN_RBRACKET, TOKEN_COMMA, TOKEN_COLON, TOKEN_EQUALS, TOKEN_EOF,
	}
	if len(tokens) != len(expected) {
		t.Fatalf("expected %d tokens, got %d", len(expected), len(tokens))
	}
	for i, exp := range expected {
		if tokens[i].Type != exp {
			t.Errorf("token[%d]: expected %s, got %s", i, exp, tokens[i].Type)
		}
	}
}

func TestLexSignedPositiveNumber(t *testing.T) {
	input := `+42`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}
	if tokens[0].Type != TOKEN_INT || tokens[0].Literal != "+42" {
		t.Errorf("expected INT +42, got %s %q", tokens[0].Type, tokens[0].Literal)
	}
}

func TestLexStandaloneSignIsError(t *testing.T) {
	input := `+ hello`
	_, errs := Lex(input)
	if len(errs) == 0 {
		t.Error("expected lexer error for standalone +")
	}
}

// ---------- P1-8: Parser Edge Cases ----------

func TestParseKeywordsAsIdentifiers(t *testing.T) {
	// Keywords like SIGNAL, ROUTE, etc. should be usable as identifiers in certain contexts
	input := `SIGNAL domain SIGNAL { description: "Using keyword as name" }`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	if len(prog.Signals) != 1 {
		t.Fatalf("expected 1 signal, got %d", len(prog.Signals))
	}
	if prog.Signals[0].Name != "SIGNAL" {
		t.Errorf("signal name = %q, want SIGNAL", prog.Signals[0].Name)
	}
}

func TestParseEmptyArray(t *testing.T) {
	input := `SIGNAL domain test { mmlu_categories: [] }`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	av, ok := prog.Signals[0].Fields["mmlu_categories"].(ArrayValue)
	if !ok {
		t.Fatal("expected ArrayValue")
	}
	if len(av.Items) != 0 {
		t.Errorf("expected empty array, got %d items", len(av.Items))
	}
}

func TestParseNestedObjects(t *testing.T) {
	input := `GLOBAL {
  outer: {
    inner: {
      value: "deep"
    }
  }
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	outer, ok := prog.Global.Fields["outer"].(ObjectValue)
	if !ok {
		t.Fatal("expected outer ObjectValue")
	}
	inner, ok := outer.Fields["inner"].(ObjectValue)
	if !ok {
		t.Fatal("expected inner ObjectValue")
	}
	sv, ok := inner.Fields["value"].(StringValue)
	if !ok || sv.V != "deep" {
		t.Errorf("expected deep, got %v", inner.Fields["value"])
	}
}

func TestParseBareIdentAsValue(t *testing.T) {
	// Bare identifiers should be accepted as string values
	input := `SIGNAL keyword test { method: regex }`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	sv, ok := prog.Signals[0].Fields["method"].(StringValue)
	if !ok || sv.V != "regex" {
		t.Errorf("expected string 'regex', got %v", prog.Signals[0].Fields["method"])
	}
}

func TestParseModelWithLoRA(t *testing.T) {
	input := `ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "qwen2.5:3b" (reasoning = true, effort = "high", lora = "math-adapter")
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	m := prog.Routes[0].Models[0]
	if m.LoRA != "math-adapter" {
		t.Errorf("lora = %q, want math-adapter", m.LoRA)
	}
}

func TestParseModelNoOptions(t *testing.T) {
	input := `ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "qwen2.5:3b"
}`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	m := prog.Routes[0].Models[0]
	if m.Model != "qwen2.5:3b" {
		t.Errorf("model = %q, want qwen2.5:3b", m.Model)
	}
	if m.Reasoning != nil {
		t.Error("reasoning should be nil when not specified")
	}
	if m.Effort != "" {
		t.Error("effort should be empty when not specified")
	}
}

// ---------- P1-9: CompileAST Direct Usage ----------

func TestCompileASTDirect(t *testing.T) {
	prog := &Program{
		Signals: []*SignalDecl{
			{
				SignalType: "domain",
				Name:       "math",
				Fields: map[string]Value{
					"description":     StringValue{V: "Math"},
					"mmlu_categories": ArrayValue{Items: []Value{StringValue{V: "math"}}},
				},
			},
		},
		Routes: []*RouteDecl{
			{
				Name:     "test_route",
				Priority: 42,
				When:     &SignalRefExpr{SignalType: "domain", SignalName: "math"},
				Models: []*ModelRef{
					{Model: "test:1b"},
				},
			},
		},
	}

	cfg, errs := CompileAST(prog)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.Categories) != 1 || cfg.Categories[0].Name != "math" {
		t.Errorf("categories = %v", cfg.Categories)
	}
	if len(cfg.Decisions) != 1 || cfg.Decisions[0].Priority != 42 {
		t.Errorf("decisions = %v", cfg.Decisions)
	}
}

// ---------- P1-10: Provider Profile Full Fields ----------

func TestCompileProviderProfileFullFields(t *testing.T) {
	input := `
BACKEND provider_profile azure_prod {
  type: "azure"
  base_url: "https://my-azure.openai.azure.com"
  auth_header: "api-key"
  auth_prefix: ""
  api_version: "2024-02-01"
  chat_path: "/openai/deployments/gpt-4/chat/completions"
  extra_headers: {
    x-custom: "value1"
    x-trace: "trace123"
  }
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	pp, ok := cfg.ProviderProfiles["azure_prod"]
	if !ok {
		t.Fatal("azure_prod provider profile not found")
	}
	if pp.Type != "azure" {
		t.Errorf("type = %q, want azure", pp.Type)
	}
	if pp.AuthHeader != "api-key" {
		t.Errorf("auth_header = %q", pp.AuthHeader)
	}
	if pp.APIVersion != "2024-02-01" {
		t.Errorf("api_version = %q", pp.APIVersion)
	}
	if len(pp.ExtraHeaders) != 2 {
		t.Errorf("extra_headers count = %d, want 2", len(pp.ExtraHeaders))
	}
	if pp.ExtraHeaders["x-custom"] != "value1" {
		t.Errorf("x-custom = %q", pp.ExtraHeaders["x-custom"])
	}
}

// ---------- P1-11: Embedding Model with HNSW Config ----------

func TestCompileEmbeddingModelHNSW(t *testing.T) {
	input := `
BACKEND embedding_model ultra {
  mmbert_model_path: "models/mmbert-ultra"
  bert_model_path: "models/bert-base"
  use_cpu: true
  hnsw_config: {
    model_type: "mmbert"
    preload_embeddings: true
    target_dimension: 768
    min_score_threshold: 0.5
    enable_soft_matching: true
  }
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if cfg.EmbeddingModels.MmBertModelPath != "models/mmbert-ultra" {
		t.Errorf("mmbert path = %q", cfg.EmbeddingModels.MmBertModelPath)
	}
	if cfg.EmbeddingModels.BertModelPath != "models/bert-base" {
		t.Errorf("bert path = %q", cfg.EmbeddingModels.BertModelPath)
	}
	if !cfg.EmbeddingModels.UseCPU {
		t.Error("expected use_cpu = true")
	}
	hnsw := cfg.EmbeddingModels.HNSWConfig
	if hnsw.ModelType != "mmbert" {
		t.Errorf("hnsw model_type = %q", hnsw.ModelType)
	}
	if !hnsw.PreloadEmbeddings {
		t.Error("expected preload_embeddings = true")
	}
	if hnsw.TargetDimension != 768 {
		t.Errorf("target_dimension = %d", hnsw.TargetDimension)
	}
	if hnsw.MinScoreThreshold != 0.5 {
		t.Errorf("min_score_threshold = %v", hnsw.MinScoreThreshold)
	}
	if hnsw.EnableSoftMatching == nil || !*hnsw.EnableSoftMatching {
		t.Error("expected enable_soft_matching = true")
	}
}

// ---------- P1-12: Semantic Cache Backend ----------

func TestCompileSemanticCacheBackendFull(t *testing.T) {
	input := `
BACKEND semantic_cache main {
  enabled: true
  backend_type: "redis"
  similarity_threshold: 0.85
  max_entries: 5000
  ttl_seconds: 1800
  eviction_policy: "lru"
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if !cfg.SemanticCache.Enabled {
		t.Error("expected enabled")
	}
	if cfg.SemanticCache.BackendType != "redis" {
		t.Errorf("backend_type = %q", cfg.SemanticCache.BackendType)
	}
	if cfg.SemanticCache.MaxEntries != 5000 {
		t.Errorf("max_entries = %d", cfg.SemanticCache.MaxEntries)
	}
	if cfg.SemanticCache.TTLSeconds != 1800 {
		t.Errorf("ttl_seconds = %d", cfg.SemanticCache.TTLSeconds)
	}
	if cfg.SemanticCache.EvictionPolicy != "lru" {
		t.Errorf("eviction_policy = %q", cfg.SemanticCache.EvictionPolicy)
	}
}

// ---------- P1-13: Keyword Signal Full Fields ----------

func TestCompileKeywordSignalFullFields(t *testing.T) {
	input := `
SIGNAL keyword advanced_kw {
  operator: "all"
  keywords: ["hello", "world"]
  case_sensitive: true
  method: "bm25"
  fuzzy_match: true
  fuzzy_threshold: 3
  bm25_threshold: 0.7
  ngram_threshold: 0.8
  ngram_arity: 4
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.KeywordRules) != 1 {
		t.Fatalf("expected 1 keyword rule")
	}
	kw := cfg.KeywordRules[0]
	if kw.Operator != "all" {
		t.Errorf("operator = %q", kw.Operator)
	}
	if !kw.CaseSensitive {
		t.Error("expected case_sensitive = true")
	}
	if kw.Method != "bm25" {
		t.Errorf("method = %q", kw.Method)
	}
	if !kw.FuzzyMatch {
		t.Error("expected fuzzy_match = true")
	}
	if kw.FuzzyThreshold != 3 {
		t.Errorf("fuzzy_threshold = %d", kw.FuzzyThreshold)
	}
	if kw.BM25Threshold != 0.7 {
		t.Errorf("bm25_threshold = %v", kw.BM25Threshold)
	}
	if kw.NgramThreshold != 0.8 {
		t.Errorf("ngram_threshold = %v", kw.NgramThreshold)
	}
	if kw.NgramArity != 4 {
		t.Errorf("ngram_arity = %d", kw.NgramArity)
	}
}

// ---------- P1-14: VLLM Endpoint Full Fields ----------

func TestCompileVLLMEndpointFullFields(t *testing.T) {
	input := `
BACKEND vllm_endpoint prod {
  address: "10.0.0.1"
  port: 8000
  weight: 3
  type: "vllm"
  api_key: "sk-secret"
  provider_profile: "openai_prod"
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	ep := cfg.VLLMEndpoints[0]
	if ep.Name != "prod" {
		t.Errorf("name = %q", ep.Name)
	}
	if ep.Weight != 3 {
		t.Errorf("weight = %d", ep.Weight)
	}
	if ep.Type != "vllm" {
		t.Errorf("type = %q", ep.Type)
	}
	if ep.APIKey != "sk-secret" {
		t.Errorf("api_key = %q", ep.APIKey)
	}
	if ep.ProviderProfileName != "openai_prod" {
		t.Errorf("provider_profile = %q", ep.ProviderProfileName)
	}
}

// ---------- P1-15: Confidence Algorithm Full Fields ----------

func TestCompileConfidenceAlgoFullFields(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m1:7b", "m2:3b"
  ALGORITHM confidence {
    confidence_method: "logprob"
    threshold: 0.8
    on_error: "fallback"
    escalation_order: "asc"
    cost_quality_tradeoff: 0.7
    hybrid_weights: { logprob_weight: 0.5, margin_weight: 0.5 }
  }
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	c := cfg.Decisions[0].Algorithm.Confidence
	if c.ConfidenceMethod != "logprob" {
		t.Errorf("method = %q", c.ConfidenceMethod)
	}
	if c.OnError != "fallback" {
		t.Errorf("on_error = %q", c.OnError)
	}
	if c.EscalationOrder != "asc" {
		t.Errorf("escalation_order = %q", c.EscalationOrder)
	}
	if c.CostQualityTradeoff != 0.7 {
		t.Errorf("cost_quality_tradeoff = %v", c.CostQualityTradeoff)
	}
}

// ---------- P1-16: ReMoM Algorithm Full Fields ----------

func TestCompileReMoMAlgoFullFields(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m1:7b", "m2:3b"
  ALGORITHM remom {
    breadth_schedule: [8, 4, 2]
    model_distribution: "uniform"
    temperature: 0.8
    include_reasoning: true
    compaction_strategy: "summarize"
    compaction_tokens: 512
    synthesis_template: "custom-template"
    max_concurrent: 8
    on_error: "skip"
    include_intermediate_responses: true
  }
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	r := cfg.Decisions[0].Algorithm.ReMoM
	if len(r.BreadthSchedule) != 3 || r.BreadthSchedule[2] != 2 {
		t.Errorf("breadth_schedule = %v", r.BreadthSchedule)
	}
	if r.ModelDistribution != "uniform" {
		t.Errorf("model_distribution = %q", r.ModelDistribution)
	}
	if r.CompactionStrategy != "summarize" {
		t.Errorf("compaction_strategy = %q", r.CompactionStrategy)
	}
	if r.CompactionTokens != 512 {
		t.Errorf("compaction_tokens = %d", r.CompactionTokens)
	}
	if r.SynthesisTemplate != "custom-template" {
		t.Errorf("synthesis_template = %q", r.SynthesisTemplate)
	}
	if r.MaxConcurrent != 8 {
		t.Errorf("max_concurrent = %d", r.MaxConcurrent)
	}
	if !r.IncludeIntermediateResponses {
		t.Error("expected include_intermediate_responses = true")
	}
}

// ==================== P2 Tests ====================
// Stress, idempotency, regression, and large-scale tests.

// ---------- P2-1: Compile Idempotency ----------
// Same input compiled twice should produce identical configs.

func TestCompileIdempotency(t *testing.T) {
	cfg1, errs1 := Compile(fullDSLExample)
	if len(errs1) > 0 {
		t.Fatalf("first compile errors: %v", errs1)
	}
	cfg2, errs2 := Compile(fullDSLExample)
	if len(errs2) > 0 {
		t.Fatalf("second compile errors: %v", errs2)
	}

	yaml1, err1 := EmitYAMLFromConfig(cfg1)
	if err1 != nil {
		t.Fatalf("emit1 error: %v", err1)
	}
	yaml2, err2 := EmitYAMLFromConfig(cfg2)
	if err2 != nil {
		t.Fatalf("emit2 error: %v", err2)
	}

	if !bytes.Equal(yaml1, yaml2) {
		t.Error("compile is not idempotent: two identical inputs produced different YAML")
	}
}

// ---------- P2-2: Double Round-Trip ----------
// DSL → YAML → Unmarshal → Marshal → Unmarshal and compare.

func TestDoubleRoundTrip(t *testing.T) {
	cfg, errs := Compile(fullDSLExample)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	// First round-trip
	yaml1, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("emit1 error: %v", err)
	}
	var rt1 config.RouterConfig
	if unmarshalErr := yaml.Unmarshal(yaml1, &rt1); unmarshalErr != nil {
		t.Fatalf("unmarshal1 error: %v", unmarshalErr)
	}

	// Second round-trip: marshal rt1 back to YAML using yaml.v2
	yaml2, err := yaml.Marshal(&rt1)
	if err != nil {
		t.Fatalf("marshal2 error: %v", err)
	}
	var rt2 config.RouterConfig
	if err := yaml.Unmarshal(yaml2, &rt2); err != nil {
		t.Fatalf("unmarshal2 error: %v", err)
	}

	// Compare key fields
	if rt1.DefaultModel != rt2.DefaultModel {
		t.Errorf("default_model: %q vs %q", rt1.DefaultModel, rt2.DefaultModel)
	}
	if rt1.Strategy != rt2.Strategy {
		t.Errorf("strategy: %q vs %q", rt1.Strategy, rt2.Strategy)
	}
	if len(rt1.Categories) != len(rt2.Categories) {
		t.Errorf("categories count: %d vs %d", len(rt1.Categories), len(rt2.Categories))
	}
	if len(rt1.Decisions) != len(rt2.Decisions) {
		t.Errorf("decisions count: %d vs %d", len(rt1.Decisions), len(rt2.Decisions))
	}
	if len(rt1.VLLMEndpoints) != len(rt2.VLLMEndpoints) {
		t.Errorf("endpoints count: %d vs %d", len(rt1.VLLMEndpoints), len(rt2.VLLMEndpoints))
	}
}

// ---------- P2-3: Large-Scale Input ----------
// Stress test with many signals and routes.

func TestLargeScaleInput(t *testing.T) {
	var sb strings.Builder
	numSignals := 50
	numRoutes := 50

	for i := 0; i < numSignals; i++ {
		fmt.Fprintf(&sb, "SIGNAL domain domain_%d { description: \"Domain %d\" mmlu_categories: [\"cat_%d\"] }\n", i, i, i)
	}
	for i := 0; i < numRoutes; i++ {
		fmt.Fprintf(&sb, `ROUTE route_%d {
  PRIORITY %d
  WHEN domain("domain_%d")
  MODEL "model:1b" (reasoning = false)
}
`, i, i+1, i%numSignals)
	}
	sb.WriteString(`GLOBAL { default_model: "model:1b" strategy: "priority" }`)

	cfg, errs := Compile(sb.String())
	if len(errs) > 0 {
		t.Fatalf("compile errors with large input: %v", errs)
	}
	if len(cfg.Categories) != numSignals {
		t.Errorf("expected %d categories, got %d", numSignals, len(cfg.Categories))
	}
	if len(cfg.Decisions) != numRoutes {
		t.Errorf("expected %d decisions, got %d", numRoutes, len(cfg.Decisions))
	}

	// Verify YAML emission works
	yamlBytes, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("emit YAML error: %v", err)
	}
	if len(yamlBytes) == 0 {
		t.Error("YAML output is empty")
	}
}

// ---------- P2-4: All Signal Types YAML Round-Trip ----------

func TestAllSignalTypesRoundTrip(t *testing.T) {
	input := `
SIGNAL keyword kw { operator: "any" keywords: ["test"] case_sensitive: true method: "exact" }
SIGNAL embedding emb { threshold: 0.75 candidates: ["test"] aggregation_method: "max" }
SIGNAL domain dom { description: "test" mmlu_categories: ["math"] }
SIGNAL fact_check fc { description: "fact check" }
SIGNAL user_feedback uf { description: "feedback" }
SIGNAL preference pref { description: "preference" }
SIGNAL language lang { description: "English" }
SIGNAL context ctx { min_tokens: "1K" max_tokens: "32K" }
SIGNAL complexity comp { threshold: 0.1 hard: { candidates: ["hard"] } easy: { candidates: ["easy"] } }
SIGNAL modality mod { description: "image" }
SIGNAL authz auth { role: "admin" subjects: [{ kind: "User", name: "admin" }] }

ROUTE test_route {
  PRIORITY 1
  WHEN domain("dom")
  MODEL "m:1b" (reasoning = false)
}

GLOBAL { default_model: "m:1b" strategy: "priority" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	yamlBytes, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("emit error: %v", err)
	}

	var rt config.RouterConfig
	if err := yaml.Unmarshal(yamlBytes, &rt); err != nil {
		t.Fatalf("unmarshal error: %v\nYAML:\n%s", err, string(yamlBytes))
	}

	if len(rt.KeywordRules) != 1 {
		t.Errorf("keyword rules: %d", len(rt.KeywordRules))
	}
	if len(rt.EmbeddingRules) != 1 {
		t.Errorf("embedding rules: %d", len(rt.EmbeddingRules))
	}
	if len(rt.Categories) != 1 {
		t.Errorf("categories: %d", len(rt.Categories))
	}
	if len(rt.FactCheckRules) != 1 {
		t.Errorf("fact_check rules: %d", len(rt.FactCheckRules))
	}
	if len(rt.UserFeedbackRules) != 1 {
		t.Errorf("user_feedback rules: %d", len(rt.UserFeedbackRules))
	}
	if len(rt.PreferenceRules) != 1 {
		t.Errorf("preference rules: %d", len(rt.PreferenceRules))
	}
	if len(rt.LanguageRules) != 1 {
		t.Errorf("language rules: %d", len(rt.LanguageRules))
	}
	if len(rt.ContextRules) != 1 {
		t.Errorf("context rules: %d", len(rt.ContextRules))
	}
	if len(rt.ComplexityRules) != 1 {
		t.Errorf("complexity rules: %d", len(rt.ComplexityRules))
	}
	if len(rt.ModalityRules) != 1 {
		t.Errorf("modality rules: %d", len(rt.ModalityRules))
	}
	if len(rt.RoleBindings) != 1 {
		t.Errorf("role bindings: %d", len(rt.RoleBindings))
	}
}

// ---------- P2-5: Multiple Routes Same Priority ----------

func TestMultipleRoutesSamePriority(t *testing.T) {
	input := `
SIGNAL domain math { description: "Math" }
SIGNAL domain physics { description: "Physics" }
SIGNAL domain bio { description: "Bio" }

ROUTE route_a { PRIORITY 100 WHEN domain("math") MODEL "m:1b" (reasoning = false) }
ROUTE route_b { PRIORITY 100 WHEN domain("physics") MODEL "m:1b" (reasoning = false) }
ROUTE route_c { PRIORITY 50 WHEN domain("bio") MODEL "m:1b" (reasoning = false) }

GLOBAL { default_model: "m:1b" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.Decisions) != 3 {
		t.Fatalf("expected 3 decisions, got %d", len(cfg.Decisions))
	}

	// Verify order is preserved
	if cfg.Decisions[0].Name != "route_a" {
		t.Errorf("first decision = %q, want route_a", cfg.Decisions[0].Name)
	}
	if cfg.Decisions[1].Name != "route_b" {
		t.Errorf("second decision = %q, want route_b", cfg.Decisions[1].Name)
	}
	// Both should have priority 100
	if cfg.Decisions[0].Priority != 100 || cfg.Decisions[1].Priority != 100 {
		t.Error("route_a and route_b should both have priority 100")
	}
}

// ---------- P2-6: Whitespace and Comment Variants ----------

func TestWhitespaceAndCommentVariants(t *testing.T) {
	// Minimal whitespace
	compact := `SIGNAL domain m{description:"Math"}ROUTE r{PRIORITY 1 WHEN domain("m")MODEL "m:1b"}GLOBAL{default_model:"m:1b"}`

	cfg1, errs1 := Compile(compact)
	if len(errs1) > 0 {
		t.Fatalf("compact compile errors: %v", errs1)
	}
	if len(cfg1.Categories) != 1 || len(cfg1.Decisions) != 1 {
		t.Error("compact input failed to parse correctly")
	}

	// Heavy comments
	commented := `
# Top level comment
SIGNAL domain m { # inline comment after brace
  description: "Math" # field comment
  # standalone comment
}
# Between declarations
ROUTE r {
  PRIORITY 1
  WHEN domain("m") # after WHEN
  MODEL "m:1b" # after MODEL
}
# Final comment
GLOBAL { default_model: "m:1b" }
`
	cfg2, errs2 := Compile(commented)
	if len(errs2) > 0 {
		t.Fatalf("commented compile errors: %v", errs2)
	}
	if len(cfg2.Categories) != 1 || len(cfg2.Decisions) != 1 {
		t.Error("commented input failed to parse correctly")
	}
}

// ---------- P2-7: Plugin Template Merge Semantics ----------

func TestPluginTemplateMergeSemantics(t *testing.T) {
	input := `
PLUGIN my_cache semantic_cache {
  enabled: true
  similarity_threshold: 0.80
}

SIGNAL domain test { description: "test" }

ROUTE route_a {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b" (reasoning = false)
  PLUGIN my_cache
}

ROUTE route_b {
  PRIORITY 2
  WHEN domain("test")
  MODEL "m:1b" (reasoning = false)
  PLUGIN my_cache {
    similarity_threshold: 0.95
  }
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	// route_a: uses template as-is
	pA := cfg.Decisions[0].Plugins[0]
	if pA.Type != "semantic-cache" {
		t.Errorf("route_a plugin type = %q, want semantic-cache", pA.Type)
	}

	// route_b: uses template with override
	pB := cfg.Decisions[1].Plugins[0]
	if pB.Type != "semantic-cache" {
		t.Errorf("route_b plugin type = %q, want semantic-cache", pB.Type)
	}
}

// ---------- P2-8: Full Example YAML Round-Trip ----------

func TestFullExampleYAMLRoundTrip(t *testing.T) {
	cfg, errs := Compile(fullDSLExample)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	yamlBytes, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("emit YAML error: %v", err)
	}

	var rt config.RouterConfig
	if err := yaml.Unmarshal(yamlBytes, &rt); err != nil {
		t.Fatalf("unmarshal error: %v\nYAML:\n%s", err, string(yamlBytes))
	}

	// Comprehensive field check
	if rt.DefaultModel != "qwen2.5:3b" {
		t.Errorf("default_model = %q", rt.DefaultModel)
	}
	if rt.Strategy != "priority" {
		t.Errorf("strategy = %q", rt.Strategy)
	}
	if rt.DefaultReasoningEffort != "low" {
		t.Errorf("default_reasoning_effort = %q", rt.DefaultReasoningEffort)
	}
	if len(rt.Categories) != 3 {
		t.Errorf("categories = %d, want 3", len(rt.Categories))
	}
	if len(rt.EmbeddingRules) != 1 {
		t.Errorf("embedding_rules = %d", len(rt.EmbeddingRules))
	}
	if len(rt.KeywordRules) != 1 {
		t.Errorf("keyword_rules = %d", len(rt.KeywordRules))
	}
	if len(rt.Decisions) != 4 {
		t.Errorf("decisions = %d, want 4", len(rt.Decisions))
	}
	if len(rt.VLLMEndpoints) != 1 {
		t.Errorf("vllm_endpoints = %d", len(rt.VLLMEndpoints))
	}
	if !rt.Observability.Tracing.Enabled {
		t.Error("tracing should be enabled")
	}
	if !rt.PromptGuard.Enabled {
		t.Error("prompt_guard should be enabled")
	}

	// Check urgent_ai_route specifically
	for _, d := range rt.Decisions {
		if d.Name == "urgent_ai_route" {
			if d.Priority != 200 {
				t.Errorf("urgent priority = %d", d.Priority)
			}
			if len(d.ModelRefs) != 2 {
				t.Errorf("urgent model_refs = %d", len(d.ModelRefs))
			}
			if d.Algorithm == nil || d.Algorithm.Type != "confidence" {
				t.Error("urgent should have confidence algorithm")
			}
			if len(d.Plugins) != 2 {
				t.Errorf("urgent plugins = %d, want 2", len(d.Plugins))
			}
			break
		}
	}
}

// ---------- P2-9: Token Position Tracking Across Multiple Lines ----------

func TestTokenPositionMultiLine(t *testing.T) {
	input := `SIGNAL domain math {
  description: "Math"
  mmlu_categories: ["math"]
}
ROUTE math_route {
  PRIORITY 100
}`
	tokens, errs := Lex(input)
	if len(errs) > 0 {
		t.Fatalf("lex errors: %v", errs)
	}

	// SIGNAL should be at line 1
	if tokens[0].Pos.Line != 1 {
		t.Errorf("SIGNAL at line %d, want 1", tokens[0].Pos.Line)
	}
	// "description" should be at line 2
	var descToken *Token
	for i := range tokens {
		if tokens[i].Literal == "description" {
			descToken = &tokens[i]
			break
		}
	}
	if descToken == nil {
		t.Fatal("description token not found")
	}
	if descToken.Pos.Line != 2 {
		t.Errorf("description at line %d, want 2", descToken.Pos.Line)
	}

	// ROUTE should be at line 5
	var routeToken *Token
	for i := range tokens {
		if tokens[i].Type == TOKEN_ROUTE {
			routeToken = &tokens[i]
			break
		}
	}
	if routeToken == nil {
		t.Fatal("ROUTE token not found")
	}
	if routeToken.Pos.Line != 5 {
		t.Errorf("ROUTE at line %d, want 5", routeToken.Pos.Line)
	}
}

// ---------- P2-10: Error Recovery Across Multiple Blocks ----------

func TestParseErrorRecoveryMultipleBlocks(t *testing.T) {
	input := `
SIGNAL domain valid1 { description: "OK" }
SIGNAL domain broken1 { description "missing colon" }
SIGNAL domain valid2 { description: "Also OK" }
ROUTE valid_route {
  PRIORITY 1
  WHEN domain("valid1")
  MODEL "m:1b"
}
`
	prog, errs := Parse(input)
	if len(errs) == 0 {
		t.Fatal("expected parse errors for broken signal")
	}
	if prog == nil {
		t.Fatal("expected non-nil program even with errors")
	}
	// Should recover and parse the valid route
	if len(prog.Routes) != 1 {
		t.Errorf("expected 1 route after recovery, got %d", len(prog.Routes))
	}
}

// ---------- P2-11: Fuzz-like Random Inputs ----------

func TestFuzzLikeInputsDoNotPanic(t *testing.T) {
	inputs := []string{
		// Deeply nested braces
		`SIGNAL domain a { nested: { nested: { nested: { nested: { value: "deep" } } } } }`,
		// Very long string
		`SIGNAL domain a { description: "` + strings.Repeat("x", 10000) + `" }`,
		// Many commas
		`SIGNAL keyword a { keywords: ["a", "b", "c", "d", "e", "f", "g", "h"] }`,
		// Mixed valid and invalid
		`SIGNAL domain a {} ROUTE {} GLOBAL {}`,
		// Only whitespace and comments
		`   # comment only   `,
		// Nested arrays (unusual but shouldn't crash)
		`SIGNAL domain a { data: [["inner"]] }`,
		// Empty field block
		`SIGNAL domain a {}`,
		// Route with all optional fields missing
		`ROUTE bare_route { WHEN domain("x") MODEL "m:1b" }`,
		// Unicode in strings
		`SIGNAL domain uni { description: "日本語テスト" }`,
		// Very long identifier
		`SIGNAL domain ` + strings.Repeat("a", 1000) + ` { description: "test" }`,
	}

	for i, input := range inputs {
		t.Run(fmt.Sprintf("fuzz_%d", i), func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("panicked on fuzz input %d: %v", i, r)
				}
			}()
			_, _ = Compile(input)
		})
	}
}

// ---------- P2-12: Route Description Preserved ----------

func TestRouteDescriptionPreserved(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE my_route (description = "This is a detailed description") {
  PRIORITY 42
  WHEN domain("test")
  MODEL "m:1b" (reasoning = false)
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if cfg.Decisions[0].Description != "This is a detailed description" {
		t.Errorf("description = %q", cfg.Decisions[0].Description)
	}

	// Verify survives YAML round-trip
	yamlBytes, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("emit error: %v", err)
	}
	var rt config.RouterConfig
	if err := yaml.Unmarshal(yamlBytes, &rt); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if rt.Decisions[0].Description != "This is a detailed description" {
		t.Errorf("round-trip description = %q", rt.Decisions[0].Description)
	}
}

// ---------- P2-13: Model LoRA Compile and Round-Trip ----------

func TestModelLoRACompileAndRoundTrip(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "qwen2.5:3b" (reasoning = true, effort = "high", lora = "math-lora-v2")
}
GLOBAL { default_model: "qwen2.5:3b" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if cfg.Decisions[0].ModelRefs[0].LoRAName != "math-lora-v2" {
		t.Errorf("lora_name = %q", cfg.Decisions[0].ModelRefs[0].LoRAName)
	}

	yamlBytes, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("emit error: %v", err)
	}
	var rt config.RouterConfig
	if err := yaml.Unmarshal(yamlBytes, &rt); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if rt.Decisions[0].ModelRefs[0].LoRAName != "math-lora-v2" {
		t.Errorf("round-trip lora_name = %q", rt.Decisions[0].ModelRefs[0].LoRAName)
	}
}

// ---------- P2-14: No GLOBAL Block ----------

func TestCompileWithoutGlobal(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b" (reasoning = false)
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	// Should compile fine with empty global settings
	if cfg.DefaultModel != "" {
		t.Errorf("expected empty default_model, got %q", cfg.DefaultModel)
	}
	if cfg.Strategy != "" {
		t.Errorf("expected empty strategy, got %q", cfg.Strategy)
	}
}

// ---------- P2-15: Algorithm on_error at Top Level ----------

func TestAlgorithmOnErrorTopLevel(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m1:7b", "m2:3b"
  ALGORITHM confidence {
    confidence_method: "hybrid"
    threshold: 0.5
    on_error: "fallback"
  }
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	algo := cfg.Decisions[0].Algorithm
	// For confidence/ratings/remom, on_error goes into the sub-config, not the top level
	if algo.OnError != "" {
		t.Errorf("algo top-level on_error = %q, want empty", algo.OnError)
	}
	if algo.Confidence == nil || algo.Confidence.OnError != "fallback" {
		t.Errorf("algo.Confidence.OnError = %q, want fallback", algo.Confidence.OnError)
	}
}

// ==================== Step 5: Validator Tests ====================

func TestValidateCleanInput(t *testing.T) {
	diags, errs := Validate(fullDSLExample)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	// Full valid input should have no errors or warnings
	for _, d := range diags {
		if d.Level == DiagError {
			t.Errorf("unexpected error diagnostic: %s", d)
		}
	}
}

func TestValidateUndefinedSignalRef(t *testing.T) {
	input := `
SIGNAL domain math { description: "Math" }
ROUTE test {
  PRIORITY 1
  WHEN domain("nonexistent")
  MODEL "m:1b"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagWarning && strings.Contains(d.Message, "nonexistent") && strings.Contains(d.Message, "not defined") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected warning for undefined signal reference 'nonexistent'")
	}
}

func TestValidateUndefinedSignalSuggestion(t *testing.T) {
	input := `
SIGNAL domain math { description: "Math" }
ROUTE test {
  PRIORITY 1
  WHEN domain("mth")
  MODEL "m:1b"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagWarning && d.Fix != nil && d.Fix.NewText == "math" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected 'Did you mean math?' suggestion for typo 'mth'")
	}
}

func TestValidateUndefinedPluginRef(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN nonexistent_plugin
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagWarning && strings.Contains(d.Message, "nonexistent_plugin") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected warning for undefined plugin reference")
	}
}

func TestValidateInlinePluginNoWarning(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN fast_response { message: "blocked" }
}
`
	diags, _ := Validate(input)
	for _, d := range diags {
		if d.Level == DiagWarning && strings.Contains(d.Message, "fast_response") {
			t.Error("should not warn about recognized inline plugin type 'fast_response'")
		}
	}
}

func TestValidateThresholdOutOfRange(t *testing.T) {
	input := `
SIGNAL embedding test { threshold: 1.5 candidates: ["test"] }
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagConstraint && strings.Contains(d.Message, "threshold") && strings.Contains(d.Message, "<= 1") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected constraint violation for threshold > 1.0")
	}
}

func TestValidatePortOutOfRange(t *testing.T) {
	input := `
BACKEND vllm_endpoint test { address: "127.0.0.1" port: 99999 }
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagConstraint && strings.Contains(d.Message, "port") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected constraint violation for port > 65535")
	}
}

func TestValidateNegativePriority(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY -1
  WHEN domain("test")
  MODEL "m:1b"
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagConstraint && strings.Contains(d.Message, "priority") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected constraint violation for negative priority")
	}
}

func TestValidateUnknownAlgorithmType(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b"
  ALGORITHM confdence { threshold: 0.5 }
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagConstraint && strings.Contains(d.Message, "confdence") {
			found = true
			// Good — got suggestion if d.Fix != nil && d.Fix.NewText == "confidence"
			break
		}
	}
	if !found {
		t.Error("expected constraint violation for unknown algorithm type 'confdence'")
	}
}

func TestValidateUnknownSignalType(t *testing.T) {
	input := `
SIGNAL unknown_type test { description: "test" }
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagConstraint && strings.Contains(d.Message, "unknown_type") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected constraint violation for unknown signal type")
	}
}

func TestValidateRouteNoModel(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagWarning && strings.Contains(d.Message, "no MODEL") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected warning for route without MODEL")
	}
}

func TestValidateASTDirect(t *testing.T) {
	prog := &Program{
		Signals: []*SignalDecl{
			{SignalType: "domain", Name: "math", Fields: map[string]Value{"description": StringValue{V: "Math"}}},
		},
		Routes: []*RouteDecl{
			{
				Name:     "test",
				Priority: 10,
				When:     &SignalRefExpr{SignalType: "domain", SignalName: "nonexistent"},
				Models:   []*ModelRef{{Model: "m:1b"}},
			},
		},
	}
	diags := ValidateAST(prog)
	found := false
	for _, d := range diags {
		if d.Level == DiagWarning && strings.Contains(d.Message, "nonexistent") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected warning from ValidateAST for undefined signal")
	}
}

func TestValidateNestedConstraints(t *testing.T) {
	input := `
GLOBAL {
  prompt_guard: {
    enabled: true
    threshold: 1.5
  }
}
`
	diags, _ := Validate(input)
	found := false
	for _, d := range diags {
		if d.Level == DiagConstraint && strings.Contains(d.Message, "threshold") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected constraint violation for nested prompt_guard threshold")
	}
}

func TestValidateSyntaxError(t *testing.T) {
	input := `SIGNAL domain test description: "missing braces"`
	diags, errs := Validate(input)
	if len(errs) == 0 && len(diags) == 0 {
		t.Error("expected at least one error for syntax issue")
	}
	foundError := false
	for _, d := range diags {
		if d.Level == DiagError {
			foundError = true
			break
		}
	}
	if !foundError && len(errs) == 0 {
		t.Error("expected Level 1 error diagnostic")
	}
}

// ==================== Step 6: Decompiler Tests ====================

func TestDecompileBasic(t *testing.T) {
	cfg, errs := Compile(fullDSLExample)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}

	// Should contain section headers
	if !strings.Contains(dslText, "# SIGNALS") {
		t.Error("missing SIGNALS section")
	}
	if !strings.Contains(dslText, "# ROUTES") {
		t.Error("missing ROUTES section")
	}
	if !strings.Contains(dslText, "# BACKENDS") {
		t.Error("missing BACKENDS section")
	}
	if !strings.Contains(dslText, "# GLOBAL") {
		t.Error("missing GLOBAL section")
	}

	// Should contain key elements
	if !strings.Contains(dslText, "SIGNAL domain math") {
		t.Error("missing domain math signal")
	}
	if !strings.Contains(dslText, "ROUTE math_decision") {
		t.Error("missing math_decision route")
	}
	if !strings.Contains(dslText, "default_model:") {
		t.Error("missing default_model in GLOBAL")
	}
}

func TestDecompileRoundTrip(t *testing.T) {
	// Compile DSL → RouterConfig
	cfg1, errs := Compile(fullDSLExample)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	// Decompile → DSL text
	dslText, err := Decompile(cfg1)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}

	// Recompile DSL text → RouterConfig
	cfg2, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("recompile errors: %v\nDSL:\n%s", errs, dslText)
	}

	// Compare key fields
	if cfg1.DefaultModel != cfg2.DefaultModel {
		t.Errorf("default_model: %q vs %q", cfg1.DefaultModel, cfg2.DefaultModel)
	}
	if cfg1.Strategy != cfg2.Strategy {
		t.Errorf("strategy: %q vs %q", cfg1.Strategy, cfg2.Strategy)
	}
	if len(cfg1.Categories) != len(cfg2.Categories) {
		t.Errorf("categories: %d vs %d", len(cfg1.Categories), len(cfg2.Categories))
	}
	if len(cfg1.Decisions) != len(cfg2.Decisions) {
		t.Errorf("decisions: %d vs %d", len(cfg1.Decisions), len(cfg2.Decisions))
	}
	if len(cfg1.VLLMEndpoints) != len(cfg2.VLLMEndpoints) {
		t.Errorf("vllm_endpoints: %d vs %d", len(cfg1.VLLMEndpoints), len(cfg2.VLLMEndpoints))
	}

	// Compare each decision
	for i := range cfg1.Decisions {
		if i >= len(cfg2.Decisions) {
			break
		}
		if cfg1.Decisions[i].Name != cfg2.Decisions[i].Name {
			t.Errorf("decision[%d].name: %q vs %q", i, cfg1.Decisions[i].Name, cfg2.Decisions[i].Name)
		}
		if cfg1.Decisions[i].Priority != cfg2.Decisions[i].Priority {
			t.Errorf("decision[%d].priority: %d vs %d", i, cfg1.Decisions[i].Priority, cfg2.Decisions[i].Priority)
		}
	}
}

func TestDecompileRuleNodeExpressions(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name: "simple signal ref",
			input: `
SIGNAL domain math { description: "Math" }
ROUTE r { PRIORITY 1 WHEN domain("math") MODEL "m:1b" }`,
			expected: `domain("math")`,
		},
		{
			name: "AND expression",
			input: `
SIGNAL domain a { description: "A" }
SIGNAL domain b { description: "B" }
ROUTE r { PRIORITY 1 WHEN domain("a") AND domain("b") MODEL "m:1b" }`,
			expected: `domain("a") AND domain("b")`,
		},
		{
			name: "OR expression",
			input: `
SIGNAL domain a { description: "A" }
SIGNAL domain b { description: "B" }
ROUTE r { PRIORITY 1 WHEN domain("a") OR domain("b") MODEL "m:1b" }`,
			expected: `(domain("a") OR domain("b"))`,
		},
		{
			name: "NOT expression",
			input: `
SIGNAL domain a { description: "A" }
ROUTE r { PRIORITY 1 WHEN NOT domain("a") MODEL "m:1b" }`,
			expected: `NOT domain("a")`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg, errs := Compile(tc.input)
			if len(errs) > 0 {
				t.Fatalf("compile errors: %v", errs)
			}
			dslText, err := Decompile(cfg)
			if err != nil {
				t.Fatalf("decompile error: %v", err)
			}
			if !strings.Contains(dslText, tc.expected) {
				t.Errorf("decompiled DSL does not contain %q\nGot:\n%s", tc.expected, dslText)
			}
		})
	}
}

func TestDecompileModelOptions(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "qwen2.5:3b" (reasoning = true, effort = "high", lora = "math-v2")
}
GLOBAL { default_model: "qwen2.5:3b" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}

	if !strings.Contains(dslText, "reasoning = true") {
		t.Error("missing reasoning option in decompiled output")
	}
	if !strings.Contains(dslText, `effort = "high"`) {
		t.Error("missing effort option in decompiled output")
	}
	if !strings.Contains(dslText, `lora = "math-v2"`) {
		t.Error("missing lora option in decompiled output")
	}
}

func TestDecompileAlgorithmFields(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m1:7b", "m2:3b"
  ALGORITHM confidence {
    confidence_method: "hybrid"
    threshold: 0.5
    on_error: "skip"
  }
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}

	if !strings.Contains(dslText, "ALGORITHM confidence") {
		t.Error("missing algorithm in decompiled output")
	}
	if !strings.Contains(dslText, "confidence_method") {
		t.Error("missing confidence_method in decompiled output")
	}
}

func TestDecompileToAST(t *testing.T) {
	cfg, errs := Compile(fullDSLExample)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	ast := DecompileToAST(cfg)
	if ast == nil {
		t.Fatal("expected non-nil AST")
	}
	if len(ast.Signals) != 7 { // 3 domain + 1 embedding + 1 keyword + 1 context + 1 complexity
		t.Errorf("expected 7 signals, got %d", len(ast.Signals))
	}
	if len(ast.Routes) != 4 {
		t.Errorf("expected 4 routes, got %d", len(ast.Routes))
	}
	if ast.Global == nil {
		t.Error("expected global block")
	}
}

func TestDecompileAllSignalTypes(t *testing.T) {
	input := `
SIGNAL keyword kw { operator: "any" keywords: ["test"] }
SIGNAL embedding emb { threshold: 0.75 candidates: ["test"] }
SIGNAL domain dom { description: "test" mmlu_categories: ["math"] }
SIGNAL fact_check fc { description: "fact check" }
SIGNAL user_feedback uf { description: "feedback" }
SIGNAL preference pref { description: "preference" }
SIGNAL language lang { description: "English" }
SIGNAL context ctx { min_tokens: "1K" max_tokens: "32K" }
SIGNAL complexity comp { threshold: 0.1 hard: { candidates: ["hard"] } easy: { candidates: ["easy"] } }
SIGNAL modality mod { description: "image" }
SIGNAL authz auth { role: "admin" subjects: [{ kind: "User", name: "admin" }] }
ROUTE test_route { PRIORITY 1 WHEN domain("dom") MODEL "m:1b" }
GLOBAL { default_model: "m:1b" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}

	expectedSignals := []string{
		"SIGNAL domain dom", "SIGNAL keyword kw", "SIGNAL embedding emb",
		"SIGNAL fact_check fc", "SIGNAL user_feedback uf",
		"SIGNAL preference pref", "SIGNAL language lang",
		"SIGNAL context ctx", "SIGNAL complexity comp",
		"SIGNAL modality mod", "SIGNAL authz auth",
	}
	for _, sig := range expectedSignals {
		if !strings.Contains(dslText, sig) {
			t.Errorf("missing %q in decompiled output", sig)
		}
	}
}

func TestDecompileBackendTypes(t *testing.T) {
	input := `
BACKEND vllm_endpoint ep1 { address: "10.0.0.1" port: 8000 }
BACKEND provider_profile pp1 { type: "openai" base_url: "https://api.openai.com/v1" }
BACKEND embedding_model em1 { mmbert_model_path: "models/mmbert" use_cpu: true }
BACKEND semantic_cache sc1 { enabled: true backend_type: "redis" }
BACKEND memory mem1 { enabled: true auto_store: true }
BACKEND response_api resp1 { enabled: true store_backend: "redis" }
GLOBAL { default_model: "m:1b" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}

	expectedBackends := []string{
		"BACKEND vllm_endpoint", "BACKEND provider_profile",
		"BACKEND embedding_model", "BACKEND semantic_cache",
		"BACKEND memory", "BACKEND response_api",
	}
	for _, be := range expectedBackends {
		if !strings.Contains(dslText, be) {
			t.Errorf("missing %q in decompiled output", be)
		}
	}
}

func TestDecompileRouteDescription(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE my_route (description = "A detailed description") {
  PRIORITY 42
  WHEN domain("test")
  MODEL "m:1b"
}
GLOBAL { default_model: "m:1b" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}

	if !strings.Contains(dslText, "description = \"A detailed description\"") {
		t.Errorf("route description not preserved in decompiled output:\n%s", dslText)
	}
}

// ==================== Step 7: Format Tests ====================

func TestFormatProducesValidDSL(t *testing.T) {
	// Messy input with inconsistent formatting
	input := `SIGNAL domain math{description:"Math" mmlu_categories:["math"]}
ROUTE r{PRIORITY 100 WHEN domain("math") MODEL "m:1b"(reasoning=true)}
GLOBAL{default_model:"m:1b" strategy:"priority"}`

	formatted, err := Format(input)
	if err != nil {
		t.Fatalf("format error: %v", err)
	}

	// Formatted output should be valid DSL
	cfg, errs := Compile(formatted)
	if len(errs) > 0 {
		t.Fatalf("formatted output is not valid DSL: %v\nFormatted:\n%s", errs, formatted)
	}

	if cfg.DefaultModel != "m:1b" {
		t.Errorf("default_model = %q after format round-trip", cfg.DefaultModel)
	}
}

func TestFormatIdempotency(t *testing.T) {
	// Format once to get canonical form, then format again — second and third should be identical
	input := fullDSLExample

	formatted1, err := Format(input)
	if err != nil {
		t.Fatalf("first format error: %v", err)
	}

	formatted2, err := Format(formatted1)
	if err != nil {
		t.Fatalf("second format error: %v", err)
	}

	formatted3, err := Format(formatted2)
	if err != nil {
		t.Fatalf("third format error: %v", err)
	}

	// After the second pass, output should stabilize
	if formatted2 != formatted3 {
		// Find differences
		lines2 := strings.Split(formatted2, "\n")
		lines3 := strings.Split(formatted3, "\n")
		maxLen := len(lines2)
		if len(lines3) > maxLen {
			maxLen = len(lines3)
		}
		for i := 0; i < maxLen; i++ {
			var l2, l3 string
			if i < len(lines2) {
				l2 = lines2[i]
			}
			if i < len(lines3) {
				l3 = lines3[i]
			}
			if l2 != l3 {
				t.Errorf("line %d diff:\n  fmt2: %q\n  fmt3: %q", i+1, l2, l3)
			}
		}
		t.Error("Format is not idempotent: formatting the second and third times produces different results")
	}
}

// ==================== CLI Unit Tests ====================

func TestCLIValidateOutput(t *testing.T) {
	// Write a test DSL file
	tmpFile, err := os.CreateTemp("", "test*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	_, _ = tmpFile.WriteString(fullDSLExample)
	tmpFile.Close()

	var buf bytes.Buffer
	errCount := CLIValidate(tmpFile.Name(), &buf)

	output := buf.String()
	if errCount > 0 {
		t.Errorf("expected no errors for valid input, got %d\nOutput: %s", errCount, output)
	}
	if !strings.Contains(output, "No issues found") {
		t.Errorf("expected 'No issues found' message, got: %s", output)
	}
}

func TestCLIValidateWithErrors(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test_bad*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	// Write DSL with undefined reference
	_, _ = tmpFile.WriteString(`
SIGNAL domain math { description: "Math" }
ROUTE test {
  PRIORITY 1
  WHEN domain("nonexistent")
  MODEL "m:1b"
}
`)
	tmpFile.Close()

	var buf bytes.Buffer
	_ = CLIValidate(tmpFile.Name(), &buf)

	output := buf.String()
	if !strings.Contains(output, "nonexistent") {
		t.Errorf("expected warning about 'nonexistent', got: %s", output)
	}
}

func TestCLICompileAndDecompile(t *testing.T) {
	// Write a DSL file
	dslFile, err := os.CreateTemp("", "test*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(dslFile.Name())
	_, _ = dslFile.WriteString(fullDSLExample)
	dslFile.Close()

	// Compile DSL → YAML
	yamlFile, err := os.CreateTemp("", "test*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(yamlFile.Name())
	yamlFile.Close()

	if compileErr := CLICompile(dslFile.Name(), yamlFile.Name(), "yaml", "", ""); compileErr != nil {
		t.Fatalf("CLICompile error: %v", compileErr)
	}

	// Read the YAML output
	yamlData, err := os.ReadFile(yamlFile.Name())
	if err != nil {
		t.Fatal(err)
	}
	if len(yamlData) == 0 {
		t.Fatal("YAML output is empty")
	}

	// Decompile YAML → DSL
	dslOutFile, err := os.CreateTemp("", "test_out*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(dslOutFile.Name())
	dslOutFile.Close()

	if decompileErr := CLIDecompile(yamlFile.Name(), dslOutFile.Name()); decompileErr != nil {
		t.Fatalf("CLIDecompile error: %v", decompileErr)
	}

	// Read the DSL output
	dslData, err := os.ReadFile(dslOutFile.Name())
	if err != nil {
		t.Fatal(err)
	}
	if len(dslData) == 0 {
		t.Fatal("DSL output is empty")
	}

	// The decompiled DSL should be valid — compile it again
	cfg, errs := Compile(string(dslData))
	if len(errs) > 0 {
		t.Fatalf("recompile errors: %v\nDSL:\n%s", errs, string(dslData))
	}
	if cfg.DefaultModel != "qwen2.5:3b" {
		t.Errorf("round-trip default_model = %q", cfg.DefaultModel)
	}
}

func TestCLICompileCRD(t *testing.T) {
	dslFile, err := os.CreateTemp("", "test*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(dslFile.Name())
	_, _ = dslFile.WriteString(fullDSLExample)
	dslFile.Close()

	crdFile, err := os.CreateTemp("", "test*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(crdFile.Name())
	crdFile.Close()

	if compileErr := CLICompile(dslFile.Name(), crdFile.Name(), "crd", "my-router", "production"); compileErr != nil {
		t.Fatalf("CLICompile CRD error: %v", compileErr)
	}

	data, err := os.ReadFile(crdFile.Name())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "apiVersion: vllm.ai/v1alpha1") {
		t.Error("CRD output missing apiVersion")
	}
	if !strings.Contains(string(data), "kind: SemanticRouter") {
		t.Error("CRD output missing kind")
	}
	if !strings.Contains(string(data), "name: my-router") {
		t.Error("CRD output missing name")
	}
}

func TestCLIFormat(t *testing.T) {
	dslFile, err := os.CreateTemp("", "test*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(dslFile.Name())
	_, _ = dslFile.WriteString(fullDSLExample)
	dslFile.Close()

	outFile, err := os.CreateTemp("", "test_fmt*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(outFile.Name())
	outFile.Close()

	if fmtErr := CLIFormat(dslFile.Name(), outFile.Name()); fmtErr != nil {
		t.Fatalf("CLIFormat error: %v", fmtErr)
	}

	data, err := os.ReadFile(outFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	// The formatted output should be valid DSL
	_, errs := Compile(string(data))
	if len(errs) > 0 {
		t.Fatalf("formatted output is not valid DSL: %v", errs)
	}
}

// TestEmitUserYAML verifies that EmitUserYAML produces nested signals/providers format.
func TestEmitUserYAML(t *testing.T) {
	cfg, errs := Compile(fullDSLExample)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	userYAML, err := EmitUserYAML(cfg)
	if err != nil {
		t.Fatalf("EmitUserYAML error: %v", err)
	}

	yamlStr := string(userYAML)

	// Should have nested "signals" section
	if !strings.Contains(yamlStr, "signals:") {
		t.Error("expected 'signals:' section in user YAML")
	}
	// Should have nested signal types
	if !strings.Contains(yamlStr, "domains:") {
		t.Error("expected 'domains:' under signals")
	}
	if !strings.Contains(yamlStr, "keywords:") {
		t.Error("expected 'keywords:' under signals")
	}
	if !strings.Contains(yamlStr, "embeddings:") {
		t.Error("expected 'embeddings:' under signals")
	}
	if !strings.Contains(yamlStr, "context:") {
		t.Error("expected 'context:' under signals")
	}

	// Should have nested "providers" section
	if !strings.Contains(yamlStr, "providers:") {
		t.Error("expected 'providers:' section in user YAML")
	}
	if !strings.Contains(yamlStr, "default_model:") {
		t.Error("expected 'default_model:' under providers")
	}

	// Should NOT have flat RouterConfig keys at top level
	if strings.Contains(yamlStr, "keyword_rules:") {
		t.Error("should not have flat 'keyword_rules:' key")
	}
	if strings.Contains(yamlStr, "embedding_rules:") {
		t.Error("should not have flat 'embedding_rules:' key")
	}
	// Check that top-level "categories:" is gone (note: mmlu_categories is fine as a nested field)
	if strings.Contains(yamlStr, "\ncategories:") || strings.HasPrefix(yamlStr, "categories:") {
		t.Error("should not have top-level 'categories:' key (should be signals.domains)")
	}
	if strings.Contains(yamlStr, "vllm_endpoints:") {
		t.Error("should not have flat 'vllm_endpoints:' key (should be providers.models)")
	}
}

// TestEmitHelm verifies that EmitHelm produces a valid Helm values.yaml structure.
func TestEmitHelm(t *testing.T) {
	cfg, errs := Compile(fullDSLExample)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	helmYAML, err := EmitHelm(cfg)
	if err != nil {
		t.Fatalf("EmitHelm error: %v", err)
	}

	yamlStr := string(helmYAML)

	// Should have top-level "config:" key
	if !strings.Contains(yamlStr, "config:") {
		t.Error("expected top-level 'config:' key in Helm values")
	}
	// Should contain decisions under config
	if !strings.Contains(yamlStr, "decisions:") {
		t.Error("expected 'decisions:' under config")
	}
	// Should contain default_model
	if !strings.Contains(yamlStr, "default_model:") {
		t.Error("expected 'default_model:' in Helm values")
	}
	// Should NOT have apiVersion (that's CRD format)
	if strings.Contains(yamlStr, "apiVersion:") {
		t.Error("Helm values should not contain apiVersion")
	}
	// Should NOT have kind
	if strings.Contains(yamlStr, "kind:") {
		t.Error("Helm values should not contain kind")
	}
}

// TestCLICompileHelm verifies the CLI can compile to Helm format.
func TestCLICompileHelm(t *testing.T) {
	dslFile, err := os.CreateTemp("", "test*.dsl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(dslFile.Name())
	_, _ = dslFile.WriteString(fullDSLExample)
	dslFile.Close()

	helmFile, err := os.CreateTemp("", "test*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(helmFile.Name())
	helmFile.Close()

	if compileErr := CLICompile(dslFile.Name(), helmFile.Name(), "helm", "", ""); compileErr != nil {
		t.Fatalf("CLICompile Helm error: %v", compileErr)
	}

	data, err := os.ReadFile(helmFile.Name())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "config:") {
		t.Error("Helm output missing config: key")
	}
	if !strings.Contains(string(data), "decisions:") {
		t.Error("Helm output missing decisions")
	}
}

// TestDecompileGlobalAuthzRatelimit verifies authz and ratelimit round-trip through decompile.
func TestDecompileGlobalAuthzRatelimit(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE test_route { PRIORITY 1 WHEN domain("test") MODEL "m:1b" }
GLOBAL {
  default_model: "m:1b"
  authz: { fail_open: true }
  ratelimit: { fail_open: true }
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	if !cfg.Authz.FailOpen {
		t.Fatal("authz.fail_open should be true after compile")
	}
	if !cfg.RateLimit.FailOpen {
		t.Fatal("ratelimit.fail_open should be true after compile")
	}

	// Decompile and verify the text output includes authz/ratelimit
	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}

	if !strings.Contains(dslText, "authz") {
		t.Error("decompiled DSL missing 'authz'")
	}
	if !strings.Contains(dslText, "ratelimit") {
		t.Error("decompiled DSL missing 'ratelimit'")
	}
	if !strings.Contains(dslText, "fail_open") {
		t.Error("decompiled DSL missing 'fail_open'")
	}

	// Re-compile from decompiled text — should preserve authz/ratelimit
	cfg2, errs2 := Compile(dslText)
	if len(errs2) > 0 {
		t.Fatalf("re-compile errors: %v", errs2)
	}
	if !cfg2.Authz.FailOpen {
		t.Error("authz.fail_open lost during round-trip")
	}
	if !cfg2.RateLimit.FailOpen {
		t.Error("ratelimit.fail_open lost during round-trip")
	}
}

// TestDecompileRAGPlugin verifies RAG plugin config decompiles correctly.
func TestDecompileRAGPlugin(t *testing.T) {
	// Build a config with RAG plugin via compile, then inject the RAG plugin
	input := `
SIGNAL domain test { description: "test" }
ROUTE rag_route (description = "RAG route") {
  PRIORITY 100
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN rag {
    enabled: true
    backend: "milvus"
    similarity_threshold: 0.8
    top_k: 5
    max_context_length: 2000
    injection_mode: "tool_role"
  }
}
GLOBAL { default_model: "m:1b" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}

	if !strings.Contains(dslText, "rag") {
		t.Error("decompiled DSL missing 'rag' plugin")
	}
	if !strings.Contains(dslText, "milvus") {
		t.Error("decompiled DSL missing RAG backend 'milvus'")
	}
	if !strings.Contains(dslText, "0.8") {
		t.Error("decompiled DSL missing RAG similarity_threshold")
	}
	if !strings.Contains(dslText, "top_k") {
		t.Error("decompiled DSL missing RAG top_k")
	}
	if !strings.Contains(dslText, "tool_role") {
		t.Error("decompiled DSL missing RAG injection_mode")
	}
}
