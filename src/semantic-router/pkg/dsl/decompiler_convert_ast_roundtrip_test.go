package dsl

import (
	"math"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestDecompileToASTPreservesKeywordRuleFields locks in the AST emit contract
// for KeywordRule. The runtime KeywordRule schema carries 9 configurable
// fields; DecompileToAST must surface every non-zero field so that
// programmatic consumers of the AST (dashboard editors, projection contract
// alignment, CLI validators) do not silently observe a truncated rule.
//
// Regression coverage for the gap where fuzzy_match, fuzzy_threshold,
// bm25_threshold, ngram_threshold, and ngram_arity were dropped on the
// AST path even though they round-trip correctly through DSL text.
func TestDecompileToASTPreservesKeywordRuleFields(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				KeywordRules: []config.KeywordRule{{
					Name:           "fuzzy_kw",
					Operator:       "any",
					Keywords:       []string{"alpha", "beta"},
					CaseSensitive:  true,
					Method:         "fuzzy",
					FuzzyMatch:     true,
					FuzzyThreshold: 80,
					BM25Threshold:  0.5,
					NgramThreshold: 0.25,
					NgramArity:     3,
				}},
			},
		},
	}

	prog := DecompileToAST(cfg)

	sig := findSignal(t, prog, "keyword", "fuzzy_kw")
	requireStringField(t, sig, "operator", "any")
	requireFieldExists(t, sig, "keywords")
	requireBoolField(t, sig, "case_sensitive", true)
	requireStringField(t, sig, "method", "fuzzy")
	requireBoolField(t, sig, "fuzzy_match", true)
	requireIntField(t, sig, "fuzzy_threshold", 80)
	requireFloatField(t, sig, "bm25_threshold", 0.5)
	requireFloatField(t, sig, "ngram_threshold", 0.25)
	requireIntField(t, sig, "ngram_arity", 3)
}

// TestDecompileToASTPreservesContextRuleFields locks in the AST emit
// contract for ContextRule.Description. Without this, a runtime ContextRule
// authored with a human-readable description is silently stripped of that
// description when surfaced through DecompileToAST.
func TestDecompileToASTPreservesContextRuleFields(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ContextRules: []config.ContextRule{{
					Name:        "long_ctx",
					MinTokens:   "1K",
					MaxTokens:   "32K",
					Description: "Long requests requiring large context window",
				}},
			},
		},
	}

	prog := DecompileToAST(cfg)

	sig := findSignal(t, prog, "context", "long_ctx")
	requireStringField(t, sig, "min_tokens", "1K")
	requireStringField(t, sig, "max_tokens", "32K")
	requireStringField(t, sig, "description", "Long requests requiring large context window")
}

// TestDecompileToASTPreservesAuthzDescription locks in the AST emit contract
// for RoleBinding.Description. compileAuthzSignal reads this field, so the
// decompiler must emit it to avoid a lossy runtime config -> AST round trip.
func TestDecompileToASTPreservesAuthzDescription(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				RoleBindings: []config.RoleBinding{{
					Name:        "admins",
					Role:        "admin",
					Description: "Admin callers",
					Subjects: []config.Subject{{
						Kind: "User",
						Name: "alice",
					}},
				}},
			},
		},
	}

	prog := DecompileToAST(cfg)

	sig := findSignal(t, prog, "authz", "admins")
	requireStringField(t, sig, "role", "admin")
	requireStringField(t, sig, "description", "Admin callers")
	requireFieldExists(t, sig, "subjects")
}

func TestDecompilePreservesAuthzDescription(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				RoleBindings: []config.RoleBinding{{
					Name:        "admins",
					Role:        "admin",
					Description: "Admin callers",
				}},
			},
		},
	}

	out, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("Decompile: %v", err)
	}

	if !strings.Contains(out, `description: "Admin callers"`) {
		t.Fatalf("expected authz description in decompiled DSL, got:\n%s", out)
	}
}

// TestDecompileToASTOmitsZeroValuedKeywordRuleFields documents the
// zero-value omission contract: only non-zero / non-empty fields should
// appear in the AST, preserving the existing behaviour for the original
// four fields and matching the convention used by the text decompiler.
func TestDecompileToASTOmitsZeroValuedKeywordRuleFields(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				KeywordRules: []config.KeywordRule{{
					Name:     "minimal_kw",
					Operator: "any",
					Keywords: []string{"foo"},
				}},
			},
		},
	}

	prog := DecompileToAST(cfg)
	sig := findSignal(t, prog, "keyword", "minimal_kw")

	for _, omitted := range []string{
		"case_sensitive", "method", "fuzzy_match", "fuzzy_threshold",
		"bm25_threshold", "ngram_threshold", "ngram_arity",
	} {
		if _, ok := sig.Fields[omitted]; ok {
			t.Errorf("expected %q to be omitted from AST when zero-valued, but it was present", omitted)
		}
	}
}

func findSignal(t *testing.T, prog *Program, signalType, name string) *SignalDecl {
	t.Helper()
	for _, sig := range prog.Signals {
		if sig.SignalType == signalType && sig.Name == name {
			return sig
		}
	}
	t.Fatalf("expected SIGNAL %s %s in AST, not found", signalType, name)
	return nil
}

func requireFieldExists(t *testing.T, sig *SignalDecl, name string) {
	t.Helper()
	if _, ok := sig.Fields[name]; !ok {
		t.Errorf("expected SIGNAL %s %s to carry field %q", sig.SignalType, sig.Name, name)
	}
}

func requireStringField(t *testing.T, sig *SignalDecl, name, want string) {
	t.Helper()
	v, ok := sig.Fields[name]
	if !ok {
		t.Errorf("expected SIGNAL %s %s to carry string field %q", sig.SignalType, sig.Name, name)
		return
	}
	sv, ok := v.(StringValue)
	if !ok {
		t.Errorf("field %q: expected StringValue, got %T", name, v)
		return
	}
	if sv.V != want {
		t.Errorf("field %q: want %q, got %q", name, want, sv.V)
	}
}

func requireBoolField(t *testing.T, sig *SignalDecl, name string, want bool) {
	t.Helper()
	v, ok := sig.Fields[name]
	if !ok {
		t.Errorf("expected SIGNAL %s %s to carry bool field %q", sig.SignalType, sig.Name, name)
		return
	}
	bv, ok := v.(BoolValue)
	if !ok {
		t.Errorf("field %q: expected BoolValue, got %T", name, v)
		return
	}
	if bv.V != want {
		t.Errorf("field %q: want %v, got %v", name, want, bv.V)
	}
}

func requireIntField(t *testing.T, sig *SignalDecl, name string, want int) {
	t.Helper()
	v, ok := sig.Fields[name]
	if !ok {
		t.Errorf("expected SIGNAL %s %s to carry int field %q", sig.SignalType, sig.Name, name)
		return
	}
	iv, ok := v.(IntValue)
	if !ok {
		t.Errorf("field %q: expected IntValue, got %T", name, v)
		return
	}
	if iv.V != want {
		t.Errorf("field %q: want %d, got %d", name, want, iv.V)
	}
}

func requireFloatField(t *testing.T, sig *SignalDecl, name string, want float64) {
	t.Helper()
	v, ok := sig.Fields[name]
	if !ok {
		t.Errorf("expected SIGNAL %s %s to carry float field %q", sig.SignalType, sig.Name, name)
		return
	}
	fv, ok := v.(FloatValue)
	if !ok {
		t.Errorf("field %q: expected FloatValue, got %T", name, v)
		return
	}
	if math.Abs(fv.V-want) > 0.000001 {
		t.Errorf("field %q: want %v, got %v", name, want, fv.V)
	}
}
