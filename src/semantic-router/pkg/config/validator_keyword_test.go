package config

import (
	"strings"
	"testing"
)

// A count or density counts the occurrences an OR rule's literal or regex
// keywords have; every other combination is refused, and a rule without
// either keeps its operator and method.
func TestKeywordRuleCountAndDensityContract(t *testing.T) {
	three, four := 3.0, 4.0
	counting := func(edit func(*KeywordRule)) KeywordRule {
		rule := KeywordRule{Name: "questions", Operator: "OR", Method: "regex", Keywords: []string{`\?`}, Count: &NumericPredicate{GTE: &three}}
		edit(&rule)
		return rule
	}
	valid := map[string]KeywordRule{
		"regex count":     counting(func(*KeywordRule) {}),
		"literal density": counting(func(r *KeywordRule) { r.Method, r.Count, r.Density = "", nil, &NumericPredicate{GT: &four} }),
		"no count":        {Name: "all", Operator: "AND", Method: "bm25", Keywords: []string{"a", "b"}},
	}
	for name, rule := range valid {
		if err := ValidateKeywordRuleContract(rule); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	invalid := map[string]struct {
		rule KeywordRule
		want string
	}{
		"both":        {counting(func(r *KeywordRule) { r.Density = &NumericPredicate{GT: &four} }), "not both"},
		"empty":       {counting(func(r *KeywordRule) { r.Count = &NumericPredicate{} }), "count: predicate must set at least one comparator"},
		"and":         {counting(func(r *KeywordRule) { r.Operator = "AND" }), `operator "AND"`},
		"nor":         {counting(func(r *KeywordRule) { r.Operator = "NOR" }), `operator "NOR"`},
		"bm25":        {counting(func(r *KeywordRule) { r.Method = "bm25" }), "method bm25"},
		"ngram":       {counting(func(r *KeywordRule) { r.Method = "ngram" }), "method ngram"},
		"fuzzy match": {counting(func(r *KeywordRule) { r.Method, r.FuzzyMatch = "", true }), "fuzzy_match"},
	}
	for name, test := range invalid {
		if err := ValidateKeywordRuleContract(test.rule); err == nil || !strings.Contains(err.Error(), test.want) {
			t.Errorf("%s: %v, want %q", name, err, test.want)
		}
	}
	cfg := &RouterConfig{}
	cfg.KeywordRules = []KeywordRule{invalid["and"].rule}
	if err := validateKeywordContracts(cfg); err == nil || !strings.HasPrefix(err.Error(), `recipes[default].signals.keywords["questions"]: count`) {
		t.Fatalf("the error names the rule: %v", err)
	}
}

// A rule reads the tool results beside or instead of the current user turn,
// with literal or regex keywords; on_unscanned needs a rule that reads them.
func TestKeywordRuleInputContract(t *testing.T) {
	no, three := false, 3.0
	toolResults := func(edit func(*KeywordRule)) KeywordRule {
		rule := KeywordRule{Name: "credentials", Operator: "OR", Method: "regex", Keywords: []string{`AKIA[0-9A-Z]{16}`}, Input: KeywordRuleInput{ToolResults: true}}
		edit(&rule)
		return rule
	}
	valid := map[string]KeywordRule{
		"beside the turn":  toolResults(func(*KeywordRule) {}),
		"instead of it":    toolResults(func(r *KeywordRule) { r.Input.Request = &no }),
		"literal keywords": toolResults(func(r *KeywordRule) { r.Method = "" }),
		"history only":     {Name: "earlier", Operator: "OR", Keywords: []string{"x"}, IncludeHistory: true, Input: KeywordRuleInput{Request: &no}},
		"match":            toolResults(func(r *KeywordRule) { r.OnUnscanned = KeywordUnscannedMatch }),
		"no_match":         toolResults(func(r *KeywordRule) { r.OnUnscanned = KeywordUnscannedNoMatch }),
		"best_effort":      toolResults(func(r *KeywordRule) { r.OnUnscanned = KeywordUnscannedBestEffort }),
	}
	for name, rule := range valid {
		if err := ValidateKeywordRuleContract(rule); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	invalid := map[string]struct {
		rule KeywordRule
		want string
	}{
		"nothing read":  {KeywordRule{Name: "none", Operator: "OR", Keywords: []string{"x"}, Input: KeywordRuleInput{Request: &no}}, "input.request: false"},
		"bm25":          {toolResults(func(r *KeywordRule) { r.Method = "bm25" }), "method bm25"},
		"ngram":         {toolResults(func(r *KeywordRule) { r.Method = "NGRAM" }), "method NGRAM"},
		"fuzzy":         {toolResults(func(r *KeywordRule) { r.Method, r.FuzzyMatch = "", true }), "fuzzy_match"},
		"count":         {toolResults(func(r *KeywordRule) { r.Count = &NumericPredicate{GTE: &three} }), "count and density"},
		"unknown":       {toolResults(func(r *KeywordRule) { r.OnUnscanned = "unknown" }), `on_unscanned "unknown"`},
		"not read":      {KeywordRule{Name: "turn", Operator: "OR", Keywords: []string{"x"}, OnUnscanned: KeywordUnscannedMatch}, "reads input.tool_results"},
		"history alone": {KeywordRule{Name: "turn", Operator: "OR", Keywords: []string{"x"}, IncludeHistory: true, OnUnscanned: KeywordUnscannedMatch}, "reads input.tool_results"},
	}
	for name, test := range invalid {
		if err := ValidateKeywordRuleContract(test.rule); err == nil || !strings.Contains(err.Error(), test.want) {
			t.Errorf("%s: %v, want %q", name, err, test.want)
		}
	}
}

// input.request defaults to reading the current turn, and false is kept apart
// from unset so the DSL and the dashboard can round-trip it.
func TestKeywordRuleInputReadsRequest(t *testing.T) {
	yes, no := true, false
	for _, test := range []struct {
		input KeywordRuleInput
		want  bool
	}{{KeywordRuleInput{}, true}, {KeywordRuleInput{Request: &yes}, true}, {KeywordRuleInput{Request: &no}, false}} {
		if got := test.input.ReadsRequest(); got != test.want {
			t.Errorf("%#v.ReadsRequest() = %v, want %v", test.input, got, test.want)
		}
	}
}
