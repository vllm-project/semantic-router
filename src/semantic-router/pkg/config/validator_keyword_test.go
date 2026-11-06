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
	if err := validateKeywordContracts(cfg); err == nil || !strings.HasPrefix(err.Error(), `routing.signals.keywords["questions"]: count`) {
		t.Fatalf("the error names the rule: %v", err)
	}
}
