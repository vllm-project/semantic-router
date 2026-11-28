package classification

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// RuleSpans finds what matching matches, at its places in code points: a
// regex at each non-overlapping match, a literal only as a word of its own,
// case folded unless the rule is case sensitive.
func TestKeywordRuleSpansFindWhatMatchingMatches(t *testing.T) {
	for _, tc := range []struct {
		name string
		rule config.KeywordRule
		text string
		want []string
	}{
		{
			"regex",
			config.KeywordRule{Method: "regex", Keywords: []string{`\bE-\d{6}\b`}},
			"E-104233 paid E-998877, not E-1234567",
			[]string{"E-104233", "E-998877"},
		},
		{
			"literal word",
			config.KeywordRule{Keywords: []string{"cat"}},
			"cat concatenate Cat",
			[]string{"cat", "Cat"},
		},
		{
			"case sensitive",
			config.KeywordRule{Keywords: []string{"cat"}, CaseSensitive: true},
			"cat Cat",
			[]string{"cat"},
		},
		{
			"code points",
			config.KeywordRule{Method: "regex", Keywords: []string{`acct-\d+`}},
			"계정 acct-42 확인",
			[]string{"acct-42"},
		},
		{
			"two keywords in order",
			config.KeywordRule{Method: "regex", Keywords: []string{`b+`, `a+`}},
			"aa bb aa",
			[]string{"aa", "bb", "aa"},
		},
		{
			"empty match",
			config.KeywordRule{Method: "regex", Keywords: []string{`(?m)^`}},
			"one\ntwo", nil,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.rule.Name, tc.rule.Operator = "found", "OR"
			c, err := NewKeywordClassifier([]config.KeywordRule{tc.rule})
			if err != nil {
				t.Fatal(err)
			}
			spans, err := c.RuleSpans("found", tc.text)
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			runes := []rune(tc.text)
			for _, span := range spans {
				if span.Label != "found" || string(runes[span.Start:span.End]) != span.Text {
					t.Fatalf("span %+v does not cover its text in code points of %q", span, tc.text)
				}
				got = append(got, span.Text)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("spans = %q, want %q", got, tc.want)
			}
		})
	}
}

// A rule that does not say where it matched has no spans, and the config
// check that keeps it out of masking gives the reason.
func TestKeywordRuleSpansRefuseRulesThatDoNotLocate(t *testing.T) {
	gte := 2.0
	for _, rule := range []config.KeywordRule{
		{Operator: "AND", Keywords: []string{"a", "b"}},
		{Operator: "NOR", Keywords: []string{"a"}},
		{Operator: "OR", Keywords: []string{"a"}, FuzzyMatch: true},
		{Operator: "OR", Keywords: []string{"a"}, Method: "bm25"},
		{Operator: "OR", Keywords: []string{"a"}, Count: &config.NumericPredicate{GTE: &gte}},
	} {
		rule.Name = "rule"
		t.Run(rule.Operator+"/"+rule.Method, func(t *testing.T) {
			if err := config.KeywordRuleFindsText(rule); err == nil {
				t.Fatalf("KeywordRuleFindsText accepted %+v", rule)
			}
			c, err := NewKeywordClassifier([]config.KeywordRule{rule})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.RuleSpans("rule", "a b"); err == nil || !strings.Contains(err.Error(), "does not locate") {
				t.Fatalf("RuleSpans error = %v, want a refusal", err)
			}
		})
	}
	if _, err := (&KeywordClassifier{}).RuleSpans("missing", "a"); err == nil {
		t.Fatal("RuleSpans found a rule that is not declared")
	}
}
