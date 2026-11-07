package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestKeywordRuleSourceYAMLContract(t *testing.T) {
	var signals Signals
	if err := yaml.Unmarshal([]byte(`keyword_rules:
  - name: credentials
    operator: OR
    method: regex
    keywords: ["AKIA[0-9A-Z]{16}"]
    source: tool_result
`), &signals); err != nil {
		t.Fatalf("yaml.Unmarshal() error = %v", err)
	}

	if len(signals.KeywordRules) != 1 || signals.KeywordRules[0].Source != KeywordSourceToolResult {
		t.Fatalf("keyword rules = %#v, want source %q", signals.KeywordRules, KeywordSourceToolResult)
	}
}

func TestValidateKeywordContractsAllowsLegacyAndToolResultSources(t *testing.T) {
	for _, source := range []string{"", KeywordSourceToolResult} {
		if err := validateKeywordContracts(&RouterConfig{IntelligentRouting: IntelligentRouting{
			Signals: Signals{KeywordRules: []KeywordRule{{Name: "rule", Operator: "OR", Keywords: []string{"x"}, Source: source}}},
		}}); err != nil {
			t.Fatalf("validateKeywordContracts(source=%q) error = %v", source, err)
		}
	}
}

func TestValidateKeywordContractsRejectsUnknownSource(t *testing.T) {
	err := validateKeywordContracts(&RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{KeywordRules: []KeywordRule{{Name: "rule", Operator: "OR", Keywords: []string{"x"}, Source: "assistant_output"}}},
	}})
	if err == nil {
		t.Fatal("validateKeywordContracts() error = nil, want unknown source error")
	}
	if !strings.Contains(err.Error(), `"assistant_output"`) {
		t.Fatalf("validateKeywordContracts() error = %q, want the rejected source named", err)
	}
}

func TestValidateKeywordContractsRejectsToolResultWithScoredMethod(t *testing.T) {
	for _, method := range []string{"bm25", "ngram", "BM25"} {
		err := validateKeywordContracts(&RouterConfig{IntelligentRouting: IntelligentRouting{
			Signals: Signals{KeywordRules: []KeywordRule{{
				Name: "rule", Operator: "OR", Keywords: []string{"x"},
				Source: KeywordSourceToolResult, Method: method,
			}}},
		}})
		if err == nil {
			t.Fatalf("validateKeywordContracts(method=%q) error = nil, want regex-only error", method)
		}
		if !strings.Contains(err.Error(), "regex only") {
			t.Fatalf("validateKeywordContracts(method=%q) error = %q, want regex-only named", method, err)
		}
	}
	for _, method := range []string{"", "regex"} {
		if err := validateKeywordContracts(&RouterConfig{IntelligentRouting: IntelligentRouting{
			Signals: Signals{KeywordRules: []KeywordRule{{
				Name: "rule", Operator: "OR", Keywords: []string{"x"},
				Source: KeywordSourceToolResult, Method: method,
			}}},
		}}); err != nil {
			t.Fatalf("validateKeywordContracts(method=%q) error = %v", method, err)
		}
	}
}

func TestValidateKeywordContractsRejectsToolResultWithIncludeHistory(t *testing.T) {
	err := validateKeywordContracts(&RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{KeywordRules: []KeywordRule{{
			Name: "rule", Operator: "OR", Keywords: []string{"x"},
			Source: KeywordSourceToolResult, IncludeHistory: true,
		}}},
	}})
	if err == nil {
		t.Fatal("validateKeywordContracts() error = nil, want include_history conflict error")
	}
	if !strings.Contains(err.Error(), "include_history") {
		t.Fatalf("validateKeywordContracts() error = %q, want include_history named", err)
	}
}
