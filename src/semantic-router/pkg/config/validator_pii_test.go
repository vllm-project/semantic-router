package config

import (
	"testing"

	"gopkg.in/yaml.v2"
)

func TestPIIRuleSourceYAMLContract(t *testing.T) {
	var signals Signals
	if err := yaml.Unmarshal([]byte(`pii:
  - name: tool-data
    source: tool_result
`), &signals); err != nil {
		t.Fatalf("yaml.Unmarshal() error = %v", err)
	}

	if len(signals.PIIRules) != 1 || signals.PIIRules[0].Source != PIISourceToolResult {
		t.Fatalf("PII rules = %#v, want source %q", signals.PIIRules, PIISourceToolResult)
	}
}

func TestValidatePIIContractsAllowsLegacyAndToolResultSources(t *testing.T) {
	for _, source := range []string{"", PIISourceToolResult} {
		if err := validatePIIContracts(&RouterConfig{IntelligentRouting: IntelligentRouting{
			Signals: Signals{PIIRules: []PIIRule{{Name: "pii-rule", Source: source}}},
		}}); err != nil {
			t.Fatalf("validatePIIContracts(source=%q) error = %v", source, err)
		}
	}
}

func TestValidatePIIContractsRejectsUnknownSource(t *testing.T) {
	err := validatePIIContracts(&RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{PIIRules: []PIIRule{{Name: "pii-rule", Source: "assistant_output"}}},
	}})
	if err == nil {
		t.Fatal("validatePIIContracts() error = nil, want unknown source error")
	}
}
