package config

import (
	"fmt"
	"strings"
	"testing"
)

func systemOneClassifierConfig(rule ClassifierSignalRule) *RouterConfig {
	return &RouterConfig{
		ExternalModels: []ExternalModelConfig{{
			Name:          "decision-runtime",
			ModelRole:     ModelRoleClassification,
			ModelName:     "llm-semantic-router/Decision-1.0-Kai-0.6B",
			ModelEndpoint: ClassifierVLLMEndpoint{Address: "decision", Port: 8710},
		}},
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{ClassifierRules: []ClassifierSignalRule{rule}},
		},
	}
}

func validSystemOneClassifierRule() ClassifierSignalRule {
	return ClassifierSignalRule{
		Name:         "tone",
		Type:         ClassifierSignalTypeSystemOne,
		Model:        "decision-runtime",
		Labels:       []string{"formal", "casual", "hostile"},
		Instructions: "Which tone does this request use?",
	}
}

func TestValidateClassifierSignalContractsAcceptsSystemOne(t *testing.T) {
	if err := validateClassifierSignalContracts(systemOneClassifierConfig(validSystemOneClassifierRule())); err != nil {
		t.Fatalf("validateClassifierSignalContracts() error = %v", err)
	}
}

func TestValidateClassifierSignalContractsRejectsInvalidSystemOne(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*RouterConfig)
	}{
		{"missing model", func(c *RouterConfig) { c.ClassifierRules[0].Model = "" }},
		{"undeclared model", func(c *RouterConfig) { c.ClassifierRules[0].Model = "absent" }},
		{"model_path is local-only", func(c *RouterConfig) { c.ClassifierRules[0].ModelPath = "models/tone" }},
		{"use_cpu is local-only", func(c *RouterConfig) { c.ClassifierRules[0].UseCPU = true }},
		{"disable_rationale is llm-only", func(c *RouterConfig) { c.ClassifierRules[0].DisableRationale = true }},
		// Unlike sequence_classifier, the Choice contract needs the question text.
		{"instructions are required", func(c *RouterConfig) { c.ClassifierRules[0].Instructions = "" }},
		{"blank instructions", func(c *RouterConfig) { c.ClassifierRules[0].Instructions = "   " }},
		{"single label has no distribution", func(c *RouterConfig) { c.ClassifierRules[0].Labels = []string{"formal"} }},
		{"wrong role", func(c *RouterConfig) { c.ExternalModels[0].ModelRole = ModelRoleGuardrail }},
		// The request names the model explicitly, so the catalog entry must too.
		{"missing llm_model_name", func(c *RouterConfig) { c.ExternalModels[0].ModelName = "" }},
		{"missing endpoint port", func(c *RouterConfig) { c.ExternalModels[0].ModelEndpoint.Port = 0 }},
		{"unsupported endpoint protocol", func(c *RouterConfig) { c.ExternalModels[0].ModelEndpoint.Protocol = "grpc" }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := systemOneClassifierConfig(validSystemOneClassifierRule())
			tt.mutate(cfg)
			if err := validateClassifierSignalContracts(cfg); err == nil {
				t.Fatal("expected a validation error, got nil")
			}
		})
	}
}

// The Choice contract accepts at most 255 options, so a rule that declares more
// is rejected at load time rather than on every request.
func TestValidateClassifierSignalContractsRejectsSystemOneOverChoiceArity(t *testing.T) {
	rule := validSystemOneClassifierRule()
	rule.Labels = make([]string, systemOneMaxChoiceOptions+1)
	for i := range rule.Labels {
		rule.Labels[i] = fmt.Sprintf("label-%d", i)
	}
	err := validateClassifierSignalContracts(systemOneClassifierConfig(rule))
	if err == nil {
		t.Fatalf("expected a validation error for %d labels, got nil", len(rule.Labels))
	}
	if !strings.Contains(err.Error(), "at most") {
		t.Fatalf("error does not report the arity limit: %v", err)
	}
}

// The unsupported-type message is what tells an operator which types exist, so
// it has to name the new one.
func TestValidateClassifierSignalContractsUnsupportedTypeNamesSystemOne(t *testing.T) {
	rule := validSystemOneClassifierRule()
	rule.Type = "not-a-type"
	err := validateClassifierSignalContracts(systemOneClassifierConfig(rule))
	if err == nil {
		t.Fatal("expected a validation error for an unknown type, got nil")
	}
	if !strings.Contains(err.Error(), ClassifierSignalTypeSystemOne) {
		t.Fatalf("unsupported-type error does not list %q: %v", ClassifierSignalTypeSystemOne, err)
	}
}

// The shared backend validator gates every consumer, so the new protocol has to
// pass it rather than only the rule-level validator.
func TestRemoteClassifierBackendAcceptsSystemOneProtocol(t *testing.T) {
	backend := &RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPSystemOne,
		Model:    "decision-runtime",
		Contract: RemoteClassifierContractLabelDistribution,
	}
	if err := backend.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}
