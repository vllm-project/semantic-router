package config

import "testing"

func sequenceClassifierConfig(rule ClassifierSignalRule) *RouterConfig {
	return &RouterConfig{
		ExternalModels: []ExternalModelConfig{{
			Name:          "toxicity-endpoint",
			ModelRole:     ModelRoleClassification,
			ModelEndpoint: ClassifierVLLMEndpoint{Address: "toxicity", Port: 8080},
		}},
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{ClassifierRules: []ClassifierSignalRule{rule}},
		},
	}
}

func validSequenceClassifierRule() ClassifierSignalRule {
	return ClassifierSignalRule{
		Name:   "toxicity",
		Type:   ClassifierSignalTypeSequenceClassifier,
		Model:  "toxicity-endpoint",
		Labels: []string{"benign", "toxic"},
	}
}

func TestValidateClassifierSignalContractsAcceptsSequenceClassifier(t *testing.T) {
	if err := validateClassifierSignalContracts(sequenceClassifierConfig(validSequenceClassifierRule())); err != nil {
		t.Fatalf("validateClassifierSignalContracts() error = %v", err)
	}
}

func TestValidateClassifierSignalContractsRejectsInvalidSequenceClassifier(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*ClassifierSignalRule)
	}{
		{"missing model", func(r *ClassifierSignalRule) { r.Model = "" }},
		{"undeclared model", func(r *ClassifierSignalRule) { r.Model = "absent" }},
		{"model_path is local-only", func(r *ClassifierSignalRule) { r.ModelPath = "models/toxicity" }},
		{"use_cpu is local-only", func(r *ClassifierSignalRule) { r.UseCPU = true }},
		{"instructions are llm-only", func(r *ClassifierSignalRule) { r.Instructions = "Choose." }},
		{"single label has no distribution", func(r *ClassifierSignalRule) { r.Labels = []string{"toxic"} }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rule := validSequenceClassifierRule()
			tt.mutate(&rule)
			if err := validateClassifierSignalContracts(sequenceClassifierConfig(rule)); err == nil {
				t.Fatal("expected a validation error, got nil")
			}
		})
	}
}

func TestValidateClassifierSignalContractsRejectsSequenceClassifierWrongRole(t *testing.T) {
	cfg := sequenceClassifierConfig(validSequenceClassifierRule())
	cfg.ExternalModels[0].ModelRole = ModelRoleGuardrail

	if err := validateClassifierSignalContracts(cfg); err == nil {
		t.Fatal("expected a model_role validation error, got nil")
	}
}

func TestValidateClassifierSignalContractsRejectsSequenceClassifierInvalidProtocol(t *testing.T) {
	cfg := sequenceClassifierConfig(validSequenceClassifierRule())
	cfg.ExternalModels[0].ModelEndpoint.Protocol = "grpc"

	if err := validateClassifierSignalContracts(cfg); err == nil {
		t.Fatal("expected an llm_endpoint.protocol validation error, got nil")
	}
}

func TestValidateClassifierSignalContractsRejectsUnknownType(t *testing.T) {
	rule := validSequenceClassifierRule()
	rule.Type = "sequence-classifier"

	if err := validateClassifierSignalContracts(sequenceClassifierConfig(rule)); err == nil {
		t.Fatal("expected an unsupported type error, got nil")
	}
}
