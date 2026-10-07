package config

import (
	"strings"
	"testing"
)

func classifierModalityConfig(bindings map[string]ModelBinding) *RouterConfig {
	cfg := &RouterConfig{}
	cfg.GlobalModelBindings = bindings
	cfg.ModalityDetector = ModalityDetectorConfig{Enabled: true, ModalityDetectionConfig: ModalityDetectionConfig{
		Method: ModalityDetectionClassifier, ConfidenceThreshold: 0.6,
	}}
	return cfg
}

func TestModalityClassifierNeedsAModelPathOrABinding(t *testing.T) {
	err := validateGlobalModalityContracts(classifierModalityConfig(nil))
	if err == nil || !strings.Contains(err.Error(), "requires classifier.model_path") {
		t.Fatalf("a classifier with neither a model_path nor a binding is refused, got %v", err)
	}
	bound := classifierModalityConfig(map[string]ModelBinding{
		"modality_detector": {Deployment: "vela2", Contract: RemoteClassifierContractLabelDistribution},
	})
	if err := validateGlobalModalityContracts(bound); err != nil {
		t.Fatalf("a modality_detector binding supplies the classifier: %v", err)
	}
	bound.ModalityDetector.ConfidenceThreshold = 0
	if err := validateGlobalModalityContracts(bound); err == nil || !strings.Contains(err.Error(), "confidence_threshold is required") {
		t.Fatalf("a binding waives only the model_path, got %v", err)
	}
}
