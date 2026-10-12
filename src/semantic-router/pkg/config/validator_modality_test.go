package config

import (
	"strings"
	"testing"
)

func classifierModalityConfig(method string, bindings map[string]ModelBinding) *RouterConfig {
	cfg := &RouterConfig{}
	cfg.GlobalModelBindings = bindings
	cfg.ModalityDetector = ModalityDetectorConfig{Enabled: true, ModalityDetectionConfig: ModalityDetectionConfig{
		Method: method, ConfidenceThreshold: 0.6, LowerThresholdRatio: 0.7,
	}}
	return cfg
}

func TestModalityClassifierRunsVela2WithoutAModelPath(t *testing.T) {
	cfg := classifierModalityConfig(ModalityDetectionClassifier, nil)
	if err := validateGlobalModalityContracts(cfg); err != nil {
		t.Fatalf("the classifier method runs Vela 2.0 0.3B without a model_path: %v", err)
	}
	if model, useCPU, ok := cfg.ModalityClassifierModel(); !ok || model != Vela2SignalModel || !useCPU {
		t.Fatalf("classifier model = %q (cpu %t, ok %t), want Vela 2.0 0.3B on CPU", model, useCPU, ok)
	}
	cfg.ModalityDetector.Classifier = &ModalityClassifierConfig{ModelPath: "models/Vela-1.0-Encoder-307M-Modality", UseCPU: true}
	if model, _, _ := cfg.ModalityClassifierModel(); model != "models/Vela-1.0-Encoder-307M-Modality" {
		t.Fatalf("an explicit model_path must win, got %q", model)
	}
	cfg.ModalityDetector.ConfidenceThreshold = 0
	if err := validateGlobalModalityContracts(cfg); err == nil || !strings.Contains(err.Error(), "confidence_threshold is required") {
		t.Fatalf("the default model waives only the model_path, got %v", err)
	}
}

func TestHybridModalityNeedsAClassifierKeywordsOrABinding(t *testing.T) {
	keywordsOnly := classifierModalityConfig(ModalityDetectionHybrid, nil)
	if err := validateGlobalModalityContracts(keywordsOnly); err == nil || !strings.Contains(err.Error(), "requires at least one of classifier or keywords") {
		t.Fatalf("a hybrid detector with neither a classifier nor keywords is refused, got %v", err)
	}
	keywordsOnly.ModalityDetector.Keywords = []string{"draw"}
	if err := validateGlobalModalityContracts(keywordsOnly); err != nil {
		t.Fatal(err)
	}
	if _, _, ok := keywordsOnly.ModalityClassifierModel(); ok {
		t.Fatal("a hybrid detector without a classifier block stays keyword-only")
	}
	bound := classifierModalityConfig(ModalityDetectionHybrid, map[string]ModelBinding{
		"modality_detector": {Deployment: "vela2", Contract: RemoteClassifierContractLabelDistribution},
	})
	if err := validateGlobalModalityContracts(bound); err != nil {
		t.Fatalf("a modality_detector binding supplies the classifier: %v", err)
	}
}
