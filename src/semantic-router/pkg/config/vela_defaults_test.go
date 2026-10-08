package config

import (
	"strings"
	"testing"
)

func TestBuiltInSignalsDefaultToVela2WithoutChangingInputPolicies(t *testing.T) {
	cfg := DefaultGlobalConfig()
	paths := map[string]string{
		"Domain":    cfg.CategoryModel.ModelID,
		"Guard":     cfg.PromptGuard.ModelID,
		"Safety":    cfg.SafetyModels.Safety.ModelID,
		"PII":       cfg.PIIModel.ModelID,
		"FactCheck": cfg.HallucinationMitigation.FactCheckModel.ModelID,
		"Halu":      cfg.HallucinationMitigation.HallucinationModel.ModelID,
		"Feedback":  cfg.FeedbackDetector.ModelID,
	}
	for task, path := range paths {
		if path != Vela2SignalModel {
			t.Fatalf("%s default = %s, want %s", task, path, Vela2SignalModel)
		}
	}
	if spec := GetModelByPath(cfg.MmBertModelPath); cfg.MmBertModelPath != "models/Vela-1.0-Encoder-307M-Embedding" || spec == nil || len(spec.Revision) != 40 {
		t.Fatalf("embeddings keep their pinned Vela 1.0 model, got %s", cfg.MmBertModelPath)
	}
	if cfg.CategoryModel.MaxSequenceLength != 0 || cfg.PIIModel.MaxSequenceLength != 0 || cfg.HallucinationMitigation.FactCheckModel.MaxSequenceLength != 0 || cfg.FeedbackDetector.MaxSequenceLength != 0 {
		t.Fatal("model migration changed the existing zero/512 input policy")
	}
	if cfg.EmbeddingConfig.FullContext || cfg.EmbeddingConfig.TargetLayer != 22 || cfg.EmbeddingConfig.TargetDimension != 0 {
		t.Fatal("embedding input/representation defaults changed")
	}
	// The 0.3B's thresholds keep the Vela 1.0 specialists' operating points on
	// the router signal suite's dev split (docs/records/vela2-router-signals.md).
	for name, got := range map[string][2]float32{
		"Guard":     {cfg.PromptGuard.Threshold, .75},
		"Domain":    {cfg.CategoryModel.Threshold, .28},
		"PII":       {cfg.PIIModel.Threshold, .01},
		"FactCheck": {cfg.HallucinationMitigation.FactCheckModel.Threshold, .93},
		"Feedback":  {cfg.FeedbackDetector.Threshold, .37},
	} {
		if got[0] != got[1] {
			t.Fatalf("%s threshold = %v, want the calibrated %v", name, got[0], got[1])
		}
	}
	if cfg.SafetyModels.Hazard.ModelID != "" {
		t.Fatal("Hazard must use an explicit binding with its artifact operating point")
	}
}

func TestVela1SystemModelsRestoreTheSpecialistsAndTheirOperatingPoints(t *testing.T) {
	vela1 := Vela1SystemModels()
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
global:
  model_catalog:
    system:
      safety: ` + vela1.Safety + `
      prompt_guard: ` + vela1.PromptGuard + `
      domain_classifier: ` + vela1.DomainClassifier + `
      pii_classifier: ` + vela1.PIIClassifier + `
      fact_check_classifier: ` + vela1.FactCheckClassifier + `
      hallucination_detector: ` + vela1.HallucinationDetector + `
      feedback_detector: ` + vela1.FeedbackDetector + `
    modules:
      prompt_guard:
        threshold: 0.6
`))
	if err != nil {
		t.Fatal(err)
	}
	for got, want := range map[string]string{
		cfg.CategoryModel.ModelID: vela1.DomainClassifier, cfg.PromptGuard.ModelID: vela1.PromptGuard,
		cfg.SafetyModels.Safety.ModelID: vela1.Safety, cfg.PIIModel.ModelID: vela1.PIIClassifier,
		cfg.HallucinationMitigation.FactCheckModel.ModelID:     vela1.FactCheckClassifier,
		cfg.HallucinationMitigation.HallucinationModel.ModelID: vela1.HallucinationDetector,
		cfg.FeedbackDetector.ModelID:                           vela1.FeedbackDetector,
	} {
		if got != want {
			t.Fatalf("restored module runs %s, want %s", got, want)
		}
	}
	// A threshold the document does not set follows its model; an explicit one stays.
	for name, got := range map[string][2]float32{
		"Guard":     {cfg.PromptGuard.Threshold, .6},
		"Domain":    {cfg.CategoryModel.Threshold, .5},
		"PII":       {cfg.PIIModel.Threshold, .9},
		"FactCheck": {cfg.HallucinationMitigation.FactCheckModel.Threshold, .95},
		"Feedback":  {cfg.FeedbackDetector.Threshold, .7},
	} {
		if got[0] != got[1] {
			t.Fatalf("%s threshold = %v, want %v", name, got[0], got[1])
		}
	}
}

func TestARemoteGuardKeepsItsPreviousThreshold(t *testing.T) {
	global := DefaultCanonicalGlobal()
	if err := resolveModuleModelRefs(&global); err != nil {
		t.Fatal(err)
	}
	global.ModelCatalog.Modules.PromptGuard.Backend = &RemoteClassifierBackend{Model: "guard-service"}
	raw, err := NewStructuredPayload(map[string]interface{}{"model_catalog": map[string]interface{}{
		"modules": map[string]interface{}{"prompt_guard": map[string]interface{}{"backend": map[string]interface{}{"model": "guard-service"}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	normalizeModuleOperatingPoints(&global, raw)
	if got := global.ModelCatalog.Modules.PromptGuard.Threshold; got != .5 {
		t.Fatalf("a remote Guard keeps 0.5, got %v", got)
	}
	if got := global.ModelCatalog.Modules.Classifier.Domain.Threshold; got != .28 {
		t.Fatalf("the local domain module keeps the 0.3B's 0.28, got %v", got)
	}
}

func TestVelaDefaultsPreserveExplicitLegacyModelsAndBudgets(t *testing.T) {
	raw := []byte(`version: v0.3
global:
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: models/mmbert-embed-32k-2d-matryoshka
        embedding_config:
          full_context: false
    system:
      domain_classifier: models/mmbert32k-intent-classifier-merged
      pii_classifier: models/mmbert32k-pii-detector-merged
      fact_check_classifier: models/mmbert32k-factcheck-classifier-merged
      feedback_detector: models/mmbert32k-feedback-detector-merged
    modules:
      classifier:
        domain:
          max_sequence_length: 0
          category_mapping_path: models/mmbert32k-intent-classifier-merged/category_mapping.json
        pii:
          max_sequence_length: 0
          pii_mapping_path: models/mmbert32k-pii-detector-merged/pii_type_mapping.json
      hallucination_mitigation:
        fact_check:
          max_sequence_length: 0
          threshold: 0.6
      feedback_detector:
        max_sequence_length: 0
`)
	cfg, err := ParseYAMLBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{cfg.CategoryModel.ModelID, cfg.PIIModel.ModelID, cfg.HallucinationMitigation.FactCheckModel.ModelID, cfg.FeedbackDetector.ModelID, cfg.MmBertModelPath} {
		if strings.Contains(path, "Vela") {
			t.Fatalf("explicit old model silently migrated: %s", path)
		}
		spec := GetModelByPath(path)
		if spec == nil || strings.Contains(spec.RepoID, "Vela") {
			t.Fatalf("old alias was retargeted: %s", path)
		}
	}
	if cfg.HallucinationMitigation.FactCheckModel.Threshold != float32(.6) || cfg.CategoryModel.MaxSequenceLength != 0 || cfg.PIIModel.MaxSequenceLength != 0 || cfg.FeedbackDetector.MaxSequenceLength != 0 {
		t.Fatal("explicit historical settings were overwritten")
	}
}

func TestVelaLongContextRemainsAnExplicitDeploymentChoice(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          full_context: true
    modules:
      classifier:
        domain:
          max_sequence_length: 32768
        pii:
          max_sequence_length: 0
      hallucination_mitigation:
        fact_check:
          max_sequence_length: 32768
      feedback_detector:
        max_sequence_length: 32768
      modality_detector:
        classifier:
          max_sequence_length: 32768
`))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.EmbeddingConfig.FullContext || cfg.CategoryModel.MaxSequenceLength != 32768 || cfg.HallucinationMitigation.FactCheckModel.MaxSequenceLength != 32768 || cfg.FeedbackDetector.MaxSequenceLength != 32768 || cfg.ModalityDetector.Classifier.MaxSequenceLength != 32768 {
		t.Fatal("explicit full-context settings were lost")
	}
	if cfg.PIIModel.MaxSequenceLength != 0 {
		t.Fatal("long deployment bypassed PII scanning")
	}
}
