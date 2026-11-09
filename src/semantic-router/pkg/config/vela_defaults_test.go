package config

import (
	"strings"
	"testing"
)

func TestBuiltInSignalsDefaultToVela3WithoutChangingInputPolicies(t *testing.T) {
	cfg := DefaultGlobalConfig()
	consumers := map[string]string{
		"Domain":    "domain_classifier",
		"Guard":     "prompt_guard",
		"Safety":    "safety.unsafe",
		"PII":       "pii_classifier",
		"FactCheck": "fact_check_classifier",
		"Halu":      "hallucination_detector",
		"Feedback":  "feedback_detector",
	}
	for task, consumer := range consumers {
		if got := effectiveModel(t, &cfg, consumer); !sameModel(got, Vela3SignalModel) {
			t.Fatalf("%s default = %s, want %s", task, got, Vela3SignalModel)
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
	// vela3-fast's thresholds keep the Vela 1.0 specialists' operating points on
	// the router signal suite's dev split (docs/records/vela3-router-signals.md).
	for name, got := range map[string][2]float32{
		"Guard":     {cfg.PromptGuard.Threshold, .88},
		"Domain":    {cfg.CategoryModel.Threshold, .41},
		"PII":       {cfg.PIIModel.Threshold, .18},
		"FactCheck": {cfg.HallucinationMitigation.FactCheckModel.Threshold, .9999},
		"Feedback":  {cfg.FeedbackDetector.Threshold, .47},
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
	for consumer, want := range map[string]string{
		"domain_classifier": vela1.DomainClassifier, "prompt_guard": vela1.PromptGuard,
		"safety.unsafe": vela1.Safety, "pii_classifier": vela1.PIIClassifier,
		"fact_check_classifier":  vela1.FactCheckClassifier,
		"hallucination_detector": vela1.HallucinationDetector,
		"feedback_detector":      vela1.FeedbackDetector,
	} {
		if got := effectiveModel(t, cfg, consumer); !sameModel(got, want) {
			t.Fatalf("restored %s runs %s, want %s", consumer, got, want)
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
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
global:
  model_catalog:
    external:
      - name: guard-service
        model_role: guardrail
        llm_provider: vllm
        llm_endpoint: {address: guard.example.com, port: 8000}
    modules:
      prompt_guard:
        jailbreak_mapping_path: /tmp/jailbreak_mapping.json
        backend: {protocol: http_classify, model: guard-service}
`))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.PromptGuard.Threshold; got != .5 {
		t.Fatalf("a remote Guard keeps 0.5, got %v", got)
	}
	if got := cfg.CategoryModel.Threshold; got != .41 {
		t.Fatalf("the local domain module keeps vela3-fast's 0.41, got %v", got)
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
	paths := []string{cfg.MmBertModelPath}
	for _, consumer := range []string{"domain_classifier", "pii_classifier", "fact_check_classifier", "feedback_detector"} {
		paths = append(paths, effectiveModel(t, cfg, consumer))
	}
	for _, path := range paths {
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
