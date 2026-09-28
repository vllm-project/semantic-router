package config

import (
	"strings"
	"testing"
)

func TestVelaDefaultsUsePublishedPathsWithoutChangingInputPolicies(t *testing.T) {
	cfg := DefaultGlobalConfig()
	paths := map[string]string{
		"Domain":    cfg.CategoryModel.ModelID,
		"Guard":     cfg.PromptGuard.ModelID,
		"Safety":    cfg.SafetyModels.Safety.ModelID,
		"PII":       cfg.PIIModel.ModelID,
		"FactCheck": cfg.HallucinationMitigation.FactCheckModel.ModelID,
		"Feedback":  cfg.FeedbackDetector.ModelID,
		"Embedding": cfg.MmBertModelPath,
	}
	for task, path := range paths {
		want := "models/Vela-1.0-Encoder-307M-" + task
		if path != want {
			t.Fatalf("%s default = %s, want %s", task, path, want)
		}
		spec := GetModelByPath(path)
		if spec == nil || len(spec.Revision) != 40 {
			t.Fatalf("unversioned default %s", path)
		}
	}
	if cfg.CategoryModel.MaxSequenceLength != 0 || cfg.PIIModel.MaxSequenceLength != 0 || cfg.HallucinationMitigation.FactCheckModel.MaxSequenceLength != 0 || cfg.FeedbackDetector.MaxSequenceLength != 0 {
		t.Fatal("model migration changed the existing zero/512 input policy")
	}
	if cfg.EmbeddingConfig.FullContext || cfg.EmbeddingConfig.TargetLayer != 22 || cfg.EmbeddingConfig.TargetDimension != 0 {
		t.Fatal("embedding input/representation defaults changed")
	}
	if cfg.HallucinationMitigation.FactCheckModel.Threshold != float32(.95) {
		t.Fatal("FactCheck did not use its frozen Vela operating point")
	}
	if cfg.FeedbackDetector.Threshold != float32(.7) {
		t.Fatal("Feedback abstention changed")
	}
	if cfg.PromptGuard.Threshold != float32(.5) {
		t.Fatal("Guard did not use its frozen Vela operating point")
	}
	if cfg.SafetyModels.Hazard.ModelID != "" {
		t.Fatal("Hazard must use an explicit binding with its artifact operating point")
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
