package config

import (
	"testing"
)

func TestToLegacyRegistry_IncludesAliases(t *testing.T) {
	registry := ToLegacyRegistry()

	// Test PII model paths
	piiRepo := "LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model"
	piiTests := []string{
		"models/mom-pii-classifier",
		"models/pii_classifier_modernbert-base_presidio_token_model",
		"models/pii_classifier_modernbert-base_model",
		"models/lora_pii_detector_bert-base-uncased_model",
		"models/pii-detector",
		"pii-detector",
	}
	for _, path := range piiTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != piiRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, piiRepo, repo)
		}
	}

	// Test Intent/Category model paths
	intentRepo := "LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model"
	intentTests := []string{
		"models/mom-domain-classifier",
		"models/category_classifier_modernbert-base_model",
		"models/lora_intent_classifier_bert-base-uncased_model",
		"models/domain-classifier",
		"domain-classifier",
	}
	for _, path := range intentTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != intentRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, intentRepo, repo)
		}
	}

	// Test Jailbreak/Security model paths
	jailbreakRepo := "LLM-Semantic-Router/lora_jailbreak_classifier_bert-base-uncased_model"
	jailbreakTests := []string{
		"models/mom-jailbreak-classifier",
		"models/jailbreak_classifier_modernbert-base_model",
		"models/lora_jailbreak_classifier_bert-base-uncased_model",
		"models/jailbreak-detector",
		"jailbreak-detector",
	}
	for _, path := range jailbreakTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != jailbreakRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, jailbreakRepo, repo)
		}
	}
}

func TestGetModelByPath_FindsByAlias(t *testing.T) {
	// Test finding by primary path
	model := GetModelByPath("models/mom-pii-classifier")
	if model == nil {
		t.Fatal("Expected to find model by primary path")
	}
	if model.LocalPath != "models/mom-pii-classifier" {
		t.Errorf("Expected LocalPath to be models/mom-pii-classifier, got %s", model.LocalPath)
	}

	// Test finding by old alias
	model = GetModelByPath("models/pii_classifier_modernbert-base_presidio_token_model")
	if model == nil {
		t.Fatal("Expected to find model by old alias path")
	}
	if model.LocalPath != "models/mom-pii-classifier" {
		t.Errorf("Expected LocalPath to be models/mom-pii-classifier, got %s", model.LocalPath)
	}

	// Test finding by short alias
	model = GetModelByPath("pii-detector")
	if model == nil {
		t.Fatal("Expected to find model by short alias")
	}
	if model.LocalPath != "models/mom-pii-classifier" {
		t.Errorf("Expected LocalPath to be models/mom-pii-classifier, got %s", model.LocalPath)
	}
}
