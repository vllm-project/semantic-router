package config

import (
	"testing"
)

func TestToLegacyRegistry_IncludesAliases(t *testing.T) {
	registry := ToLegacyRegistry()

	assertRegistryAliases(t, registry,
		"LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model",
		"models/mom-pii-classifier",
		"models/lora_pii_detector_bert-base-uncased_model",
		"models/pii-detector",
		"pii-detector",
	)
	assertRegistryAliases(t, registry,
		"llm-semantic-router/mmbert-pii-detector-merged",
		"models/mom-mmbert-pii-detector",
		"models/pii_classifier_modernbert-base_presidio_token_model",
		"models/pii_classifier_modernbert-base_model",
		"models/pii_classifier_modernbert_model",
		"models/pii_classifier_modernbert_ai4privacy_token_model",
		"models/mmbert-pii-detector",
		"mmbert-pii-detector",
	)
	assertRegistryAliases(t, registry,
		"LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model",
		"models/mom-domain-classifier",
		"models/category_classifier_modernbert-base_model",
		"models/lora_intent_classifier_bert-base-uncased_model",
		"models/domain-classifier",
		"domain-classifier",
	)
	assertRegistryAliases(t, registry,
		"LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model",
		"models/mom-jailbreak-classifier",
		"models/jailbreak_classifier_modernbert-base_model",
		"models/jailbreak_classifier_modernbert_model",
		"models/lora_jailbreak_classifier_bert-base-uncased_model",
		"models/jailbreak-detector",
		"jailbreak-detector",
	)
}

func assertRegistryAliases(t *testing.T, registry map[string]string, wantRepo string, aliases ...string) {
	t.Helper()
	for _, alias := range aliases {
		repo, ok := registry[alias]
		if !ok {
			t.Errorf("Expected %s to be in registry", alias)
			continue
		}
		if repo != wantRepo {
			t.Errorf("Expected %s to map to %s, got %s", alias, wantRepo, repo)
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

	// Test finding by old alias (now maps to ModernBERT model)
	model = GetModelByPath("models/pii_classifier_modernbert-base_presidio_token_model")
	if model == nil {
		t.Fatal("Expected to find model by old alias path")
	}
	if model.LocalPath != "models/mom-mmbert-pii-detector" {
		t.Errorf("Expected LocalPath to be models/mom-mmbert-pii-detector, got %s", model.LocalPath)
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
