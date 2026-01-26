package config

import (
	"testing"
)

func TestToLegacyRegistry_IncludesAliases(t *testing.T) {
	registry := ToLegacyRegistry()

	// Test BERT LoRA PII model paths
	piiLoraRepo := "LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model"
	piiLoraTests := []string{
		"models/mom-pii-classifier",
		"models/lora_pii_detector_bert-base-uncased_model",
		"models/pii-detector",
		"pii-detector",
	}
	for _, path := range piiLoraTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != piiLoraRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, piiLoraRepo, repo)
		}
	}

	// Test ModernBERT PII model paths
	piiModernBertRepo := "llm-semantic-router/mmbert-pii-detector-merged"
	piiModernBertTests := []string{
		"models/mom-mmbert-pii-detector",
		"models/pii_classifier_modernbert-base_presidio_token_model",
		"models/pii_classifier_modernbert-base_model",
		"models/pii_classifier_modernbert_model",
		"models/pii_classifier_modernbert_ai4privacy_token_model",
		"models/mmbert-pii-detector",
		"mmbert-pii-detector",
	}
	for _, path := range piiModernBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != piiModernBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, piiModernBertRepo, repo)
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
	jailbreakRepo := "LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model"
	jailbreakTests := []string{
		"models/mom-jailbreak-classifier",
		"models/jailbreak_classifier_modernbert-base_model",
		"models/jailbreak_classifier_modernbert_model",
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

	// Test finding by old alias (now maps to ModernBERT model)
	model = GetModelByPath("models/pii_classifier_modernbert-base_presidio_token_model")
	if model == nil {
		t.Fatal("Expected to find model by old alias path")
	}
	if model.LocalPath != "models/mom-mmbert-pii-detector-merged" {
		t.Errorf("Expected LocalPath to be models/mom-mmbert-pii-detector-merged, got %s", model.LocalPath)
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

func TestMmBertModelsInRegistry(t *testing.T) {
	registry := ToLegacyRegistry()

	// Test mmBERT intent classifier
	intentMmBertRepo := "llm-semantic-router/mmbert-intent-classifier-merged"
	intentMmBertTests := []string{
		"models/mom-mmbert-intent-classifier-merged",
		"models/mmbert-intent-classifier",
		"models/mmbert-domain-classifier",
		"models/mmbert-category-classifier",
		"mmbert-intent-classifier",
		"mmbert-domain-classifier",
		"mmbert-category-classifier",
	}
	for _, path := range intentMmBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != intentMmBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, intentMmBertRepo, repo)
		}
	}

	// Test mmBERT fact-check model
	factCheckMmBertRepo := "llm-semantic-router/mmbert-fact-check-merged"
	factCheckMmBertTests := []string{
		"models/mom-mmbert-fact-check-merged",
		"models/mmbert-fact-check",
		"models/mmbert-fact-check-detector",
		"models/mmbert-halugate-sentinel",
		"mmbert-fact-check",
		"mmbert-fact-check-detector",
		"mmbert-halugate-sentinel",
	}
	for _, path := range factCheckMmBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != factCheckMmBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, factCheckMmBertRepo, repo)
		}
	}

	// Test mmBERT jailbreak detector
	jailbreakMmBertRepo := "llm-semantic-router/mmbert-jailbreak-detector-merged"
	jailbreakMmBertTests := []string{
		"models/mom-mmbert-jailbreak-detector-merged",
		"models/mmbert-jailbreak-detector",
		"models/mmbert-jailbreak-classifier",
		"models/mmbert-prompt-guard",
		"mmbert-jailbreak-detector",
		"mmbert-jailbreak-classifier",
		"mmbert-prompt-guard",
	}
	for _, path := range jailbreakMmBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != jailbreakMmBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, jailbreakMmBertRepo, repo)
		}
	}

	// Test mmBERT PII detector
	piiMmBertRepo := "llm-semantic-router/mmbert-pii-detector-merged"
	piiMmBertTests := []string{
		"models/mom-mmbert-pii-detector-merged",
		"models/mom-mmbert-pii-detector",
		"models/mmbert-pii-detector",
		"mmbert-pii-detector",
	}
	for _, path := range piiMmBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != piiMmBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, piiMmBertRepo, repo)
		}
	}
}
