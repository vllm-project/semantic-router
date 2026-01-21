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

	// Test Intent/Category model paths - BERT LoRA
	intentLoraRepo := "LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model"
	intentLoraTests := []string{
		"models/mom-domain-classifier",
		"models/category_classifier_modernbert-base_model",
		"models/lora_intent_classifier_bert-base-uncased_model",
		"models/domain-classifier",
		"domain-classifier",
	}
	for _, path := range intentLoraTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != intentLoraRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, intentLoraRepo, repo)
		}
	}

	// Test Intent/Category model paths - mmBERT LoRA
	intentMmBertLoraRepo := "llm-semantic-router/mmbert-intent-classifier-lora"
	intentMmBertLoraTests := []string{
		"models/mom-mmbert-intent-classifier-lora",
		"models/mmbert-intent-classifier-lora",
		"mmbert-intent-classifier-lora",
	}
	for _, path := range intentMmBertLoraTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != intentMmBertLoraRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, intentMmBertLoraRepo, repo)
		}
	}

	// Test Intent/Category model paths - mmBERT merged
	intentMmBertRepo := "llm-semantic-router/mmbert-intent-classifier-merged"
	intentMmBertTests := []string{
		"models/mom-mmbert-intent-classifier",
		"models/mmbert-intent-classifier-merged",
		"mmbert-intent-classifier",
	}
	for _, path := range intentMmBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != intentMmBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, intentMmBertRepo, repo)
		}
	}

	// Test Jailbreak/Security model paths - BERT LoRA
	jailbreakLoraRepo := "LLM-Semantic-Router/lora_jailbreak_classifier_bert-base-uncased_model"
	jailbreakLoraTests := []string{
		"models/mom-jailbreak-classifier",
		"models/jailbreak_classifier_modernbert-base_model",
		"models/jailbreak_classifier_modernbert_model",
		"models/lora_jailbreak_classifier_bert-base-uncased_model",
		"models/jailbreak-detector",
		"jailbreak-detector",
	}
	for _, path := range jailbreakLoraTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != jailbreakLoraRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, jailbreakLoraRepo, repo)
		}
	}

	// Test Jailbreak/Security model paths - mmBERT LoRA
	jailbreakMmBertLoraRepo := "llm-semantic-router/mmbert-jailbreak-detector-lora"
	jailbreakMmBertLoraTests := []string{
		"models/mom-mmbert-jailbreak-classifier-lora",
		"models/mmbert-jailbreak-detector-lora",
		"mmbert-jailbreak-detector-lora",
	}
	for _, path := range jailbreakMmBertLoraTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != jailbreakMmBertLoraRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, jailbreakMmBertLoraRepo, repo)
		}
	}

	// Test Jailbreak/Security model paths - mmBERT merged
	jailbreakMmBertRepo := "llm-semantic-router/mmbert-jailbreak-detector-merged"
	jailbreakMmBertTests := []string{
		"models/mom-mmbert-jailbreak-classifier",
		"models/mmbert-jailbreak-detector-merged",
		"mmbert-jailbreak-detector",
	}
	for _, path := range jailbreakMmBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != jailbreakMmBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, jailbreakMmBertRepo, repo)
		}
	}

	// Test PII Detection - mmBERT LoRA
	piiMmBertLoraRepo := "llm-semantic-router/mmbert-pii-detector-lora"
	piiMmBertLoraTests := []string{
		"models/mom-mmbert-pii-detector-lora",
		"models/mmbert-pii-detector-lora",
		"mmbert-pii-detector-lora",
	}
	for _, path := range piiMmBertLoraTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != piiMmBertLoraRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, piiMmBertLoraRepo, repo)
		}
	}

	// Test PII Detection - mmBERT merged
	piiMmBertRepo := "llm-semantic-router/mmbert-pii-detector-merged"
	piiMmBertTests := []string{
		"models/mom-mmbert-pii-detector",
		"models/mmbert-pii-detector-merged",
		"mmbert-pii-detector",
	}
	for _, path := range piiMmBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != piiMmBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, piiMmBertRepo, repo)
		}
	}

	// Test Hallucination Sentinel - mmBERT LoRA
	halugateMmBertLoraRepo := "llm-semantic-router/mmbert-fact-check-lora"
	halugateMmBertLoraTests := []string{
		"models/mom-mmbert-halugate-sentinel-lora",
		"models/mmbert-fact-check-lora",
		"mmbert-halugate-sentinel-lora",
		"mmbert-fact-check-lora",
	}
	for _, path := range halugateMmBertLoraTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != halugateMmBertLoraRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, halugateMmBertLoraRepo, repo)
		}
	}

	// Test Hallucination Sentinel - mmBERT merged
	halugateMmBertRepo := "llm-semantic-router/mmbert-fact-check-merged"
	halugateMmBertTests := []string{
		"models/mom-mmbert-halugate-sentinel",
		"models/mmbert-fact-check-merged",
		"mmbert-halugate-sentinel",
		"fact-check-sentinel",
	}
	for _, path := range halugateMmBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != halugateMmBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, halugateMmBertRepo, repo)
		}
	}

	// Test Feedback Detection - mmBERT LoRA
	feedbackMmBertLoraRepo := "llm-semantic-router/mmbert-feedback-detector-lora"
	feedbackMmBertLoraTests := []string{
		"models/mom-mmbert-feedback-detector-lora",
		"models/mmbert-feedback-detector-lora",
		"mmbert-feedback-detector-lora",
	}
	for _, path := range feedbackMmBertLoraTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != feedbackMmBertLoraRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, feedbackMmBertLoraRepo, repo)
		}
	}

	// Test Feedback Detection - mmBERT merged
	feedbackMmBertRepo := "llm-semantic-router/mmbert-feedback-detector-merged"
	feedbackMmBertTests := []string{
		"models/mom-mmbert-feedback-detector",
		"models/mmbert-feedback-detector-merged",
		"mmbert-feedback-detector",
	}
	for _, path := range feedbackMmBertTests {
		if repo, ok := registry[path]; !ok {
			t.Errorf("Expected %s to be in registry", path)
		} else if repo != feedbackMmBertRepo {
			t.Errorf("Expected %s to map to %s, got %s", path, feedbackMmBertRepo, repo)
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
