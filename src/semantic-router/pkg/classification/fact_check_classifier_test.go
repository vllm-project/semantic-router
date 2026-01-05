package classification

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// getHalugateSentinelModelPath returns the model path from env or a default
func getHalugateSentinelModelPath() string {
	if path := os.Getenv("HALUGATE_SENTINEL_MODEL_PATH"); path != "" {
		return path
	}
	// Default path - relative to pkg/classification directory (test working dir)
	return "../../../../models/mom-halugate-sentinel"
}

// skipIfNoFactCheckModel skips the test if the halugate-sentinel model is not available
func skipIfNoFactCheckModel(t *testing.T) {
	modelPath := getHalugateSentinelModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Skipping test: halugate-sentinel model not found at %s. Set HALUGATE_SENTINEL_MODEL_PATH env var.", modelPath)
	}
}

// TestFactCheckClassifier_NilConfig tests that nil config returns nil classifier
func TestFactCheckClassifier_NilConfig(t *testing.T) {
	classifier, err := NewFactCheckClassifier(nil)
	if err != nil {
		t.Errorf("Unexpected error for nil config: %v", err)
	}
	if classifier != nil {
		t.Error("Expected nil classifier for nil config")
	}
}

// TestFactCheckClassifier_RequiresModelID tests that ModelID is required
func TestFactCheckClassifier_RequiresModelID(t *testing.T) {
	cfg := &config.FactCheckModelConfig{
		ModelID:   "", // No model configured
		Threshold: 0.7,
	}

	classifier, err := NewFactCheckClassifier(cfg)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	err = classifier.Initialize()
	if err == nil {
		t.Error("Expected error when ModelID is not configured")
	}
}

// TestFactCheckClassifier_EmptyText tests handling of empty text
func TestFactCheckClassifier_EmptyText(t *testing.T) {
	skipIfNoFactCheckModel(t)

	cfg := &config.FactCheckModelConfig{
		ModelID:   getHalugateSentinelModelPath(),
		Threshold: 0.7,
		UseCPU:    true,
	}

	classifier, err := NewFactCheckClassifier(cfg)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	err = classifier.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize classifier: %v", err)
	}

	result, err := classifier.Classify("")
	if err != nil {
		t.Errorf("Unexpected error for empty text: %v", err)
	}
	if result.NeedsFactCheck {
		t.Error("Empty text should not need fact checking")
	}
	if result.Confidence != 1.0 {
		t.Errorf("Expected confidence 1.0 for empty text, got %f", result.Confidence)
	}
}

// TestFactCheckClassifier_Initialize tests model initialization
func TestFactCheckClassifier_Initialize(t *testing.T) {
	skipIfNoFactCheckModel(t)

	cfg := &config.FactCheckModelConfig{
		ModelID:   getHalugateSentinelModelPath(),
		Threshold: 0.7,
		UseCPU:    true,
	}

	classifier, err := NewFactCheckClassifier(cfg)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	err = classifier.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize classifier: %v", err)
	}

	if !classifier.IsInitialized() {
		t.Error("Classifier should be initialized")
	}

	t.Log("halugate-sentinel model initialized successfully")
}

// TestFactCheckClassifier_FactCheckNeeded tests classification of prompts that need fact-checking
func TestFactCheckClassifier_FactCheckNeeded(t *testing.T) {
	skipIfNoFactCheckModel(t)

	cfg := &config.FactCheckModelConfig{
		ModelID:   getHalugateSentinelModelPath(),
		Threshold: 0.5, // Lower threshold for testing
		UseCPU:    true,
	}

	classifier, err := NewFactCheckClassifier(cfg)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	err = classifier.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize classifier: %v", err)
	}

	// Test prompts that should likely need fact-checking
	factCheckPrompts := []string{
		"When was the Eiffel Tower built?",
		"What is the population of Tokyo?",
		"Who invented the telephone?",
		"What year did World War II end?",
		"How tall is Mount Everest?",
	}

	t.Log("Testing prompts that should need fact-checking:")
	for _, prompt := range factCheckPrompts {
		result, err := classifier.Classify(prompt)
		if err != nil {
			t.Errorf("Classification failed for '%s': %v", prompt, err)
			continue
		}
		t.Logf("  '%s' -> needs_fact_check=%v, confidence=%.3f, label=%s",
			prompt, result.NeedsFactCheck, result.Confidence, result.Label)
	}
}

// TestFactCheckClassifier_NoFactCheckNeeded tests classification of prompts that don't need fact-checking
func TestFactCheckClassifier_NoFactCheckNeeded(t *testing.T) {
	skipIfNoFactCheckModel(t)

	cfg := &config.FactCheckModelConfig{
		ModelID:   getHalugateSentinelModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	classifier, err := NewFactCheckClassifier(cfg)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	err = classifier.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize classifier: %v", err)
	}

	// Test prompts that should NOT need fact-checking
	noFactCheckPrompts := []string{
		"Write a poem about the ocean",
		"Can you help me debug this Python code?",
		"Calculate 15 * 7 + 3",
		"What do you think about modern art?",
		"Translate 'hello' to Spanish",
	}

	t.Log("Testing prompts that should NOT need fact-checking:")
	for _, prompt := range noFactCheckPrompts {
		result, err := classifier.Classify(prompt)
		if err != nil {
			t.Errorf("Classification failed for '%s': %v", prompt, err)
			continue
		}
		t.Logf("  '%s' -> needs_fact_check=%v, confidence=%.3f, label=%s",
			prompt, result.NeedsFactCheck, result.Confidence, result.Label)
	}
}

// TestFactCheckResult_JSONSerialization tests that results can be serialized
func TestFactCheckResult_JSONSerialization(t *testing.T) {
	result := &FactCheckResult{
		NeedsFactCheck: true,
		Confidence:     0.85,
		Label:          FactCheckLabelNeeded,
	}

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("Failed to marshal result: %v", err)
	}

	var decoded FactCheckResult
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	if decoded.NeedsFactCheck != result.NeedsFactCheck {
		t.Error("NeedsFactCheck mismatch after serialization")
	}
	if decoded.Confidence != result.Confidence {
		t.Error("Confidence mismatch after serialization")
	}
	if decoded.Label != result.Label {
		t.Error("Label mismatch after serialization")
	}
}

// TestFactCheckClassifier_OpenAIPipeline tests the full pipeline with OpenAI-style messages
func TestFactCheckClassifier_OpenAIPipeline(t *testing.T) {
	skipIfNoFactCheckModel(t)

	cfg := &config.FactCheckModelConfig{
		ModelID:   getHalugateSentinelModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	classifier, err := NewFactCheckClassifier(cfg)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	err = classifier.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize classifier: %v", err)
	}

	// Simulate OpenAI-style user message
	userMessage := "What is the current population of China and how has it changed since 2000?"

	result, err := classifier.Classify(userMessage)
	if err != nil {
		t.Fatalf("Classification failed: %v", err)
	}

	t.Logf("OpenAI Pipeline test:")
	t.Logf("  User message: %s", userMessage)
	t.Logf("  Needs fact check: %v", result.NeedsFactCheck)
	t.Logf("  Confidence: %.3f", result.Confidence)
	t.Logf("  Label: %s", result.Label)

	// This is a factual question about real-world data, so it should likely need fact-checking
	if !result.NeedsFactCheck {
		t.Log("Note: Model classified this factual question as not needing fact-check")
	}
}

// TestFactCheckClassifier_Threshold tests threshold behavior
func TestFactCheckClassifier_Threshold(t *testing.T) {
	skipIfNoFactCheckModel(t)

	// Test with high threshold (0.9)
	highThresholdCfg := &config.FactCheckModelConfig{
		ModelID:   getHalugateSentinelModelPath(),
		Threshold: 0.9,
		UseCPU:    true,
	}

	classifier, err := NewFactCheckClassifier(highThresholdCfg)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	err = classifier.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize classifier: %v", err)
	}

	prompt := "When was the first iPhone released?"
	result, err := classifier.Classify(prompt)
	if err != nil {
		t.Fatalf("Classification failed: %v", err)
	}

	t.Logf("Threshold test (threshold=0.9):")
	t.Logf("  Prompt: %s", prompt)
	t.Logf("  Needs fact check: %v", result.NeedsFactCheck)
	t.Logf("  Confidence: %.3f", result.Confidence)
	t.Logf("  Label: %s", result.Label)
}
