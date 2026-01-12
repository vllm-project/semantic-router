//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

// Package classifier provides real ML-based classifiers using Candle bindings
package classifier

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// CandleSemanticRouterClassifier implements SemanticRouterClassifier using real ML models
type CandleSemanticRouterClassifier struct {
	mu sync.RWMutex

	// Enabled flags
	classifierEnabled bool
	piiEnabled        bool
	jailbreakEnabled  bool

	// Model paths (from HuggingFace cache)
	intentModelPath    string
	jailbreakModelPath string
	piiModelPath       string

	// Initialization status
	initialized bool
}

// CandleClassifierConfig holds configuration for the Candle classifier
type CandleClassifierConfig struct {
	// Model paths - use HuggingFace model IDs or local paths
	IntentModelPath    string
	JailbreakModelPath string
	PIIModelPath       string

	// Feature flags
	EnableClassifier bool
	EnablePII        bool
	EnableJailbreak  bool

	// Use CPU instead of GPU
	UseCPU bool
}

// DefaultCandleConfig returns a default configuration using local model paths
func DefaultCandleConfig() CandleClassifierConfig {
	// Try to find models directory - check common locations
	modelsPath := findModelsPath()

	return CandleClassifierConfig{
		IntentModelPath:    modelsPath + "/category_classifier_modernbert-base_model",
		JailbreakModelPath: modelsPath + "/jailbreak_classifier_modernbert-base_model",
		PIIModelPath:       modelsPath + "/pii_classifier_modernbert-base_presidio_token_model",
		EnableClassifier:   true,
		EnablePII:          true,
		EnableJailbreak:    true,
		UseCPU:             true, // Default to CPU for compatibility
	}
}

// findModelsPath searches for the models directory in common locations
func findModelsPath() string {
	// Check environment variable first
	if envPath := os.Getenv("VSR_MODELS_PATH"); envPath != "" {
		return envPath
	}

	// Common relative paths to try
	paths := []string{
		"models",          // Running from semantic-router root
		"../models",       // Running from poc-bbr-plugin
		"../../models",    // Running from poc-bbr-plugin/tests or poc-bbr-plugin/cmd/*
		"../../../models", // Running from deeper subdirectory
	}

	for _, p := range paths {
		if fileExists(p + "/category_classifier_modernbert-base_model/config.json") {
			return p
		}
	}

	// Default fallback
	return "../models"
}

// Helper to check if file exists
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// NewCandleSemanticRouterClassifier creates a new classifier using Candle ML models
func NewCandleSemanticRouterClassifier(config CandleClassifierConfig) (*CandleSemanticRouterClassifier, error) {
	c := &CandleSemanticRouterClassifier{
		classifierEnabled:  config.EnableClassifier,
		piiEnabled:         config.EnablePII,
		jailbreakEnabled:   config.EnableJailbreak,
		intentModelPath:    config.IntentModelPath,
		jailbreakModelPath: config.JailbreakModelPath,
		piiModelPath:       config.PIIModelPath,
	}

	// Initialize intent classifier
	if config.EnableClassifier {
		if err := candle_binding.InitModernBertClassifier(config.IntentModelPath, config.UseCPU); err != nil {
			return nil, fmt.Errorf("failed to initialize intent classifier: %w", err)
		}
	}

	// Initialize jailbreak classifier
	if config.EnableJailbreak {
		if err := candle_binding.InitModernBertJailbreakClassifier(config.JailbreakModelPath, config.UseCPU); err != nil {
			return nil, fmt.Errorf("failed to initialize jailbreak classifier: %w", err)
		}
	}

	// Initialize PII classifier
	if config.EnablePII {
		if err := candle_binding.InitModernBertPIITokenClassifier(config.PIIModelPath, config.UseCPU); err != nil {
			return nil, fmt.Errorf("failed to initialize PII classifier: %w", err)
		}
	}

	c.initialized = true
	return c, nil
}

// Intent category mappings (from model training)
var intentCategories = []string{
	"general",
	"coding",
	"math",
	"creative",
	"analysis",
	"research",
}

// Classify performs intent classification
func (c *CandleSemanticRouterClassifier) Classify(text string) (CategoryResult, error) {
	start := time.Now()

	if !c.classifierEnabled || !c.initialized {
		return CategoryResult{
			Category:   "general",
			Confidence: 0.5,
			LatencyMs:  time.Since(start).Milliseconds(),
		}, nil
	}

	result, err := candle_binding.ClassifyModernBertText(text)
	if err != nil {
		return CategoryResult{}, fmt.Errorf("classification failed: %w", err)
	}

	// Map class index to category name
	category := "general"
	if result.Class >= 0 && result.Class < len(intentCategories) {
		category = intentCategories[result.Class]
	}

	return CategoryResult{
		Category:   category,
		Confidence: result.Confidence,
		LatencyMs:  time.Since(start).Milliseconds(),
	}, nil
}

// DetectPII performs PII detection using token classification
func (c *CandleSemanticRouterClassifier) DetectPII(text string) (PIIResult, error) {
	start := time.Now()

	if !c.piiEnabled || !c.initialized {
		return PIIResult{
			HasPII:    false,
			PIITypes:  []string{},
			Blocked:   false,
			LatencyMs: time.Since(start).Milliseconds(),
		}, nil
	}

	// Use token classification for PII
	// Note: This requires the model config path for label mapping
	configPath := c.piiModelPath + "/config.json"
	result, err := candle_binding.ClassifyModernBertPIITokens(text, configPath)
	if err != nil {
		// Fall back to sequence classification
		seqResult, seqErr := candle_binding.ClassifyModernBertPIIText(text)
		if seqErr != nil {
			return PIIResult{}, fmt.Errorf("PII detection failed: %w", err)
		}

		hasPII := seqResult.Class == 1 // Class 1 = PII detected
		return PIIResult{
			HasPII:    hasPII,
			PIITypes:  []string{},
			Blocked:   hasPII,
			LatencyMs: time.Since(start).Milliseconds(),
		}, nil
	}

	// Extract unique PII types from entities
	piiTypeMap := make(map[string]bool)
	for _, entity := range result.Entities {
		// Clean up entity type (remove B-, I- prefixes)
		entityType := strings.TrimPrefix(entity.EntityType, "B-")
		entityType = strings.TrimPrefix(entityType, "I-")
		if entityType != "O" && entityType != "" {
			piiTypeMap[entityType] = true
		}
	}

	var piiTypes []string
	for piiType := range piiTypeMap {
		piiTypes = append(piiTypes, piiType)
	}

	hasPII := len(piiTypes) > 0

	return PIIResult{
		HasPII:    hasPII,
		PIITypes:  piiTypes,
		Blocked:   hasPII,
		LatencyMs: time.Since(start).Milliseconds(),
	}, nil
}

// DetectJailbreak performs jailbreak detection
func (c *CandleSemanticRouterClassifier) DetectJailbreak(text string) (JailbreakResult, error) {
	start := time.Now()

	if !c.jailbreakEnabled || !c.initialized {
		return JailbreakResult{
			IsJailbreak: false,
			ThreatType:  "",
			Confidence:  0.0,
			Blocked:     false,
			LatencyMs:   time.Since(start).Milliseconds(),
		}, nil
	}

	result, err := candle_binding.ClassifyModernBertJailbreakText(text)
	if err != nil {
		return JailbreakResult{}, fmt.Errorf("jailbreak detection failed: %w", err)
	}

	// Class 1 = jailbreak detected
	isJailbreak := result.Class == 1
	threatType := ""
	if isJailbreak {
		threatType = "prompt_injection"
	}

	return JailbreakResult{
		IsJailbreak: isJailbreak,
		ThreatType:  threatType,
		Confidence:  result.Confidence,
		Blocked:     isJailbreak,
		LatencyMs:   time.Since(start).Milliseconds(),
	}, nil
}

// ProcessAll runs all enabled classifiers and returns combined results
func (c *CandleSemanticRouterClassifier) ProcessAll(text string) (CategoryResult, PIIResult, JailbreakResult, error) {
	var categoryResult CategoryResult
	var piiResult PIIResult
	var jailbreakResult JailbreakResult
	var err error

	// Run classifiers in parallel for better latency
	var wg sync.WaitGroup
	var catErr, piiErr, jbErr error

	if c.classifierEnabled {
		wg.Add(1)
		go func() {
			defer wg.Done()
			categoryResult, catErr = c.Classify(text)
		}()
	}

	if c.piiEnabled {
		wg.Add(1)
		go func() {
			defer wg.Done()
			piiResult, piiErr = c.DetectPII(text)
		}()
	}

	if c.jailbreakEnabled {
		wg.Add(1)
		go func() {
			defer wg.Done()
			jailbreakResult, jbErr = c.DetectJailbreak(text)
		}()
	}

	wg.Wait()

	// Return first error encountered
	if catErr != nil {
		err = catErr
	} else if piiErr != nil {
		err = piiErr
	} else if jbErr != nil {
		err = jbErr
	}

	return categoryResult, piiResult, jailbreakResult, err
}

// IsEnabled returns whether the classifier is enabled
func (c *CandleSemanticRouterClassifier) IsEnabled() bool {
	return c.classifierEnabled && c.initialized
}

// IsPIIEnabled returns whether PII detection is enabled
func (c *CandleSemanticRouterClassifier) IsPIIEnabled() bool {
	return c.piiEnabled && c.initialized
}

// IsJailbreakEnabled returns whether jailbreak detection is enabled
func (c *CandleSemanticRouterClassifier) IsJailbreakEnabled() bool {
	return c.jailbreakEnabled && c.initialized
}
