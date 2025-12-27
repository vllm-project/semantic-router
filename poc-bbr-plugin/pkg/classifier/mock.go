/*
Package classifier provides a mock implementation for testing.
In production, this would wrap the actual vSR Candle/LoRA classifiers.
*/
package classifier

import (
	"strings"
	"time"
)

// MockSemanticRouterClassifier is a mock implementation for testing
type MockSemanticRouterClassifier struct {
	classifierEnabled bool
	piiEnabled        bool
	jailbreakEnabled  bool

	// Configurable thresholds
	ClassifierThreshold float32
	PIIThreshold        float32
	JailbreakThreshold  float32
}

// NewMockSemanticRouterClassifier creates a new mock classifier with all features enabled
func NewMockSemanticRouterClassifier() *MockSemanticRouterClassifier {
	return &MockSemanticRouterClassifier{
		classifierEnabled:   true,
		piiEnabled:          true,
		jailbreakEnabled:    true,
		ClassifierThreshold: 0.7,
		PIIThreshold:        0.8,
		JailbreakThreshold:  0.9,
	}
}

// Classify performs mock category classification based on keywords
func (m *MockSemanticRouterClassifier) Classify(text string) (CategoryResult, error) {
	start := time.Now()

	text = strings.ToLower(text)

	var category string
	var confidence float32

	switch {
	case strings.Contains(text, "code") || strings.Contains(text, "python") ||
		strings.Contains(text, "function") || strings.Contains(text, "programming"):
		category = "coding"
		confidence = 0.92
	case strings.Contains(text, "math") || strings.Contains(text, "calculate") ||
		strings.Contains(text, "equation") || strings.Contains(text, "derivative"):
		category = "math"
		confidence = 0.88
	case strings.Contains(text, "story") || strings.Contains(text, "write") ||
		strings.Contains(text, "creative") || strings.Contains(text, "poem"):
		category = "creative"
		confidence = 0.85
	case strings.Contains(text, "analyze") || strings.Contains(text, "think") ||
		strings.Contains(text, "reason") || strings.Contains(text, "logic"):
		category = "reasoning"
		confidence = 0.82
	default:
		category = "general"
		confidence = 0.65
	}

	return CategoryResult{
		Category:   category,
		Confidence: confidence,
		LatencyMs:  time.Since(start).Milliseconds(),
	}, nil
}

// IsEnabled returns whether the classifier is enabled
func (m *MockSemanticRouterClassifier) IsEnabled() bool {
	return m.classifierEnabled
}

// IsPIIEnabled returns whether PII detection is enabled
func (m *MockSemanticRouterClassifier) IsPIIEnabled() bool {
	return m.piiEnabled
}

// IsJailbreakEnabled returns whether jailbreak detection is enabled
func (m *MockSemanticRouterClassifier) IsJailbreakEnabled() bool {
	return m.jailbreakEnabled
}

// DetectPII performs mock PII detection based on patterns
func (m *MockSemanticRouterClassifier) DetectPII(text string) (PIIResult, error) {
	start := time.Now()

	var piiTypes []string
	text = strings.ToLower(text)

	// Simple pattern matching for mock purposes
	if strings.Contains(text, "@") && strings.Contains(text, ".com") {
		piiTypes = append(piiTypes, "EMAIL")
	}
	if strings.Contains(text, "phone") || strings.Contains(text, "555-") {
		piiTypes = append(piiTypes, "PHONE")
	}
	if strings.Contains(text, "ssn") || strings.Contains(text, "social security") {
		piiTypes = append(piiTypes, "SSN")
	}
	if strings.Contains(text, "credit card") || strings.Contains(text, "4111") {
		piiTypes = append(piiTypes, "CREDIT_CARD")
	}

	hasPII := len(piiTypes) > 0

	return PIIResult{
		HasPII:    hasPII,
		PIITypes:  piiTypes,
		Blocked:   hasPII, // Block if PII detected
		LatencyMs: time.Since(start).Milliseconds(),
	}, nil
}

// DetectJailbreak performs mock jailbreak detection
func (m *MockSemanticRouterClassifier) DetectJailbreak(text string) (JailbreakResult, error) {
	start := time.Now()

	text = strings.ToLower(text)

	// Simple pattern matching for mock jailbreak detection
	jailbreakPatterns := []struct {
		pattern    string
		threatType string
	}{
		{"ignore previous instructions", "prompt_injection"},
		{"ignore all instructions", "prompt_injection"},
		{"disregard your instructions", "prompt_injection"},
		{"bypass your rules", "jailbreak"},
		{"pretend you are", "roleplay_attack"},
		{"act as if you have no restrictions", "jailbreak"},
		{"dan mode", "jailbreak"},
		{"developer mode", "jailbreak"},
	}

	for _, pattern := range jailbreakPatterns {
		if strings.Contains(text, pattern.pattern) {
			return JailbreakResult{
				IsJailbreak: true,
				ThreatType:  pattern.threatType,
				Confidence:  0.95,
				Blocked:     true,
				LatencyMs:   time.Since(start).Milliseconds(),
			}, nil
		}
	}

	return JailbreakResult{
		IsJailbreak: false,
		ThreatType:  "",
		Confidence:  0.0,
		Blocked:     false,
		LatencyMs:   time.Since(start).Milliseconds(),
	}, nil
}

// ProcessAll runs all enabled classifiers and returns combined results
func (m *MockSemanticRouterClassifier) ProcessAll(text string) (CategoryResult, PIIResult, JailbreakResult, error) {
	var categoryResult CategoryResult
	var piiResult PIIResult
	var jailbreakResult JailbreakResult
	var err error

	if m.classifierEnabled {
		categoryResult, err = m.Classify(text)
		if err != nil {
			return categoryResult, piiResult, jailbreakResult, err
		}
	}

	if m.piiEnabled {
		piiResult, err = m.DetectPII(text)
		if err != nil {
			return categoryResult, piiResult, jailbreakResult, err
		}
	}

	if m.jailbreakEnabled {
		jailbreakResult, err = m.DetectJailbreak(text)
		if err != nil {
			return categoryResult, piiResult, jailbreakResult, err
		}
	}

	return categoryResult, piiResult, jailbreakResult, nil
}

// SetClassifierEnabled enables/disables the classifier
func (m *MockSemanticRouterClassifier) SetClassifierEnabled(enabled bool) {
	m.classifierEnabled = enabled
}

// SetPIIEnabled enables/disables PII detection
func (m *MockSemanticRouterClassifier) SetPIIEnabled(enabled bool) {
	m.piiEnabled = enabled
}

// SetJailbreakEnabled enables/disables jailbreak detection
func (m *MockSemanticRouterClassifier) SetJailbreakEnabled(enabled bool) {
	m.jailbreakEnabled = enabled
}
