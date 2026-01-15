package classification

import (
	"github.com/abadojack/whatlanggo"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// LanguageClassifier implements language detection using whatlanggo library
// Supports 100+ languages with high accuracy
type LanguageClassifier struct {
	rules []config.LanguageRule
}

// LanguageResult represents the result of language classification
type LanguageResult struct {
	LanguageCode string  // Language code: "en", "es", "zh", "fr", etc.
	Confidence   float64 // Confidence score (0.0-1.0)
}

// NewLanguageClassifier creates a new language classifier
func NewLanguageClassifier(cfgRules []config.LanguageRule) (*LanguageClassifier, error) {
	return &LanguageClassifier{
		rules: cfgRules,
	}, nil
}

// Classify detects the language of the query using whatlanggo library
func (c *LanguageClassifier) Classify(text string) (*LanguageResult, error) {
	if text == "" {
		return &LanguageResult{
			LanguageCode: "en", // Default to English
			Confidence:   0.5,
		}, nil
	}

	// Detect language using whatlanggo
	info := whatlanggo.Detect(text)

	// Get ISO 639-1 code from whatlanggo
	languageCode := info.Lang.Iso6391()

	// Ensure confidence is in valid range
	confidence := info.Confidence
	if confidence > 1.0 {
		confidence = 1.0
	}
	if confidence < 0.0 {
		confidence = 0.0
	}

	// If confidence is too low or detection is not reliable, default to English
	if confidence < 0.3 || !info.IsReliable() || languageCode == "" {
		languageCode = "en"
		confidence = 0.5
	}

	logging.Infof("Language classification: code=%s, confidence=%.2f (whatlanggo: %s, %.2f, reliable=%v)",
		languageCode, confidence, info.Lang.String(), info.Confidence, info.IsReliable())

	return &LanguageResult{
		LanguageCode: languageCode,
		Confidence:   confidence,
	}, nil
}
