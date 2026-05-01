package classification

import (
	"strings"
	"sync"

	lingua "github.com/pemistahl/lingua-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// linguaDetector is a package-level singleton. Building the detector is
// expensive (~100ms); detection itself is fast (~1ms).
var (
	linguaOnce     sync.Once
	linguaDetector lingua.LanguageDetector
)

func init() {
	// Warm the detector at package initialization so the first language check
	// does not pay the one-time lingua setup cost on the request path.
	_ = getDetector()
}

func getDetector() lingua.LanguageDetector {
	linguaOnce.Do(func() {
		linguaDetector = lingua.NewLanguageDetectorBuilder().
			FromAllLanguages().
			Build()
	})
	return linguaDetector
}

const (
	// defaultLanguageThreshold is the minimum lingua-go per-language confidence
	// used when a LanguageRule does not specify an explicit threshold.
	// lingua-go distributes probability across 75+ languages, so raw scores are
	// lower than single-language classifiers; 0.3 is the practical reliability
	// floor analogous to whatlanggo's IsReliable() check.
	defaultLanguageThreshold = 0.3
)

// LanguageClassifier implements language detection using lingua-go library.
// lingua-go provides higher accuracy than whatlanggo, particularly for short
// texts in non-English languages where correct detection drives allow/block
// policy decisions.
type LanguageClassifier struct {
	rules []config.LanguageRule
}

// LanguageResult represents the result of language classification
type LanguageResult struct {
	LanguageCode string  // ISO 639-1 language code: "en", "es", "zh", "fr", etc.
	Confidence   float64 // Confidence score (0.0-1.0)
}

// NewLanguageClassifier creates a new language classifier
func NewLanguageClassifier(cfgRules []config.LanguageRule) (*LanguageClassifier, error) {
	return &LanguageClassifier{
		rules: cfgRules,
	}, nil
}

// Classify detects the language of the query using lingua-go with the
// built-in default confidence threshold (0.3).
func (c *LanguageClassifier) Classify(text string) (*LanguageResult, error) {
	return c.ClassifyWithThreshold(text, 0)
}

// ClassifyWithThreshold detects the language of the query using a caller-supplied
// minimum confidence threshold. A threshold of 0 uses defaultLanguageThreshold.
// This is the primary entry point used by evaluateLanguageSignal so that
// per-rule thresholds configured in LanguageRule.Threshold are honoured.
func (c *LanguageClassifier) ClassifyWithThreshold(text string, threshold float32) (*LanguageResult, error) {
	minConfidence := float64(threshold)
	if minConfidence <= 0 {
		minConfidence = defaultLanguageThreshold
	}

	if strings.TrimSpace(text) == "" {
		return &LanguageResult{
			LanguageCode: "en", // Default to English
			Confidence:   0.5,
		}, nil
	}

	detector := getDetector()
	lang, ok := detector.DetectLanguageOf(text)
	if !ok {
		return &LanguageResult{
			LanguageCode: "en",
			Confidence:   0.5,
		}, nil
	}

	code := strings.ToLower(lang.IsoCode639_1().String())
	if code == "" || code == "und" {
		return &LanguageResult{
			LanguageCode: "en",
			Confidence:   0.5,
		}, nil
	}

	confidence := detector.ComputeLanguageConfidence(text, lang)

	// Insufficient signal: fall back to English when confidence is below
	// the configured threshold (default 0.3, configurable per rule via
	// LanguageRule.Threshold to tune for deployment-specific accuracy needs).
	if confidence < minConfidence {
		return &LanguageResult{
			LanguageCode: "en",
			Confidence:   0.5,
		}, nil
	}

	logging.Infof("Language classification: code=%s, confidence=%.2f (lingua-go: %s, threshold=%.2f)",
		code, confidence, lang.String(), minConfidence)

	return &LanguageResult{
		LanguageCode: code,
		Confidence:   confidence,
	}, nil
}
