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

// Classify detects the language of the query using lingua-go.
func (c *LanguageClassifier) Classify(text string) (*LanguageResult, error) {
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

	// lingua-go distributes probability across all 75+ supported languages, so
	// the raw per-language confidence is typically low for short or ambiguous
	// text. A value below 0.3 means insufficient signal — analogous to
	// whatlanggo's IsReliable() check — so fall back to English.
	if confidence < 0.3 {
		return &LanguageResult{
			LanguageCode: "en",
			Confidence:   0.5,
		}, nil
	}

	logging.Infof("Language classification: code=%s, confidence=%.2f (lingua-go: %s)",
		code, confidence, lang.String())

	return &LanguageResult{
		LanguageCode: code,
		Confidence:   confidence,
	}, nil
}
