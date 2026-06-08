package classification

import (
	"math"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newLanguageClassifierForTest(t *testing.T) *LanguageClassifier {
	t.Helper()

	classifier, err := NewLanguageClassifier([]config.LanguageRule{
		{Name: "en"},
		{Name: "es"},
		{Name: "ru"},
		{Name: "zh"},
		{Name: "fr"},
	})
	if err != nil {
		t.Fatalf("failed to create language classifier: %v", err)
	}

	return classifier
}

func languageAllowed(result string, allowed ...string) bool {
	for _, candidate := range allowed {
		if result == candidate {
			return true
		}
	}
	return false
}

func TestLanguageClassifier_DetectsCommonLanguages(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)

	tests := []struct {
		name     string
		text     string
		allowed  []string
		minScore float64
	}{
		{name: "English", text: "Hello, how are you?", allowed: []string{"en"}, minScore: 0.3},
		{name: "Spanish", text: "Hola, ¿cómo estás? Me llamo Juan y vivo en Madrid. ¿De dónde eres tú? Esta es una pregunta en español sobre mi ubicación.", allowed: []string{"es", "en"}, minScore: 0.3},
		{name: "Russian", text: "Привет, как дела? Меня зовут Иван, и я живу в Москве. Откуда ты? Это вопрос на русском языке о моем местоположении.", allowed: []string{"ru", "en"}, minScore: 0.3},
		{name: "Chinese", text: "你好，世界", allowed: []string{"zh"}, minScore: 0.3},
		{name: "French", text: "Bonjour, comment allez-vous?", allowed: []string{"fr"}, minScore: 0.3},
		{name: "Japanese", text: "こんにちは、元気ですか？", allowed: []string{"ja"}, minScore: 0.3},
		{name: "German", text: "Guten Tag, wie geht es Ihnen? Ich heiße Hans und wohne in Berlin. Woher kommen Sie?", allowed: []string{"de"}, minScore: 0.3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := classifier.Classify(tt.text)
			if err != nil {
				t.Fatalf("classification failed: %v", err)
			}
			if !languageAllowed(result.LanguageCode, tt.allowed...) {
				t.Fatalf("expected one of %v, got %q", tt.allowed, result.LanguageCode)
			}
			if result.Confidence < tt.minScore {
				t.Fatalf("expected confidence >= %f, got %f", tt.minScore, result.Confidence)
			}
		})
	}
}

func TestLanguageClassifier_HandlesFallbackCases(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)

	tests := []struct {
		name               string
		text               string
		allowed            []string
		expectedConfidence float64
	}{
		{name: "Empty", text: "", allowed: []string{"en"}, expectedConfidence: 0.5},
		{name: "Whitespace", text: "   \n\t  ", allowed: []string{"en"}, expectedConfidence: 0.5},
		{name: "Numbers", text: "1234567890 9876543210", allowed: []string{"en"}, expectedConfidence: 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := classifier.Classify(tt.text)
			if err != nil {
				t.Fatalf("classification failed: %v", err)
			}
			if !languageAllowed(result.LanguageCode, tt.allowed...) {
				t.Fatalf("expected one of %v, got %q", tt.allowed, result.LanguageCode)
			}
			if result.Confidence != tt.expectedConfidence {
				t.Fatalf("expected confidence %f, got %f", tt.expectedConfidence, result.Confidence)
			}
		})
	}
}

func TestLanguageClassifier_HandlesMixedAndSpecialInputs(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)

	tests := []struct {
		name     string
		text     string
		allowed  []string
		minScore float64
	}{
		{name: "Mixed language", text: "Hello, bonjour, hola", allowed: []string{"en", "es", "fr"}, minScore: 0.0},
		{name: "Long English text", text: strings.Repeat("This is a very long English sentence that contains many words. ", 200), allowed: []string{"en"}, minScore: 0.3},
		{name: "Special characters and emojis", text: "Hello! 😊 🎉 🚀 How are you?", allowed: []string{"en"}, minScore: 0.3},
		{name: "Unicode edge cases", text: "Hello 世界 🌍 مرحبا", allowed: []string{"en", "zh", "ar"}, minScore: 0.3},
		{name: "Code snippets", text: "def hello(): print('world')", allowed: []string{"en", "unknown"}, minScore: 0.3},
		{name: "Very short text", text: "Hi", allowed: []string{"en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko"}, minScore: 0.3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := classifier.Classify(tt.text)
			if err != nil {
				t.Fatalf("classification failed: %v", err)
			}
			if !languageAllowed(result.LanguageCode, tt.allowed...) {
				t.Fatalf("expected one of %v, got %q", tt.allowed, result.LanguageCode)
			}
			if result.Confidence < tt.minScore {
				t.Fatalf("expected confidence >= %f, got %f", tt.minScore, result.Confidence)
			}
		})
	}
}

func TestClassifyWithThreshold_DefaultThreshold(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)
	text := "Hello, how are you?"

	result, err := classifier.ClassifyWithThreshold(text, 0)
	if err != nil {
		t.Fatalf("classification with implicit default threshold failed: %v", err)
	}

	expected, err := classifier.ClassifyWithThreshold(text, defaultLanguageThreshold)
	if err != nil {
		t.Fatalf("classification with explicit default threshold failed: %v", err)
	}

	if result.LanguageCode != expected.LanguageCode {
		t.Fatalf("expected language %q, got %q", expected.LanguageCode, result.LanguageCode)
	}
	if math.Abs(result.Confidence-expected.Confidence) > 1e-9 {
		t.Fatalf("expected confidence %f, got %f", expected.Confidence, result.Confidence)
	}
}

func TestClassifyWithThreshold_HighThresholdFallsBackToEnglish(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)

	result, err := classifier.ClassifyWithThreshold("Hello, bonjour, hola", 0.999)
	if err != nil {
		t.Fatalf("classification failed: %v", err)
	}
	if result.LanguageCode != "en" {
		t.Fatalf("expected fallback language %q, got %q", "en", result.LanguageCode)
	}
	if result.Confidence != 0.5 {
		t.Fatalf("expected fallback confidence %f, got %f", 0.5, result.Confidence)
	}
}

func TestClassifyWithThreshold_LowThresholdAllowsDetection(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)

	result, err := classifier.ClassifyWithThreshold("你好，世界，这是一个中文句子，用来测试语言检测。", 0.01)
	if err != nil {
		t.Fatalf("classification failed: %v", err)
	}
	if result.LanguageCode != "zh" {
		t.Fatalf("expected language %q, got %q", "zh", result.LanguageCode)
	}
	if result.Confidence < 0.01 {
		t.Fatalf("expected confidence >= %f, got %f", 0.01, result.Confidence)
	}
}

func TestLowestLanguageThreshold_NoRules(t *testing.T) {
	if got := lowestLanguageThreshold(nil); got != 0 {
		t.Fatalf("expected threshold %f, got %f", 0.0, got)
	}
}

func TestLowestLanguageThreshold_AllZeroThresholds(t *testing.T) {
	rules := []config.LanguageRule{{Name: "en"}, {Name: "es", Threshold: 0}, {Name: "fr"}}

	if got := lowestLanguageThreshold(rules); got != 0 {
		t.Fatalf("expected threshold %f, got %f", 0.0, got)
	}
}

func TestLowestLanguageThreshold_SingleRule(t *testing.T) {
	rules := []config.LanguageRule{{Name: "zh", Threshold: 0.65}}

	if got := lowestLanguageThreshold(rules); got != 0.65 {
		t.Fatalf("expected threshold %f, got %f", 0.65, got)
	}
}

func TestLowestLanguageThreshold_MultipleRulesReturnsMinimum(t *testing.T) {
	rules := []config.LanguageRule{
		{Name: "en", Threshold: 0.7},
		{Name: "es", Threshold: 0.2},
		{Name: "fr", Threshold: 0.5},
	}

	if got := lowestLanguageThreshold(rules); got != 0.2 {
		t.Fatalf("expected threshold %f, got %f", 0.2, got)
	}
}

func TestLowestLanguageThreshold_MixedZeroAndNonZero(t *testing.T) {
	rules := []config.LanguageRule{
		{Name: "en"},
		{Name: "es", Threshold: 0.45},
		{Name: "fr", Threshold: 0},
		{Name: "zh", Threshold: 0.15},
	}

	if got := lowestLanguageThreshold(rules); got != 0.15 {
		t.Fatalf("expected threshold %f, got %f", 0.15, got)
	}
}
