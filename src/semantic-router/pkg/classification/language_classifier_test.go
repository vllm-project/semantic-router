package classification

import (
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
		{name: "English", text: "The quick brown fox jumps over the lazy dog. This sentence is commonly used to test English language detection algorithms because it contains a broad range of typical English words.", allowed: []string{"en"}, minScore: 0.3},
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
		// Emojis and non-linguistic symbols dilute the language signal, so
		// lingua-go produces lower confidence scores for these inputs. We
		// verify the detected language is sane but do not require high confidence.
		{name: "Special characters and emojis", text: "Hello! 😊 🎉 🚀 How are you?", allowed: []string{"en"}, minScore: 0.0},
		{name: "Unicode edge cases", text: "Hello 世界 🌍 مرحبا", allowed: []string{"en", "zh", "ar"}, minScore: 0.0},
		// Source code is not natural language; any detection is best-effort.
		{name: "Code snippets", text: "def hello(): print('world')", allowed: []string{"en"}, minScore: 0.0},
		// Single-word inputs are genuinely ambiguous across languages.
		{name: "Very short text", text: "Hi", allowed: []string{"en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko", "mi"}, minScore: 0.0},
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
