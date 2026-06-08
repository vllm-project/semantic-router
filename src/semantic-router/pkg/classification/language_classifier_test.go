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

func findThresholdSensitiveLanguageSample(t *testing.T, classifier *LanguageClassifier) (string, *LanguageResult) {
	t.Helper()

	samples := []string{
		"Hola, ¿cómo estás? Me llamo Juan y vivo en Madrid. ¿De dónde eres tú? Esta es una pregunta en español sobre mi ubicación.",
		"Привет, как дела? Меня зовут Иван, и я живу в Москве. Откуда ты? Это вопрос на русском языке о моем местоположении.",
		"Bonjour, comment allez-vous? Je m'appelle Marie et j'habite à Paris. Pouvez-vous me dire où se trouve la gare la plus proche ?",
	}

	for _, sample := range samples {
		result, err := classifier.Classify(sample)
		if err != nil {
			t.Fatalf("classification failed: %v", err)
		}
		if result.LanguageCode != "en" && result.Confidence > defaultLanguageThreshold+0.05 {
			return sample, result
		}
	}

	t.Fatal("failed to find a threshold-sensitive language sample")
	return "", nil
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

func TestLanguageClassifier_CustomThreshold(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)
	sample, defaultResult := findThresholdSensitiveLanguageSample(t, classifier)

	explicitDefaultResult, err := classifier.ClassifyWithThreshold(sample, 0)
	if err != nil {
		t.Fatalf("classification with default threshold failed: %v", err)
	}
	if explicitDefaultResult.LanguageCode != defaultResult.LanguageCode {
		t.Fatalf("expected explicit default threshold to keep %q, got %q", defaultResult.LanguageCode, explicitDefaultResult.LanguageCode)
	}
	if math.Abs(explicitDefaultResult.Confidence-defaultResult.Confidence) > 1e-9 {
		t.Fatalf("expected explicit default threshold confidence %f, got %f", defaultResult.Confidence, explicitDefaultResult.Confidence)
	}

	customThreshold := float32(defaultResult.Confidence - 0.02)
	customResult, err := classifier.ClassifyWithThreshold(sample, customThreshold)
	if err != nil {
		t.Fatalf("classification with custom threshold failed: %v", err)
	}
	if customResult.LanguageCode != defaultResult.LanguageCode {
		t.Fatalf("expected custom threshold %.2f to keep %q, got %q", customThreshold, defaultResult.LanguageCode, customResult.LanguageCode)
	}
	if math.Abs(customResult.Confidence-defaultResult.Confidence) > 1e-9 {
		t.Fatalf("expected custom threshold confidence %f, got %f", defaultResult.Confidence, customResult.Confidence)
	}
}

func TestLanguageClassifier_ThresholdEnforcement(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)
	sample, baseResult := findThresholdSensitiveLanguageSample(t, classifier)

	strictThreshold := float32(baseResult.Confidence + 0.01)
	result, err := classifier.ClassifyWithThreshold(sample, strictThreshold)
	if err != nil {
		t.Fatalf("classification with strict threshold failed: %v", err)
	}
	if result.LanguageCode != "en" {
		t.Fatalf("expected strict threshold %.2f to fall back to English, got %q with confidence %f", strictThreshold, result.LanguageCode, result.Confidence)
	}
	if result.Confidence != 0.5 {
		t.Fatalf("expected fallback confidence 0.5, got %f", result.Confidence)
	}
}

func TestLanguageClassifier_ClassifyWithThreshold(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)

	t.Run("ZeroThresholdUsesDefault", func(t *testing.T) {
		// threshold=0 falls back to defaultLanguageThreshold (0.3); well-known
		// English text must be detected reliably at that level.
		result, err := classifier.ClassifyWithThreshold("Hello, how are you doing today?", 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if result.LanguageCode != "en" {
			t.Errorf("expected en, got %q", result.LanguageCode)
		}
	})

	t.Run("LowThresholdAcceptsResult", func(t *testing.T) {
		// A very low threshold (0.01) should never filter out a high-confidence result.
		result, err := classifier.ClassifyWithThreshold(
			strings.Repeat("This is a long English sentence. ", 50), 0.01)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if result.LanguageCode != "en" {
			t.Errorf("expected en, got %q", result.LanguageCode)
		}
	})

	t.Run("ImpossibleThresholdFallsBackToEnglish", func(t *testing.T) {
		// A threshold of 1.0 can never be met; the classifier must fall back to "en".
		result, err := classifier.ClassifyWithThreshold("Hola mundo", 1.0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if result.LanguageCode != "en" {
			t.Errorf("expected en fallback, got %q", result.LanguageCode)
		}
		if result.Confidence != 0.5 {
			t.Errorf("expected fallback confidence 0.5, got %f", result.Confidence)
		}
	})

	t.Run("EmptyTextAlwaysReturnsEnglish", func(t *testing.T) {
		// Empty text must return the "en" fallback regardless of threshold.
		for _, th := range []float32{0, 0.5, 1.0} {
			result, err := classifier.ClassifyWithThreshold("", th)
			if err != nil {
				t.Fatalf("threshold %.1f: unexpected error: %v", th, err)
			}
			if result.LanguageCode != "en" {
				t.Errorf("threshold %.1f: expected en, got %q", th, result.LanguageCode)
			}
		}
	})
}
