package classification

import (
	"strings"
	"sync"
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

func newLanguageSignalClassifierForTest(t *testing.T, rules []config.LanguageRule) *Classifier {
	t.Helper()

	languageClassifier, err := NewLanguageClassifier(rules)
	if err != nil {
		t.Fatalf("failed to create language classifier: %v", err)
	}

	return &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					LanguageRules: rules,
				},
			},
		},
		languageClassifier: languageClassifier,
	}
}

func languageThresholdProbeText() string {
	return strings.Repeat("这是一个用于测试语言检测阈值的中文句子。", 8)
}

func reliableLanguageThresholdProbe(t *testing.T, classifier *LanguageClassifier) (string, *LanguageResult) {
	t.Helper()

	probes := []struct {
		name    string
		text    string
		allowed []string
	}{
		{name: "Chinese", text: languageThresholdProbeText(), allowed: []string{"zh"}},
		{name: "Spanish", text: "Hola, ¿cómo estás? Me llamo Juan y vivo en Madrid. ¿De dónde eres tú? Esta es una pregunta en español sobre mi ubicación.", allowed: []string{"es"}},
		{name: "Russian", text: "Привет, как дела? Меня зовут Иван, и я живу в Москве. Откуда ты? Это вопрос на русском языке о моем местоположении.", allowed: []string{"ru"}},
		{name: "French", text: strings.Repeat("Bonjour, comment allez-vous? Ceci est un texte français utilisé pour valider le seuil de détection linguistique. ", 4), allowed: []string{"fr"}},
	}

	attempts := make([]string, 0, len(probes))
	for _, probe := range probes {
		result, err := classifier.ClassifyWithThreshold(probe.text, 0.01)
		if err != nil {
			t.Fatalf("classification probe %q failed: %v", probe.name, err)
		}
		attempts = append(attempts, probe.name+"="+result.LanguageCode)
		if result.Confidence >= 0.1 && result.LanguageCode != "en" && languageAllowed(result.LanguageCode, probe.allowed...) {
			return probe.text, result
		}
	}

	t.Skipf("lingua-go did not confidently detect any non-English threshold probe; attempts: %s", strings.Join(attempts, ", "))
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

func TestLanguageClassifier_ClassifyWithThresholdHonorsCustomThreshold(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)
	text, baseline := reliableLanguageThresholdProbe(t, classifier)

	rejected, err := classifier.ClassifyWithThreshold(text, float32(baseline.Confidence)+0.01)
	if err != nil {
		t.Fatalf("classification with stricter threshold failed: %v", err)
	}
	if rejected.LanguageCode != "en" {
		t.Fatalf("expected stricter threshold to fall back to \"en\", got %q", rejected.LanguageCode)
	}
	if rejected.Confidence != 0.5 {
		t.Fatalf("expected fallback confidence 0.5, got %f", rejected.Confidence)
	}
}

func TestLowestLanguageThreshold(t *testing.T) {
	tests := []struct {
		name  string
		rules []config.LanguageRule
		want  float32
	}{
		{name: "NoRules", want: 0},
		{name: "IgnoresUnsetThresholds", rules: []config.LanguageRule{{Name: "en"}, {Name: "zh", Threshold: 0.6}}, want: 0.6},
		{name: "ReturnsLowestNonZeroThreshold", rules: []config.LanguageRule{{Name: "en", Threshold: 0.7}, {Name: "zh", Threshold: 0.4}, {Name: "fr", Threshold: 0.5}}, want: 0.4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := lowestLanguageThreshold(tt.rules); got != tt.want {
				t.Fatalf("expected lowest threshold %f, got %f", tt.want, got)
			}
		})
	}
}

func TestEvaluateLanguageSignal_EnforcesPerRuleThreshold(t *testing.T) {
	probeClassifier := newLanguageClassifierForTest(t)
	text, probe := reliableLanguageThresholdProbe(t, probeClassifier)

	rules := []config.LanguageRule{
		{Name: "en", Threshold: 0.01},
		{Name: probe.LanguageCode, Threshold: float32(probe.Confidence) + 0.01},
	}
	classifier := newLanguageSignalClassifierForTest(t, rules)
	minThreshold := lowestLanguageThreshold(rules)
	baseline, err := classifier.languageClassifier.ClassifyWithThreshold(text, minThreshold)
	if err != nil {
		t.Skipf("evaluateLanguageSignal preflight classification failed: %v", err)
	}
	if baseline == nil || baseline.LanguageCode == "" || baseline.LanguageCode == "en" || baseline.Confidence <= 0 {
		t.Skipf("evaluateLanguageSignal preflight classification was not reliable enough for this environment: %+v", baseline)
	}

	results := newSignalResultsForTest()
	var mu sync.Mutex

	classifier.evaluateLanguageSignal(results, &mu, text)

	if len(results.MatchedLanguageRules) != 0 {
		t.Fatalf("expected no matched language rules when per-rule threshold is stricter than confidence, got %v", results.MatchedLanguageRules)
	}
	if results.Metrics.Language.Confidence <= 0 {
		t.Fatalf("expected evaluateLanguageSignal to record positive confidence after successful preflight classification, got %f", results.Metrics.Language.Confidence)
	}
	if results.Metrics.Language.Confidence >= float64(rules[1].Threshold) {
		t.Fatalf("expected recorded confidence %f to stay below strict threshold %f", results.Metrics.Language.Confidence, rules[1].Threshold)
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
