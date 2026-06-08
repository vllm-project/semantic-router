package classification

import "testing"

func TestLanguageClassifier_ThresholdFiltering(t *testing.T) {
	classifier := newLanguageClassifierForTest(t)

	text := "Hola, ¿cómo estás? Me llamo Juan y vivo en Madrid. ¿De dónde eres tú? Esta es una pregunta en español sobre mi ubicación."

	// Low threshold should allow Spanish detection
	resLow, err := classifier.ClassifyWithThreshold(text, 0.5)
	if err != nil {
		t.Fatalf("classification failed: %v", err)
	}
	if resLow.LanguageCode != "es" {
		t.Fatalf("expected language 'es' for threshold 0.5, got %q", resLow.LanguageCode)
	}

	// Very high threshold should cause fallback to English
	resHigh, err := classifier.ClassifyWithThreshold(text, 0.99)
	if err != nil {
		t.Fatalf("classification failed: %v", err)
	}
	if resHigh.LanguageCode == "es" {
		t.Fatalf("expected fallback (not 'es') for threshold 0.99, got %q", resHigh.LanguageCode)
	}
}
