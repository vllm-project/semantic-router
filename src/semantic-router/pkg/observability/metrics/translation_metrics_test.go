package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestRecordTranslationWarning_IncrementsCounter(t *testing.T) {
	TranslationLossyTotal.Reset()

	RecordTranslationWarning("anthropic", "openai", "lossy", "top_k_drop_on_openai_backend")
	RecordTranslationWarning("anthropic", "openai", "lossy", "top_k_drop_on_openai_backend")
	RecordTranslationWarning("anthropic", "openai", "info", "coerced_string")

	got := testutil.ToFloat64(TranslationLossyTotal.WithLabelValues(
		"anthropic", "openai", "lossy", "top_k_drop_on_openai_backend",
	))
	if got != 2 {
		t.Fatalf("expected lossy/top_k counter=2, got %v", got)
	}

	got = testutil.ToFloat64(TranslationLossyTotal.WithLabelValues(
		"anthropic", "openai", "info", "coerced_string",
	))
	if got != 1 {
		t.Fatalf("expected info/coerced counter=1, got %v", got)
	}
}

func TestRecordTranslationWarning_EmptyLabelsBecomeUnknown(t *testing.T) {
	TranslationLossyTotal.Reset()

	RecordTranslationWarning("", "", "", "")

	got := testutil.ToFloat64(TranslationLossyTotal.WithLabelValues(
		"unknown", "unknown", "unknown", "unknown",
	))
	if got != 1 {
		t.Fatalf("expected unknown-labelled counter=1, got %v", got)
	}
}

func TestRecordTranslationWarning_LabelIndependence(t *testing.T) {
	TranslationLossyTotal.Reset()

	RecordTranslationWarning("anthropic", "openai", "lossy", "dropped")
	RecordTranslationWarning("openai", "anthropic", "lossy", "dropped")

	got := testutil.ToFloat64(TranslationLossyTotal.WithLabelValues(
		"anthropic", "openai", "lossy", "dropped",
	))
	if got != 1 {
		t.Fatalf("expected anthropic→openai counter=1, got %v", got)
	}
	got = testutil.ToFloat64(TranslationLossyTotal.WithLabelValues(
		"openai", "anthropic", "lossy", "dropped",
	))
	if got != 1 {
		t.Fatalf("expected openai→anthropic counter=1, got %v", got)
	}
}
