package classification

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func setupRealFactCheckClassifier(t *testing.T) *FactCheckClassifier {
	t.Helper()
	cfg := config.DefaultGlobalConfig().HallucinationMitigation.FactCheckModel
	cfg.ModelID = requireRealModel(t, "VLLM_SR_FACTCHECK_MODEL", cfg.ModelID)
	classifier, err := NewFactCheckClassifier(&cfg)
	if err != nil {
		t.Fatalf("build fact-check classifier: %v", err)
	}
	t.Cleanup(func() {
		if err := classifier.Close(); err != nil {
			t.Errorf("close fact-check classifier: %v", err)
		}
	})
	if err := classifier.Initialize(); err != nil {
		t.Fatalf("initialize fact-check classifier: %v", err)
	}
	if !classifier.IsInitialized() {
		t.Fatal("fact-check model is not initialized")
	}
	assertRealModelCPU(t, classifier.backend.handle.Capability())
	return classifier
}

// TestFactCheckClassifier_NilConfig tests that nil config returns nil classifier
func TestFactCheckClassifier_NilConfig(t *testing.T) {
	classifier, err := NewFactCheckClassifier(nil)
	if err != nil {
		t.Errorf("Unexpected error for nil config: %v", err)
	}
	if classifier != nil {
		t.Error("Expected nil classifier for nil config")
	}
}

// TestFactCheckClassifier_RequiresModelID tests that ModelID is required
func TestFactCheckClassifier_RequiresModelID(t *testing.T) {
	cfg := &config.FactCheckModelConfig{
		ModelID:   "", // No model configured
		Threshold: 0.7,
	}

	classifier, err := NewFactCheckClassifier(cfg)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := classifier.Close(); closeErr != nil {
			t.Errorf("Failed to close classifier: %v", closeErr)
		}
	})

	err = classifier.Initialize()
	if err == nil {
		t.Error("Expected error when ModelID is not configured")
	}
}

// FactCheck predicts whether a request needs external factual verification;
// it does not judge the truth of a claim or the correctness of an answer.
func TestFactCheckPolicyRealModel(t *testing.T) {
	classifier := setupRealFactCheckClassifier(t)
	defaultThreshold := classifier.config.Threshold
	cases := []struct {
		name          string
		text          string
		wantFactCheck bool
	}{
		{"factual request", "What year did World War II end?", true},
		{"creative request", "Write a poem about an imaginary purple dragon.", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := classifier.backend.Classify(context.Background(), tc.text)
			if err != nil {
				t.Fatalf("read owned fact-check distribution: %v", err)
			}
			assertRealModelDistribution(t, raw.Probabilities, 2)
			class, confidence := deriveArgmax(raw.Probabilities)
			// Include a forced below-threshold case even for a saturated fp32
			// prediction. The .95 operating point comes from canonical config.
			for _, threshold := range []float32{defaultThreshold, .01, 1.01} {
				classifier.config.Threshold = threshold
				result, err := classifier.Classify(context.Background(), tc.text)
				if err != nil {
					t.Fatalf("classify fact-check request: %v", err)
				}
				wantClass := 0
				if class == 1 && confidence >= threshold {
					wantClass = 1
				}
				wantLabel := classifier.mapping.IdxToLabel[map[int]string{0: "0", 1: "1"}[wantClass]]
				if result.NeedsFactCheck != (wantClass == 1) || result.Label != wantLabel ||
					!result.ConfidenceAvailable || result.PolicyDefault != "" ||
					math.Abs(float64(result.Confidence-raw.Probabilities[wantClass])) > 1e-5 {
					t.Fatalf("threshold=%g result=%+v disagrees with probabilities=%v", threshold, result, raw.Probabilities)
				}
				if threshold == defaultThreshold && result.NeedsFactCheck != tc.wantFactCheck {
					t.Errorf("published operating point: needs_fact_check=%v, want %v for %q; probabilities=%v", result.NeedsFactCheck, tc.wantFactCheck, tc.text, raw.Probabilities)
				}
				t.Logf("input=%q threshold=%g result=%+v probabilities=%v", tc.text, threshold, result, raw.Probabilities)
			}
		})
	}
}

func TestFactCheckEmptyInputRealModel(t *testing.T) {
	classifier := setupRealFactCheckClassifier(t)
	result, err := classifier.Classify(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if result.NeedsFactCheck || result.Label != FactCheckLabelNotNeeded || result.ConfidenceAvailable || result.Confidence != 0 || result.PolicyDefault != "empty_text" {
		t.Fatalf("empty input fabricated a model prediction: %+v", result)
	}
	// Also execute a nonempty request; this test cannot pass on policy defaults
	// alone without the selected model performing inference.
	if _, err := classifier.Classify(context.Background(), "Write a short poem."); err != nil {
		t.Fatalf("nonempty model inference: %v", err)
	}
}

// TestFactCheckResult_JSONSerialization tests that results can be serialized
func TestFactCheckResult_JSONSerialization(t *testing.T) {
	result := &FactCheckResult{
		NeedsFactCheck:      true,
		Confidence:          0.85,
		ConfidenceAvailable: true,
		Label:               FactCheckLabelNeeded,
	}

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("Failed to marshal result: %v", err)
	}

	var decoded FactCheckResult
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	if decoded.NeedsFactCheck != result.NeedsFactCheck {
		t.Error("NeedsFactCheck mismatch after serialization")
	}
	if decoded.Confidence != result.Confidence {
		t.Error("Confidence mismatch after serialization")
	}
	if decoded.Label != result.Label {
		t.Error("Label mismatch after serialization")
	}
}
