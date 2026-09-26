package store

import (
	"encoding/json"
	"testing"
)

func TestLearningSuccessEstimateJSONPreservesCalibratedZeros(t *testing.T) {
	zero := 0.0
	raw, err := json.Marshal(LearningSuccessEstimate{
		Status:      "calibrated",
		Probability: &zero,
		Uncertainty: &zero,
		Coverage:    &zero,
	})
	if err != nil {
		t.Fatalf("marshal calibrated zeros: %v", err)
	}

	payload := decodeJSONObject(t, raw)
	for _, field := range []string{"probability", "uncertainty", "coverage"} {
		value, ok := payload[field]
		if !ok {
			t.Fatalf("calibrated JSON omitted %s: %s", field, raw)
		}
		got, ok := value.(float64)
		if !ok || got != 0 {
			t.Fatalf("calibrated %s = %#v, want 0: %s", field, value, raw)
		}
	}

	decoded := roundTripRecord(t, Record{
		Learning: &LearningDiagnostics{
			Adaptation: &LearningAdaptationDiagnostics{
				SuccessEstimates: map[string]LearningSuccessEstimate{
					"frontier": {
						Status:      "calibrated",
						Probability: &zero,
						Uncertainty: &zero,
						Coverage:    &zero,
					},
				},
			},
		},
	})
	got := decoded.Learning.Adaptation.SuccessEstimates["frontier"]
	assertCalibratedZeroPointers(t, got)
}

func TestLearningSuccessEstimateJSONOmitsUnsupportedMetrics(t *testing.T) {
	raw, err := json.Marshal(LearningSuccessEstimate{
		Status:         "unsupported",
		FallbackReason: "missing_calibration_artifact",
		SampleCount:    10,
	})
	if err != nil {
		t.Fatalf("marshal unsupported estimate: %v", err)
	}

	payload := decodeJSONObject(t, raw)
	for _, field := range []string{"probability", "uncertainty", "coverage"} {
		if _, ok := payload[field]; ok {
			t.Fatalf("unsupported JSON included %s: %s", field, raw)
		}
	}

	decoded := roundTripRecord(t, Record{
		Learning: &LearningDiagnostics{
			Adaptation: &LearningAdaptationDiagnostics{
				SuccessEstimates: map[string]LearningSuccessEstimate{
					"frontier": {
						Status:         "unsupported",
						FallbackReason: "missing_calibration_artifact",
						SampleCount:    10,
					},
				},
			},
		},
	})
	got := decoded.Learning.Adaptation.SuccessEstimates["frontier"]
	if got.Status != "unsupported" || got.SampleCount != 10 {
		t.Fatalf("unsupported estimate changed after round trip: %#v", got)
	}
	if got.Probability != nil || got.Uncertainty != nil || got.Coverage != nil {
		t.Fatalf("unsupported estimate invented calibrated metrics: %#v", got)
	}
}

func TestCloneLearningDiagnosticsPreservesCalibratedZeros(t *testing.T) {
	zero := 0.0
	original := &LearningDiagnostics{
		Adaptation: &LearningAdaptationDiagnostics{
			SuccessEstimates: map[string]LearningSuccessEstimate{
				"frontier": {
					Status:      "calibrated",
					Probability: &zero,
					Uncertainty: &zero,
					Coverage:    &zero,
				},
			},
		},
	}

	cloned := cloneLearningDiagnostics(original)
	got := cloned.Adaptation.SuccessEstimates["frontier"]
	assertCalibratedZeroPointers(t, got)

	mutated := 0.5
	got.Probability = &mutated
	cloned.Adaptation.SuccessEstimates["frontier"] = got
	if original.Adaptation.SuccessEstimates["frontier"].Probability == nil ||
		*original.Adaptation.SuccessEstimates["frontier"].Probability != 0 {
		t.Fatalf("clone mutated original calibrated zeros: %#v", original.Adaptation.SuccessEstimates["frontier"])
	}
}

func decodeJSONObject(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("unmarshal JSON object %s: %v", raw, err)
	}
	return payload
}

func assertCalibratedZeroPointers(t *testing.T, got LearningSuccessEstimate) {
	t.Helper()
	if got.Status != "calibrated" {
		t.Fatalf("expected calibrated status, got %#v", got)
	}
	if got.Probability == nil || *got.Probability != 0 {
		t.Fatalf("expected calibrated probability 0, got %#v", got)
	}
	if got.Uncertainty == nil || *got.Uncertainty != 0 {
		t.Fatalf("expected calibrated uncertainty 0, got %#v", got)
	}
	if got.Coverage == nil || *got.Coverage != 0 {
		t.Fatalf("expected calibrated coverage 0, got %#v", got)
	}
}
