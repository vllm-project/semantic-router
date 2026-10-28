package v1alpha1

import (
	"encoding/json"
	"testing"
)

func candidateCount(candidates *ComplexityCandidates) int {
	if candidates == nil {
		return 0
	}
	return len(candidates.Candidates)
}

// A rule read by a remote scorer states boundaries in the model's own units
// and carries no candidates. It must round-trip without growing empty
// candidate blocks, and a negative cut point must survive as written.
func TestComplexityRulesConfigRoundTripsARemoteRule(t *testing.T) {
	rule := ComplexityRulesConfig{
		Name:      "needs_reasoning",
		HardAbove: "0.85",
		EasyBelow: "-0.25",
	}

	data, err := json.Marshal(rule)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal into map: %v", err)
	}
	for _, absent := range []string{"hard", "easy", "threshold", "hard_below", "easy_above"} {
		if _, present := decoded[absent]; present {
			t.Errorf("field %q should be omitted when unset, got %v", absent, decoded[absent])
		}
	}
	if decoded["hard_above"] != "0.85" || decoded["easy_below"] != "-0.25" {
		t.Fatalf("boundaries did not round-trip: %v", decoded)
	}

	var back ComplexityRulesConfig
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if back.Hard != nil || back.Easy != nil {
		t.Fatalf("candidates must stay nil for a remote rule: %#v", back)
	}
}

// The module block and its backend mirror the router's fields, so a value
// written on the CRD must come back under the router's spelling.
func TestComplexityModelConfigRoundTrip(t *testing.T) {
	deadline := 2500
	model := ComplexityModelConfig{
		Backend: &RemoteClassifierBackendConfig{
			Protocol:   "http_classify",
			Contract:   "score.v1",
			Model:      "difficulty-scorer",
			DeadlineMs: &deadline,
		},
	}

	data, err := json.Marshal(model)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var decoded map[string]map[string]interface{}
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal into map: %v", err)
	}
	backend := decoded["backend"]
	for key, want := range map[string]interface{}{
		"protocol":    "http_classify",
		"contract":    "score.v1",
		"model":       "difficulty-scorer",
		"deadline_ms": float64(2500),
	} {
		if backend[key] != want {
			t.Errorf("backend.%s = %v, want %v", key, backend[key], want)
		}
	}

	// A backend without a deadline must not emit one: the router's block
	// rejects unknown or null spellings rather than ignoring them.
	model.Backend.DeadlineMs = nil
	data, _ = json.Marshal(model)
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if _, present := decoded["backend"]["deadline_ms"]; present {
		t.Fatalf("deadline_ms should be omitted when unset: %v", decoded["backend"])
	}
}
