package services

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

func TestGuardRiskObservationEvalResponse(t *testing.T) {
	for _, risk := range []float64{0, .2} {
		signals := &classification.SignalResults{SignalValues: map[string]float64{"jailbreak:limit": risk}, SignalErrors: map[string]string{"jailbreak:partial": "jailbreak_evaluation_failed"}}
		response := (&ClassificationService{}).buildEvalResponse("sample", signals, nil, nil)
		raw, err := json.Marshal(response)
		if err != nil {
			t.Fatal(err)
		}
		var wire map[string]any
		if err = json.Unmarshal(raw, &wire); err != nil {
			t.Fatal(err)
		}
		values, ok := wire["signal_values"].(map[string]any)
		if !ok {
			t.Fatalf("missing values: %s", raw)
		}
		got, ok := values["jailbreak:limit"]
		if !ok || got != risk {
			t.Fatalf("missing actual risk: %s", raw)
		}
		if len(response.SignalConfidences) != 0 || response.SignalErrors["jailbreak:partial"] == "" {
			t.Fatal("observation changed confidence/errors")
		}
	}
}
