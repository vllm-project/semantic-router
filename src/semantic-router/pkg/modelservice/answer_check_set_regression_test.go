package modelservice

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

func TestExplicitSetProbabilityCoverageDecodeRegression(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		invalid    bool
	}{
		{"incomplete", `{"sets":{"topics":{"selected":[],"probabilities":{"billing":0.9}}}}`, true},
		{"undeclared_replacement", `{"sets":{"topics":{"selected":[],"probabilities":{"billing":0.9,"refunds":0.8}}}}`, true},
		{"extra_label", `{"sets":{"topics":{"selected":[],"probabilities":{"billing":0.9,"shipping":0.8,"refunds":0.2}}}}`, true},
		{"empty_probabilities", `{"sets":{"topics":{"selected":[],"probabilities":{}}}}`, true},
		{"complete_independent_control", `{"sets":{"topics":{"selected":[],"probabilities":{"billing":0.9,"shipping":0.8}}}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var body api.DecisionResponse
			if err := json.Unmarshal([]byte(tc.body), &body); err != nil {
				t.Fatal(err)
			}
			answer := decodeResponse(body, []Question{topicsQuestion("topics")}).Answers["topics"]
			if tc.invalid {
				if answer.Error != "invalid_model_output" {
					t.Errorf("expected invalid_model_output, got %q", answer.Error)
				}
				if len(answer.Probabilities) != 0 || len(answer.Selected) != 0 {
					t.Errorf("invalid answer retained evidence: %+v", answer)
				}
			} else if answer.Error != "" || answer.Probabilities["billing"] != 0.9 || answer.Probabilities["shipping"] != 0.8 {
				t.Errorf("complete independent probabilities (sum 1.7) must remain usable: %+v", answer)
			}
		})
	}
}
