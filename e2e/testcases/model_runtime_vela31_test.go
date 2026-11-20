package testcases

import (
	"encoding/json"
	"math"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
)

func TestVela31AnswersRequireDescribedNoulChoiceParity(t *testing.T) {
	for _, tc := range []struct {
		name        string
		probability *float64
		yes         float64
		wantError   bool
	}{
		{"matching", vela31ProbabilityPointer(0.6), 0.6, false},
		{"missing Noul", nil, 0.6, true},
		{"descriptions dropped", vela31ProbabilityPointer(0.5), 0.6, true},
		{"below predicate", vela31ProbabilityPointer(0), 0, true},
		{"nonfinite", vela31ProbabilityPointer(math.NaN()), 0.6, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := modelruntime.DecisionsResponse{Answers: map[string]modelruntime.Answer{
				vela31Signal: {Noul: tc.probability},
				"equivalent": {Probabilities: map[string]float64{"no": 1 - tc.yes, "yes": tc.yes}},
			}}
			_, err := checkVela31Answers(response)
			if (err != nil) != tc.wantError {
				t.Fatalf("checkVela31Answers error = %v, want error %v", err, tc.wantError)
			}
		})
	}
}

func vela31ProbabilityPointer(value float64) *float64 { return &value }

func TestVela31PreviewRequiresSignalProbabilityAndRoute(t *testing.T) {
	for _, tc := range []struct {
		name      string
		payload   string
		wantError bool
	}{
		{"matching", `{"signal_values":{"systemone:vela31_described":0.6},"decision_result":{"decision_name":"vela31_described_route","matched_signals":{"systemone":["vela31_described"]}}}`, false},
		{"missing signal", `{"signal_values":{}}`, true},
		{"different signal", `{"signal_values":{"systemone:vela31_described":0.5}}`, true},
		{"unknown signal", `{"signal_errors":{"systemone:vela31_described":"question_malformed"}}`, true},
		{"fallback route", `{"signal_values":{"systemone:vela31_described":0.6},"decision_result":{"decision_name":"selector_route"}}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var preview routingPreview
			if err := json.Unmarshal([]byte(tc.payload), &preview); err != nil {
				t.Fatal(err)
			}
			if err := checkVela31Preview(preview, 0.6); (err != nil) != tc.wantError {
				t.Fatalf("checkVela31Preview error = %v, want error %v", err, tc.wantError)
			}
		})
	}
}

func TestVela31ChatRequiresSelectedRouteModelAndBackendCompletion(t *testing.T) {
	for _, tc := range []struct {
		name      string
		decision  string
		model     string
		signal    bool
		body      string
		wantError bool
	}{
		{"matching", vela31Decision, vela31Model, true, `{"object":"chat.completion","model":"math-expert","choices":[{"message":{"content":"Refund request received."}}]}`, false},
		{"wrong backend model", vela31Decision, vela31Model, true, `{"object":"chat.completion","model":"general-expert","choices":[{"message":{"content":"Refund request received."}}]}`, true},
		{"fallback route", "selector_route", vela31Model, true, `{}`, true},
		{"wrong model", vela31Decision, "general-expert", true, `{}`, true},
		{"missing signal", vela31Decision, vela31Model, false, `{}`, true},
		{"HTTP 200 only", vela31Decision, vela31Model, true, `{}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			headers := http.Header{}
			headers.Set("x-vsr-selected-decision", tc.decision)
			headers.Set("x-vsr-selected-model", tc.model)
			if tc.signal {
				headers.Set("x-vsr-trace", `signals=("systemone:vela31_described")`)
			}
			response := &localChatCompletionResponse{Headers: headers, Body: []byte(tc.body)}
			if err := checkVela31Chat(response); (err != nil) != tc.wantError {
				t.Fatalf("checkVela31Chat error = %v, want error %v", err, tc.wantError)
			}
		})
	}
}
