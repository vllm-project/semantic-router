package classification

import (
	"context"
	"reflect"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type guardProvenanceRecorder struct {
	inputs []string
}

func (r *guardProvenanceRecorder) Classify(_ context.Context, text string) (SequenceClassificationResult, error) {
	r.inputs = append(r.inputs, text)
	// Deliberately match every input: these tests verify provenance, not language
	// classification, and never need a model or security-quality fixture.
	return SequenceClassificationResult{Probabilities: []float32{0.1, 0.9}}, nil
}

func TestGuardProvenanceDispatch(t *testing.T) {
	for _, test := range []struct {
		name    string
		history bool
		input   *JailbreakInput
		want    []string
	}{
		{name: "empty projection never falls back", input: &JailbreakInput{}},
		{name: "trusted roles are not evidence", history: true, input: &JailbreakInput{
			Current: []JailbreakContent{{Role: "system", Text: "system-marker"}},
			History: []JailbreakContent{{Role: "developer", Text: "developer-marker"}, {Role: "assistant", Text: "assistant-marker"}},
		}},
		{name: "current untrusted pieces", input: &JailbreakInput{
			Current: []JailbreakContent{{Role: "user", Text: "current-user"}, {Role: "tool", Text: "current-tool"}},
			History: []JailbreakContent{{Role: "user", Text: "old-user"}},
		}, want: []string{"current-user", "current-tool"}},
		{name: "untrusted history and deduplication", history: true, input: &JailbreakInput{
			Current: []JailbreakContent{{Role: "user", Text: "current-user"}},
			History: []JailbreakContent{{Role: "user", Text: "old-user"}, {Role: "tool", Text: "old-tool"}, {Role: "tool", Text: "current-user"}},
		}, want: []string{"current-user", "old-user", "old-tool"}},
		{name: "flat text caller preserved", want: []string{"flat-text"}},
		{
			name: "flat text caller owns history scope", history: true,
			want: []string{"flat-text", "trusted-routing-context", "legacy-user"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.JailbreakRules = []config.JailbreakRule{{Name: "scope", Threshold: 0.5, IncludeHistory: test.history}}
			recorder := &guardProvenanceRecorder{}
			classifier := &Classifier{Config: cfg, jailbreakInference: recorder, JailbreakMapping: &JailbreakMapping{
				LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1}, IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"},
			}}
			results := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalConfidences: map[string]float64{}}
			dispatchers := classifier.buildPolicySignalDispatchers(results, &sync.Mutex{},
				func(string) string { return "flat-text" }, []string{"legacy-user"}, []string{"trusted-routing-context"},
				ConversationFacts{}, RequestFacts{JailbreakInput: test.input}, nil)
			for _, dispatcher := range dispatchers {
				if dispatcher.signalType == config.SignalTypeJailbreak {
					dispatcher.evaluate()
				}
			}
			if !reflect.DeepEqual(recorder.inputs, test.want) {
				t.Fatalf("Guard inputs = %v, want %v", recorder.inputs, test.want)
			}
			if results.JailbreakDetected != (len(test.want) > 0) {
				t.Fatalf("matched=%v with %d eligible pieces", results.JailbreakDetected, len(test.want))
			}
		})
	}
}
