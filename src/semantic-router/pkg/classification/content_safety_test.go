package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Exercise the shipped safety policy through the real HTTP adapter, dispatcher
// and decision engine. Model accuracy has separately pinned evaluations.
func TestContentSafetyFragment(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	fragment, err := os.ReadFile(filepath.Join(filepath.Dir(file), "../../../../config/fragments/signal/safety/content-safety.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	fragment = append(fragment, []byte(`
providers:
  defaults:
    model: default-model
  models:
    - name: default-model
      backend_refs:
        - name: default-backend
          endpoint: localhost:8000
          protocol: http
`)...)
	for _, test := range []struct {
		name                   string
		unsafe, privacy        float64
		binaryFail, hazardFail bool
		decision               string
		wantError              bool
	}{
		{name: "safe", unsafe: 0.1, privacy: 0.1},
		{name: "hazard alone cannot block safe text", unsafe: 0.1, privacy: 0.99},
		{name: "unsafe privacy", unsafe: 0.9, privacy: 0.8, decision: "unsafe-privacy-content"},
		{name: "other unsafe content", unsafe: 0.9, privacy: 0.1, decision: "unsafe-content"},
		{name: "binary unavailable", binaryFail: true, privacy: 0.8, wantError: true},
		{name: "hazard unavailable for unsafe content", unsafe: 0.9, hazardFail: true, wantError: true},
		{name: "safe does not call unavailable hazard", unsafe: 0.1, hazardFail: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes(fragment)
			if err != nil {
				t.Fatalf("parse content-safety fragment: %v", err)
			}
			input := strings.Repeat("ordinary context ", 2048) + "final request"
			var binaryCalls, hazardCalls atomic.Int32
			for i := range cfg.ExternalModels {
				model := &cfg.ExternalModels[i]
				binary := model.Name == "content-safety-binary"
				labels := cfg.SafetyRules[0].EffectiveLabels()
				if !binary {
					labels = cfg.SafetyRules[1].Hazard.Labels
				}
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if binary {
						binaryCalls.Add(1)
					} else {
						hazardCalls.Add(1)
					}
					if r.Method != http.MethodPost || r.URL.Path != "/classify" {
						t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
					}
					var body struct {
						Inputs string `json:"inputs"`
					}
					if decodeErr := json.NewDecoder(r.Body).Decode(&body); decodeErr != nil || body.Inputs != input {
						t.Errorf("input changed before reaching model: %v", decodeErr)
					}
					if binary && test.binaryFail || !binary && test.hazardFail {
						http.Error(w, "unavailable", http.StatusServiceUnavailable)
						return
					}
					var scores []httpClassifyLabelScore
					// Reverse response order to verify named labels, not response positions.
					for j := len(labels) - 1; j >= 0; j-- {
						label := labels[j]
						score := 0.75 // Independent hazards need not sum to one.
						if binary {
							score = 1 - test.unsafe
							if label == "unsafe" {
								score = test.unsafe
							}
						} else if label == "privacy" {
							score = test.privacy
						}
						scores = append(scores, httpClassifyLabelScore{Label: label, Score: float32(score)})
					}
					_ = json.NewEncoder(w).Encode(scores)
				}))
				t.Cleanup(server.Close)
				model.ModelEndpoint = endpointForTestServer(t, server)
			}
			builder := classifierOptionBuilder{cfg: cfg}
			apply, err := builder.buildSafetyClassifiersOption()
			if err != nil {
				t.Fatal(err)
			}
			classifier := &Classifier{Config: cfg}
			apply(classifier)
			t.Cleanup(func() {
				for _, detector := range classifier.safetyClassifiers {
					_ = detector.Close()
				}
			})
			results := &SignalResults{SignalConfidences: map[string]float64{}, SignalValues: map[string]float64{}, SignalErrors: map[string]string{}, Metrics: &SignalMetricsCollection{}}
			classifier.evaluateSafetySignals(context.Background(), results, &sync.Mutex{}, input, map[string]bool{"safety:unsafe-content": true, "safety:unsafe-privacy": true})
			wantBinary := int32(1)
			if test.binaryFail {
				wantBinary = 2
			} // The HTTP connector retries one 503.
			if binaryCalls.Load() != wantBinary {
				t.Errorf("shared binary model called %d times, want %d", binaryCalls.Load(), wantBinary)
			}
			wantHazard := int32(0)
			if !test.binaryFail && test.unsafe >= 0.5 {
				wantHazard = 1
				if test.hazardFail {
					wantHazard = 2
				}
			}
			if hazardCalls.Load() != wantHazard {
				t.Errorf("hazard calls = %d, want %d", hazardCalls.Load(), wantHazard)
			}
			decision, err := classifier.EvaluateDecisionWithEngine(results)
			if (err != nil) != test.wantError {
				t.Fatalf("error = %v, wantError %v", err, test.wantError)
			}
			if test.wantError {
				return
			}
			if test.decision == "" {
				if decision != nil {
					t.Fatalf("unexpected decision: %+v", decision)
				}
			} else if decision == nil || decision.Decision.Name != test.decision {
				t.Fatalf("decision = %+v, want %s", decision, test.decision)
			}
		})
	}
}
