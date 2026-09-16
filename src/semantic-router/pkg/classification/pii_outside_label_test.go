package classification

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A remote provider that reports the outside label "O" as a span must be
// rejected before the span can reach PIIEntities, the masking input, or a
// routing decision (Xunzhuo's review on #3498: the native backend drops class
// zero, the remote path must not turn it into an entity).
func TestRemoteOutsideSpanNeverBecomesAnEntity(t *testing.T) {
	const text = "John Smith called yesterday."
	_, extCfg := newTokenSpansServer(t, func(string) any {
		return []map[string]any{{"label": "O", "text": "called", "score": 0.99, "start": 11, "end": 17}}
	})
	backend, err := newPIIHTTPBackend(extCfg, testPIIMapping(), 0)
	if err != nil {
		t.Fatal(err)
	}

	newClassifier := func(onError string) *Classifier {
		cfg := &config.RouterConfig{}
		cfg.PIIModel.ModelID = "remote"
		cfg.PIIModel.Threshold = 0.5
		cfg.PIIModel.OnError = onError
		cfg.PIIMappingPath = "test-pii-mapping-path"
		cfg.PIIRules = []config.PIIRule{{Name: "no_pii", Threshold: 0.5}}
		c, err := newClassifierWithOptions(cfg, withPII(testPIIMapping(), nil, backend))
		if err != nil {
			t.Fatal(err)
		}
		return c
	}

	t.Run("detection path returns an error, not an O entity", func(t *testing.T) {
		detections, err := newClassifier(config.OnErrorAllow).ClassifyPIIWithDetails(context.Background(), text)
		if err == nil || !strings.Contains(err.Error(), "outside label") {
			t.Fatalf("want an outside-label error, got err=%v detections=%v", err, detections)
		}
		if len(detections) != 0 {
			t.Fatalf("no detections may be returned, got %v", detections)
		}
	})

	for _, tc := range []struct {
		onError      string
		wantDetected bool
	}{
		{config.OnErrorAllow, false},
		{config.OnErrorBlock, true},
	} {
		t.Run("signal path with on_error "+tc.onError, func(t *testing.T) {
			results := &SignalResults{
				Metrics:           &SignalMetricsCollection{},
				SignalConfidences: make(map[string]float64),
				SignalValues:      make(map[string]float64),
				SignalErrors:      make(map[string]string),
			}
			var mu sync.Mutex
			newClassifier(tc.onError).evaluatePIISignal(context.Background(), results, &mu, text, nil)
			if results.PIIDetected != tc.wantDetected {
				t.Fatalf("PIIDetected = %v, want %v (entities=%v)", results.PIIDetected, tc.wantDetected, results.PIIEntities)
			}
			for _, e := range results.PIIEntities {
				if e == "O" {
					t.Fatalf("outside label reached PIIEntities: %v", results.PIIEntities)
				}
			}
		})
	}
}
