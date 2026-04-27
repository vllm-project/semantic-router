package classification

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHydrateSessionLookupRuntimeErrors(t *testing.T) {
	t.Parallel()
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				SessionMetricRules: []config.SessionMetricRule{
					{
						Name:      "s",
						Kind:      "state",
						State:     "session_routing.cumulative_cost_usd",
						Normalize: "identity",
					},
					{
						Name:  "l",
						Kind:  "lookup",
						Table: "t",
						Key:   []string{"session_routing.current_model"},
					},
				},
			},
		},
	}
	results := &SignalResults{
		SignalValues:      map[string]float64{},
		SignalConfidences: map[string]float64{},
	}
	ctx := &SignalSessionContext{
		Scalars: map[string]float64{},
		Strings: map[string]string{},
	}
	errs := HydrateSessionLookupRuleValues(cfg, results, ctx)
	if len(errs) != 2 {
		t.Fatalf("expected 2 errors, got %v", errs)
	}

	results2 := &SignalResults{
		SignalValues:      map[string]float64{},
		SignalConfidences: map[string]float64{},
	}
	ctx2 := &SignalSessionContext{
		Scalars: map[string]float64{"session_routing.cumulative_cost_usd": 3},
		Strings: map[string]string{"session_routing.current_model": "a"},
		Lookup:  &failingLookupResolver{},
	}
	errs2 := HydrateSessionLookupRuleValues(cfg, results2, ctx2)
	if len(errs2) != 1 || !strings.Contains(errs2[0].Error(), "session_metric") {
		t.Fatalf("expected lookup failure, got %v", errs2)
	}
	if len(results2.MatchedSessionMetricRules) != 1 {
		t.Fatalf("session should hydrate: matched=%v errs=%v", results2.MatchedSessionMetricRules, errs2)
	}
}

type failingLookupResolver struct{}

func (f *failingLookupResolver) LookupFloat64(table string, key []string) (float64, error) {
	return 0, errors.New("boom")
}
