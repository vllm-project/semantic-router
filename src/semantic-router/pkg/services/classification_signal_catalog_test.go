package services

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMatchedSignalResponseFollowsCanonicalSignalCatalog(t *testing.T) {
	for _, signal := range config.SignalCatalog() {
		resolver, present := matchedSignalResolvers[signal.Type]
		if !signal.DecisionReferenceable {
			if present {
				t.Fatalf("response-only signal %q has a request-time observation resolver", signal.Type)
			}
			continue
		}
		if !present {
			t.Fatalf("decision signal %q has no observation resolver", signal.Type)
		}

		matched := &MatchedSignals{}
		*resolver(matched) = []string{"catalog-sentinel"}
		payload, err := json.Marshal(matched)
		if err != nil {
			t.Fatalf("marshal %q observation: %v", signal.Type, err)
		}
		var fields map[string][]string
		if err := json.Unmarshal(payload, &fields); err != nil {
			t.Fatalf("decode %q observation: %v", signal.Type, err)
		}
		if len(fields) != 1 || len(fields[signal.ObservationKey]) != 1 || fields[signal.ObservationKey][0] != "catalog-sentinel" {
			t.Fatalf("signal %q observation payload = %s, want field %q", signal.Type, payload, signal.ObservationKey)
		}
	}
}
