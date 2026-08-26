package usageledger

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestEventMetadataOwnsReplayIndependentRoutingEvidence(t *testing.T) {
	event := testTerminalEvent("metadata-routing", time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC))
	event.Dispatches[0].DecisionID = "decision-complex"
	event.Dispatches[0].DecisionName = "Complex"
	event.Dispatches[0].DecisionTier = 3
	event.Dispatches[0].ModelName = "model-z"

	duplicate := event.Dispatches[0]
	duplicate.DispatchID = "dispatch-duplicate"
	second := event.Dispatches[0]
	second.DispatchID = "dispatch-second"
	second.ModelID = "00000000-0000-4000-8000-000000000001"
	second.ModelName = "model-a"
	event.Dispatches = append(event.Dispatches, duplicate, second)

	metadata := eventMetadata(event)
	if metadata.Decision == nil || metadata.Decision.ID != "decision-complex" ||
		metadata.Decision.Name != "Complex" || metadata.Decision.Tier != 3 {
		t.Fatalf("decision snapshot = %+v", metadata.Decision)
	}
	if len(metadata.Models) != 2 || metadata.Models[0].Name != "model-a" ||
		metadata.Models[1].Name != "model-z" {
		t.Fatalf("served Model snapshots = %+v, want deterministic de-duplicated evidence", metadata.Models)
	}
}

func TestEventMetadataSerializesExplicitEmptyModelEvidence(t *testing.T) {
	metadata := eventMetadata(TerminalEvent{})
	if metadata.Decision != nil || metadata.Models == nil || len(metadata.Models) != 0 {
		t.Fatalf("empty routing evidence = decision %+v, Models %+v", metadata.Decision, metadata.Models)
	}
	payload, err := json.Marshal(metadata)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), `"models":[]`) || strings.Contains(string(payload), `"decision"`) {
		t.Fatalf("empty metadata JSON = %s, want explicit Models and omitted unknown decision", payload)
	}
}
