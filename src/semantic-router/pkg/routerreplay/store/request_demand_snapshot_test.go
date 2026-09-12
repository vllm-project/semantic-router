package store

import (
	"encoding/json"
	"testing"
)

func TestRequestDemandSnapshotsRoundTripAndClone(t *testing.T) {
	record := Record{RouteDiagnostics: &RouteDiagnostics{
		RequestDemandSnapshots: []RequestDemandSnapshot{{
			Stage:                "provider_bound",
			Representation:       "semantic",
			Model:                "model-a",
			PromptTokens:         128,
			ReservedOutputTokens: 64,
			TotalDemandTokens:    192,
			TotalDemandKnown:     true,
			CountingSource:       "fallback",
			OutputReserveSource:  "explicit",
			RequestGeneration:    3,
		}},
	}}

	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Record
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.RouteDiagnostics == nil || len(decoded.RouteDiagnostics.RequestDemandSnapshots) != 1 ||
		decoded.RouteDiagnostics.RequestDemandSnapshots[0].TotalDemandTokens != 192 {
		t.Fatalf("request demand snapshot changed during JSON round trip: %+v", decoded.RouteDiagnostics)
	}

	cloned := cloneRecord(record)
	cloned.RouteDiagnostics.RequestDemandSnapshots[0].PromptTokens = 999
	if record.RouteDiagnostics.RequestDemandSnapshots[0].PromptTokens != 128 {
		t.Fatalf("clone mutation changed original request demand: %+v", record.RouteDiagnostics.RequestDemandSnapshots)
	}
}

func TestRequestDemandSnapshotsRemainOptionalForLegacyReplay(t *testing.T) {
	var record Record
	if err := json.Unmarshal([]byte(`{"route_diagnostics":{"selection_method":"static"}}`), &record); err != nil {
		t.Fatal(err)
	}
	if record.RouteDiagnostics == nil || record.RouteDiagnostics.RequestDemandSnapshots != nil {
		t.Fatalf("legacy replay invented request demand snapshots: %+v", record.RouteDiagnostics)
	}
}
