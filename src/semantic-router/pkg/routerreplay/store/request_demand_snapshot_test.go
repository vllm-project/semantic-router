package store

import (
	"encoding/json"
	"testing"
)

func TestRequestDemandSnapshotsRoundTripAndClone(t *testing.T) {
	reserve := 64
	record := Record{RouteDiagnostics: &RouteDiagnostics{
		FinalAdmission: &FinalAdmissionDiagnostics{
			Model: "model-a", Source: "provider_render", Verdict: "fits", PromptTokens: 144,
			ReservedOutputTokens: &reserve, ContextWindow: 208,
		},
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
	admission := decoded.RouteDiagnostics.FinalAdmission
	if admission == nil || admission.PromptTokens != 144 || admission.ReservedOutputTokens == nil || *admission.ReservedOutputTokens != 64 {
		t.Fatalf("final admission changed during JSON round trip: %+v", admission)
	}

	cloned := cloneRecord(record)
	cloned.RouteDiagnostics.RequestDemandSnapshots[0].PromptTokens = 999
	if record.RouteDiagnostics.RequestDemandSnapshots[0].PromptTokens != 128 {
		t.Fatalf("clone mutation changed original request demand: %+v", record.RouteDiagnostics.RequestDemandSnapshots)
	}
	*cloned.RouteDiagnostics.FinalAdmission.ReservedOutputTokens = 999
	if *record.RouteDiagnostics.FinalAdmission.ReservedOutputTokens != 64 {
		t.Fatal("clone mutation changed the original admitted output reserve")
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
	if record.RouteDiagnostics.FinalAdmission != nil {
		t.Fatal("legacy replay invented final admission evidence")
	}
}
