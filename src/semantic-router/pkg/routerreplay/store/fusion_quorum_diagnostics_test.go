/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package store

import (
	"encoding/json"
	"testing"
)

func TestFusionQuorumDiagnosticsJSONRoundTrip(t *testing.T) {
	record := Record{
		ID: "fusion-quorum",
		RouteDiagnostics: &RouteDiagnostics{
			SelectionMethod: "fusion",
			FusionQuorum: &FusionQuorumDiagnostics{
				RequiredCount: 2,
				UsableCount:   1,
				Attempts: []FusionPanelAttemptDiagnostics{
					{Model: "panel-a", State: "usable", PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12},
					{Model: "panel-b", State: "unusable", PromptTokens: 20, CompletionTokens: 3, TotalTokens: 23},
					{Model: "panel-c", State: "failed"},
				},
			},
		},
	}

	decoded := roundTripRecord(t, record)
	assertFusionQuorumDiagnosticsRoundTrip(t, decoded.RouteDiagnostics)
}

func assertFusionQuorumDiagnosticsRoundTrip(t *testing.T, diagnostics *RouteDiagnostics) {
	t.Helper()
	if diagnostics == nil || diagnostics.FusionQuorum == nil {
		t.Fatalf("Fusion quorum diagnostics missing after round trip: %+v", diagnostics)
	}
	quorum := diagnostics.FusionQuorum
	if quorum.RequiredCount != 2 || quorum.UsableCount != 1 || len(quorum.Attempts) != 3 {
		t.Fatalf("Fusion quorum diagnostics changed after round trip: %+v", quorum)
	}
	assertFusionAttemptRoundTrip(t, quorum.Attempts[1])
}

func assertFusionAttemptRoundTrip(t *testing.T, got FusionPanelAttemptDiagnostics) {
	t.Helper()
	if got.Model != "panel-b" || got.State != "unusable" || got.PromptTokens != 20 ||
		got.CompletionTokens != 3 || got.TotalTokens != 23 {
		t.Fatalf("Fusion attempt changed after round trip: %+v", got)
	}
}

func TestCloneRecordDeepCopiesFusionQuorumDiagnostics(t *testing.T) {
	record := Record{RouteDiagnostics: &RouteDiagnostics{
		FusionQuorum: &FusionQuorumDiagnostics{
			RequiredCount: 2,
			UsableCount:   1,
			Attempts: []FusionPanelAttemptDiagnostics{{
				Model: "panel-a", State: "usable", TotalTokens: 12,
			}},
		},
	}}

	cloned := cloneRecord(record)
	if cloned.RouteDiagnostics == nil || cloned.RouteDiagnostics.FusionQuorum == nil {
		t.Fatal("cloned record lost Fusion quorum diagnostics")
	}
	if cloned.RouteDiagnostics.FusionQuorum == record.RouteDiagnostics.FusionQuorum {
		t.Fatal("Fusion quorum diagnostics pointer was not cloned")
	}
	if &cloned.RouteDiagnostics.FusionQuorum.Attempts[0] == &record.RouteDiagnostics.FusionQuorum.Attempts[0] {
		t.Fatal("Fusion quorum attempt slice was not cloned")
	}

	cloned.RouteDiagnostics.FusionQuorum.RequiredCount = 9
	cloned.RouteDiagnostics.FusionQuorum.Attempts[0].Model = "mutated"
	if record.RouteDiagnostics.FusionQuorum.RequiredCount != 2 ||
		record.RouteDiagnostics.FusionQuorum.Attempts[0].Model != "panel-a" {
		t.Fatalf("clone mutation changed original Fusion diagnostics: %+v", record.RouteDiagnostics.FusionQuorum)
	}
}

func TestFusionQuorumDiagnosticsOldRecordCompatibility(t *testing.T) {
	var record Record
	if err := json.Unmarshal([]byte(`{
		"id":"legacy-record",
		"route_diagnostics":{"selection_method":"fusion"}
	}`), &record); err != nil {
		t.Fatalf("unmarshal legacy record: %v", err)
	}
	if record.RouteDiagnostics == nil {
		t.Fatal("legacy route diagnostics missing")
	}
	if record.RouteDiagnostics.FusionQuorum != nil {
		t.Fatalf("legacy record invented Fusion quorum diagnostics: %+v", record.RouteDiagnostics.FusionQuorum)
	}
}
