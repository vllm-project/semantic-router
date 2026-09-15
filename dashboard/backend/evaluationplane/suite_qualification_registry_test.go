package evaluationplane

import "testing"

func TestNormalizedImportProvenanceMatchesPythonGolden(t *testing.T) {
	var golden struct {
		SchemaVersion           string        `json:"schema_version"`
		EvidenceLevel           EvidenceLevel `json:"evidence_level"`
		Origins                 []string      `json:"origins"`
		NativeExecutionAttested bool          `json:"native_execution_attested"`
		PromotionEligible       bool          `json:"promotion_eligible"`
		QualifiedGateIDs        []string      `json:"qualified_gate_ids"`
	}
	decodeGoldenStrict(t, "normalized-import-provenance.json", &golden)
	if golden.SchemaVersion != suiteQualificationContractVersion || golden.EvidenceLevel != "E0" ||
		golden.NativeExecutionAttested || golden.PromotionEligible || len(golden.QualifiedGateIDs) != 0 ||
		len(golden.Origins) != 2 || !validImportOrigin(golden.Origins[0], true) ||
		!validImportOrigin(golden.Origins[1], false) {
		t.Fatalf("invalid normalized import provenance golden: %+v", golden)
	}
}

func TestNormalizedAdapterRegistryPinsSourceAndWorkloadOnly(t *testing.T) {
	if len(normalizedAdapterContracts) != 13 {
		t.Fatalf("normalized adapter count=%d, want the 13 research benchmarks", len(normalizedAdapterContracts))
	}
	for adapterID, contract := range normalizedAdapterContracts {
		benchmark, found := researchBenchmarkByAdapter(adapterID)
		if !found {
			t.Fatalf("adapter %q is not in the research inventory", adapterID)
		}
		if !portableSuiteIDPattern.MatchString(adapterID) ||
			!adapterSourceRevisionPattern.MatchString(contract.sourceRevision) ||
			contract.decisionUnit != benchmark.DecisionUnit || contract.actionSpace != benchmark.ActionSpace {
			t.Fatalf("adapter %q has an invalid import contract: %+v", adapterID, contract)
		}
		if benchmark.Status == "blocked" {
			if len(contract.trackIDs) != 0 {
				t.Fatalf("blocked adapter %q advertises import tracks", adapterID)
			}
		} else if !canonicalTrackOrder(contract.trackIDs) {
			t.Fatalf("adapter %q has an invalid import track order", adapterID)
		}
		if contract.datasetRevision != "" && !adapterSourceRevisionPattern.MatchString(contract.datasetRevision) {
			t.Fatalf("adapter %q has an invalid dataset revision", adapterID)
		}
	}
}

func TestNormalizedAdapterTrackValidationDoesNotImplyEvidenceStrength(t *testing.T) {
	contract := normalizedAdapterContracts["routerarena"]
	if !normalizedAdapterTracksMatch(contract, []TrackID{"routing"}) {
		t.Fatal("registered RouterArena routing track was rejected")
	}
	if normalizedAdapterTracksMatch(contract, []TrackID{"safety"}) {
		t.Fatal("unregistered RouterArena safety track was accepted")
	}
}
