package evaluationplane

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type mixedEvidenceFixture struct {
	RunDir        string
	Manifest      RunManifest
	Qualification suiteGateQualification
	Records       recordAttestation
	RoutingCaseID string
	PoolCaseID    string
}

func writeMixedEvidenceFixture(t *testing.T, routingKind string) mixedEvidenceFixture {
	t.Helper()
	service, root := newTestService(t, &controlledProcess{}, 1)
	routingRevision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "mixed-routing", importedSuiteFixtureOptions{
		adapterID: "routerarena", trackIDs: []TrackID{"routing"},
	})
	poolRevision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "mixed-pool", importedSuiteFixtureOptions{
		adapterID: "xroutebench", trackIDs: []TrackID{"model_pool"},
	})
	manifest := RunManifest{
		Mode: ModeReplay, Target: ManifestTarget{ID: "benchmark-source", Kind: "normalized-benchmark-source"},
		SuiteIDs:       []string{"mixed-pool", "mixed-routing"},
		SuiteRevisions: map[string]string{"mixed-pool": poolRevision, "mixed-routing": routingRevision},
		SuiteExecutors: map[string]string{"mixed-pool": normalizedSuiteExecutorID, "mixed-routing": normalizedSuiteExecutorID},
		TrackIDs:       []TrackID{"routing", "model_pool"}, SampleLimit: 1, Seed: 19,
	}
	runDir := filepath.Join(root, "runs", "mixed-evidence")
	if err := os.MkdirAll(runDir, 0o700); err != nil {
		t.Fatal(err)
	}
	routingCaseID := normalizedOpaqueID("case", routingRevision, "case", "case-1")
	poolCaseID := normalizedOpaqueID("case", poolRevision, "case", "case-1")
	routingCase := validVisibleCaseRow(routingCaseID)
	poolCase := validVisibleCaseRow(poolCaseID)
	poolCase["track_ids"] = []TrackID{"model_pool"}
	writeJSONLinesForTest(t, filepath.Join(runDir, "cases.jsonl"), routingCase, poolCase)
	routingRecord := validExecutionRecordRow("routing-record", routingCaseID)
	routingRecord["attempt_id"] = "routing-attempt"
	routingRecord["evidence_kind"] = routingKind
	routingRecord["success"] = true
	routingRecord["selection_status"] = "selected"
	routingRecord["selection_method"] = normalizedSuiteExecutorID
	routingRecord["selected_arm_id"] = normalizedOpaqueID("arm", routingRevision, "arm", "arm-a")
	routingRecord["fallback"] = false
	poolRecord := validExecutionRecordRow("pool-record", poolCaseID)
	poolRecord["track_id"] = "model_pool"
	poolRecord["attempt_id"] = "pool-attempt"
	poolRecord["status"] = "unavailable"
	poolRecord["evidence_kind"] = normalizedSuiteExecutorID + ";ceiling=E0"
	delete(poolRecord, "quality")
	delete(poolRecord, "latency_ms")
	writeJSONLinesForTest(t, filepath.Join(runDir, "records.jsonl"), routingRecord, poolRecord)
	if err := writeJSONAtomic(filepath.Join(runDir, "failure-summary.json"), map[string]any{
		"schema_version": SchemaVersion, "total_records": 2, "failed": 0, "unavailable": 1,
		"by_track": []map[string]any{
			{"track_id": "model_pool", "succeeded": 0, "failed": 0, "unavailable": 1},
			{"track_id": "routing", "succeeded": 1, "failed": 0, "unavailable": 0},
		},
	}); err != nil {
		t.Fatal(err)
	}
	identities := normalizedSuiteIdentityLineage{
		SchemaVersion:  normalizedSuiteSchemaVersion,
		SuiteRevisions: manifest.SuiteRevisions,
		CaseIdentities: []normalizedLineageIdentity{
			{SuiteID: "mixed-pool", OpaqueID: poolCaseID, SourceID: "case-1"},
			{SuiteID: "mixed-routing", OpaqueID: routingCaseID, SourceID: "case-1"},
		},
		ArmIdentities: []normalizedLineageIdentity{
			{SuiteID: "mixed-pool", OpaqueID: normalizedOpaqueID("arm", poolRevision, "arm", "arm-a"), SourceID: "arm-a"},
			{SuiteID: "mixed-routing", OpaqueID: normalizedOpaqueID("arm", routingRevision, "arm", "arm-a"), SourceID: "arm-a"},
		},
		ActionIdentities: []normalizedLineageIdentity{},
	}
	if err := writeJSONAtomic(filepath.Join(runDir, "lineage.json"), map[string]any{
		"schema_version":              SchemaVersion,
		"resolved_snapshot":           testResolvedLineageSnapshot("sha256:" + strings.Repeat("0", 64)),
		"normalized_suite_identities": identities,
	}); err != nil {
		t.Fatal(err)
	}
	records, err := validateRecordsAndFailureSummaryForTest(t, runDir, manifest)
	if err != nil {
		t.Fatalf("validate explicit mixed case-track plan: %v", err)
	}
	qualification, err := resolveSuiteGateQualificationForTest(t, service.registrySource.suiteStorePath, manifest)
	if err != nil {
		t.Fatalf("resolve exact installed receipts: %v", err)
	}
	return mixedEvidenceFixture{
		RunDir: runDir, Manifest: manifest, Qualification: qualification, Records: records,
		RoutingCaseID: routingCaseID, PoolCaseID: poolCaseID,
	}
}

func TestServerEvidenceSealKeepsImportedTracksAtE0(t *testing.T) {
	fixture := writeMixedEvidenceFixture(t, normalizedSuiteExecutorID+";ceiling=E0")
	levels, err := deriveSealedEvidenceLevelsForTest(t, fixture.RunDir, fixture.Manifest, fixture.Records, fixture.Qualification)
	if err != nil {
		t.Fatal(err)
	}
	if levels.Run != "E0" || levels.ByTrack["routing"] != "E0" || levels.ByTrack["model_pool"] != "E0" {
		t.Fatalf("sealed imported levels=%+v, want every track E0", levels)
	}
	report := Report{
		Run:     Run{EvidenceLevel: "E0"},
		Summary: ReportSummary{Coverage: serverCoverage(1, 2)},
		Tracks: []TrackReport{
			{TrackID: "routing", Status: "completed", EvidenceLevel: "E0", Summary: "Collected 1 evidence records.", Coverage: serverCoverage(1, 1)},
			{TrackID: "model_pool", Status: "unavailable", EvidenceLevel: "E0", Summary: "No qualified evidence was produced.", Coverage: serverCoverage(0, 1)},
		},
	}
	if err := validateServerOwnedReportPresentation(report, fixture.Records, levels); err != nil {
		t.Fatalf("mixed server-owned presentation rejected: %v", err)
	}
}

func TestServerEvidenceSealDowngradesTamperedEvidenceKind(t *testing.T) {
	fixture := writeMixedEvidenceFixture(t, normalizedSuiteExecutorID+";ceiling=E4")
	levels, err := deriveSealedEvidenceLevelsForTest(t, fixture.RunDir, fixture.Manifest, fixture.Records, fixture.Qualification)
	if err != nil {
		t.Fatal(err)
	}
	if levels.ByTrack["routing"] != "E0" || levels.Run != "E0" {
		t.Fatalf("tampered kind levels=%+v, want fail-closed E0", levels)
	}
	report := Report{
		Run: Run{EvidenceLevel: "E0"}, Summary: ReportSummary{Coverage: serverCoverage(1, 2)},
		Tracks: []TrackReport{
			{TrackID: "routing", Status: "completed", EvidenceLevel: "E3", Summary: "Collected 1 evidence records.", Coverage: serverCoverage(1, 1)},
			{TrackID: "model_pool", Status: "unavailable", EvidenceLevel: "E0", Summary: "No qualified evidence was produced.", Coverage: serverCoverage(0, 1)},
		},
	}
	err = validateServerOwnedReportPresentation(report, fixture.Records, levels)
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "server-sealed case evidence") {
		t.Fatalf("tampered worker evidence claim error=%v, want server-seal rejection", err)
	}
}

func TestServerEvidenceSealRejectsRunCaseModalityForgedAgainstInstalledSource(t *testing.T) {
	fixture := writeMixedEvidenceFixture(t, normalizedSuiteExecutorID+";ceiling=E0")
	routingCase := validVisibleCaseRow(fixture.RoutingCaseID)
	routingCase["modality"] = "image"
	poolCase := validVisibleCaseRow(fixture.PoolCaseID)
	poolCase["track_ids"] = []TrackID{"model_pool"}
	writeJSONLinesForTest(t, filepath.Join(fixture.RunDir, "cases.jsonl"), routingCase, poolCase)

	records, err := validateRecordsAndFailureSummaryForTest(t, fixture.RunDir, fixture.Manifest)
	if err != nil {
		t.Fatalf("worker-visible forged modality should remain structurally valid: %v", err)
	}
	_, err = deriveSealedEvidenceLevelsForTest(t, fixture.RunDir, fixture.Manifest, records, fixture.Qualification)
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "installed source modality") {
		t.Fatalf("forged run modality error=%v, want installed-source rejection", err)
	}
}

func TestLiveCapacityEvidenceReachesE5OnlyAfterServerProfileAttestation(t *testing.T) {
	manifest := RunManifest{Mode: ModeLive, TrackIDs: []TrackID{"capacity"}}
	derive := func(attestation *capacitySLOAttestation) sealedEvidenceLevels {
		levels, err := deriveSealedEvidenceLevels(
			t.TempDir(),
			manifest,
			recordAttestation{},
			suiteGateQualification{},
			executorContract{},
			attestation,
			nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		return levels
	}
	if levels := derive(nil); levels.Run != "E0" || levels.ByTrack["capacity"] != "E0" {
		t.Fatalf("unattested capacity levels=%+v, want E0", levels)
	}
	if levels := derive(&capacitySLOAttestation{Headroom: -1, LevelCount: 2}); levels.Run != "E5" || levels.ByTrack["capacity"] != "E5" {
		t.Fatalf("server-attested capacity levels=%+v, want E5 even for a measured SLO failure", levels)
	}
}

func TestSealedLiveMoMEvidenceDerivesTrackSpecificLevelsOnlyFromCompleteAttestation(t *testing.T) {
	manifest, records, attestation := sealedLiveMoMEvidenceFixture()
	derive := func(candidate recordAttestation, execution *executionAttestation) sealedEvidenceLevels {
		levels, err := deriveSealedEvidenceLevels(
			t.TempDir(), manifest, candidate, suiteGateQualification{},
			builtinExecutorContractForTest(t, liveRuntimeExecutorID), nil, execution,
		)
		if err != nil {
			t.Fatal(err)
		}
		return levels
	}

	levels := derive(records, &attestation)
	if levels.Run != "E3" || levels.ByTrack["routing"] != "E3" ||
		levels.ByTrack["model_pool"] != "E4" || levels.ByTrack["joint"] != "E5" {
		t.Fatalf("complete sealed live-mom levels=%+v, want E3/E4/E5", levels)
	}

	tests := map[string]struct {
		mutate   func(*recordAttestation, *executionAttestation)
		expected map[TrackID]EvidenceLevel
	}{
		"missing receipt": {mutate: func(_ *recordAttestation, value *executionAttestation) {
			value.Entries[2].BrokerReceipt = ""
		}, expected: map[TrackID]EvidenceLevel{"routing": "E0", "model_pool": "E0", "joint": "E0"}},
		"missing frozen arm": {mutate: func(_ *recordAttestation, value *executionAttestation) {
			value.Entries = append(value.Entries[:3], value.Entries[4:]...)
		}, expected: map[TrackID]EvidenceLevel{"routing": "E3", "model_pool": "E0", "joint": "E5"}},
		"incomplete case coverage": {mutate: func(value *recordAttestation, _ *executionAttestation) {
			delete(value.EvaluatedCaseIDsByTrack["joint"], "case-1")
		}, expected: map[TrackID]EvidenceLevel{"routing": "E3", "model_pool": "E4", "joint": "E0"}},
		"unavailable cell": {mutate: func(value *recordAttestation, _ *executionAttestation) {
			value.CellEvidence["routing"]["case-1"].Unavailable = true
		}, expected: map[TrackID]EvidenceLevel{"routing": "E0", "model_pool": "E4", "joint": "E5"}},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			candidateRecords, candidateAttestation := sealedLiveMoMEvidenceFixtureCopies(records, attestation)
			test.mutate(&candidateRecords, &candidateAttestation)
			got := derive(candidateRecords, &candidateAttestation)
			if got.Run != "E0" {
				t.Fatalf("incomplete selected matrix levels=%+v, want E0 headline", got)
			}
			for trackID, expected := range test.expected {
				if got.ByTrack[trackID] != expected {
					t.Fatalf("incomplete selected matrix levels=%+v, want %s=%s", got, trackID, expected)
				}
			}
		})
	}
	if got := derive(records, nil); got.Run != "E0" {
		t.Fatalf("missing execution attestation levels=%+v, want E0", got)
	}
}

func TestSealedLiveMoMEvidenceQualifiesSelectedAspectsIndependently(t *testing.T) {
	manifest, records, attestation := sealedLiveMoMEvidenceFixture()
	tests := []struct {
		track TrackID
		level EvidenceLevel
	}{
		{track: "routing", level: "E3"},
		{track: "model_pool", level: "E4"},
		{track: "joint", level: "E5"},
	}
	for _, test := range tests {
		t.Run(string(test.track), func(t *testing.T) {
			candidateRecords, candidateAttestation := sealedLiveMoMEvidenceFixtureCopies(records, attestation)
			manifest.TrackIDs = []TrackID{test.track}
			for trackID := range candidateRecords.PlannedCaseIDsByTrack {
				if trackID != test.track {
					delete(candidateRecords.PlannedCaseIDsByTrack, trackID)
					delete(candidateRecords.EvaluatedCaseIDsByTrack, trackID)
					delete(candidateRecords.CellEvidence, trackID)
				}
			}
			filtered := candidateAttestation.Entries[:1]
			for _, entry := range candidateAttestation.Entries[1:] {
				if entry.TrackID == test.track {
					filtered = append(filtered, entry)
				}
			}
			candidateAttestation.Entries = filtered
			levels, err := deriveSealedEvidenceLevels(
				t.TempDir(), manifest, candidateRecords, suiteGateQualification{},
				builtinExecutorContractForTest(t, liveRuntimeExecutorID), nil, &candidateAttestation,
			)
			if err != nil {
				t.Fatal(err)
			}
			if levels.Run != test.level || levels.ByTrack[test.track] != test.level {
				t.Fatalf("selected %s levels=%+v, want %s", test.track, levels, test.level)
			}
			candidateRecords.CellEvidence[test.track]["case-1"].EvidenceKinds = map[string]struct{}{"wrong-kind": {}}
			levels, err = deriveSealedEvidenceLevels(
				t.TempDir(), manifest, candidateRecords, suiteGateQualification{},
				builtinExecutorContractForTest(t, liveRuntimeExecutorID), nil, &candidateAttestation,
			)
			if err != nil {
				t.Fatal(err)
			}
			if levels.Run != "E0" || levels.ByTrack[test.track] != "E0" {
				t.Fatalf("selected %s accepted the wrong evidence kind: %+v", test.track, levels)
			}
		})
	}
}

func TestSealedNormalizedLiveMultimodalEvidenceRequiresCompleteBrokeredCohort(t *testing.T) {
	manifest, records, attestation := sealedLiveMultimodalEvidenceFixture()
	derive := func(candidate recordAttestation, execution *executionAttestation) sealedEvidenceLevels {
		levels, err := deriveSealedEvidenceLevels(
			t.TempDir(), manifest, candidate, suiteGateQualification{},
			builtinExecutorContractForTest(t, normalizedSuiteLiveExecutorID), nil, execution,
		)
		if err != nil {
			t.Fatal(err)
		}
		return levels
	}

	levels := derive(records, &attestation)
	if levels.Run != "E4" || levels.ByTrack["multimodal"] != "E4" {
		t.Fatalf("complete normalized live multimodal levels=%+v, want E4", levels)
	}

	tests := map[string]func(*recordAttestation, *executionAttestation){
		"missing evaluated case": func(value *recordAttestation, _ *executionAttestation) {
			delete(value.EvaluatedCaseIDsByTrack["multimodal"], "image-2")
		},
		"unavailable case": func(value *recordAttestation, _ *executionAttestation) {
			value.CellEvidence["multimodal"]["image-1"].Unavailable = true
		},
		"unsealed evidence kind": func(value *recordAttestation, _ *executionAttestation) {
			value.CellEvidence["multimodal"]["image-1"].EvidenceKinds = map[string]struct{}{"fixture-replay.v1": {}}
		},
		"missing broker receipt": func(_ *recordAttestation, value *executionAttestation) {
			value.Entries = value.Entries[:2]
		},
		"wrong broker operation": func(_ *recordAttestation, value *executionAttestation) {
			value.Entries[1].Operation = workerBrokerArmChatCompletion
		},
		"duplicate broker receipt": func(_ *recordAttestation, value *executionAttestation) {
			value.Entries[2].BrokerReceipt = value.Entries[1].BrokerReceipt
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			candidateRecords, candidateAttestation := sealedLiveMoMEvidenceFixtureCopies(records, attestation)
			mutate(&candidateRecords, &candidateAttestation)
			got := derive(candidateRecords, &candidateAttestation)
			if got.Run != "E0" || got.ByTrack["multimodal"] != "E0" {
				t.Fatalf("fail-closed levels=%+v, want multimodal E0", got)
			}
		})
	}
	if got := derive(records, nil); got.Run != "E0" || got.ByTrack["multimodal"] != "E0" {
		t.Fatalf("missing execution attestation levels=%+v, want E0", got)
	}
}

func TestSyntheticMoMReplayIsPermanentlyE0(t *testing.T) {
	manifest, records, _ := sealedLiveMoMEvidenceFixture()
	manifest.Mode = ModeReplay
	manifest.SuiteIDs = []string{"renamed-mom-cohort"}
	manifest.SuiteExecutors = map[string]string{"renamed-mom-cohort": momReplayExecutorID}
	levels, err := deriveSealedEvidenceLevels(
		t.TempDir(), manifest, records, suiteGateQualification{},
		builtinExecutorContractForTest(t, momReplayExecutorID), nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if levels.Run != "E0" || levels.ByTrack["routing"] != "E0" ||
		levels.ByTrack["model_pool"] != "E0" || levels.ByTrack["joint"] != "E0" {
		t.Fatalf("synthetic replay levels=%+v, want every track E0", levels)
	}
}

func TestMoMReplayPermissionComesFromExecutorAndTargetProfiles(t *testing.T) {
	manifest, _, _ := sealedLiveMoMEvidenceFixture()
	manifest.Mode = ModeReplay
	manifest.SuiteIDs = []string{"new-mom-cohort"}
	manifest.SuiteExecutors = map[string]string{"new-mom-cohort": momReplayExecutorID}
	manifest.Target.EnvoyURL = "http://envoy.test"
	if !manifestUsesMoMCohortReplay(manifest) {
		t.Fatal("renamed MoM cohort was not authorized by its frozen executor contract")
	}
	manifest.SuiteExecutors["new-mom-cohort"] = fixtureReplayExecutorID
	if manifestUsesMoMCohortReplay(manifest) {
		t.Fatal("ordinary replay executor impersonated a brokered MoM cohort")
	}
}

func sealedLiveMoMEvidenceFixture() (RunManifest, recordAttestation, executionAttestation) {
	copySet := func() map[string]struct{} { return map[string]struct{}{"case-1": {}} }
	manifest := RunManifest{
		SchemaVersion: SchemaVersion, RunID: "2650bcb4-05af-46e4-96d4-cb9ec36b393d",
		ManifestDigest: digestString("sealed-live-mom-manifest"), Mode: ModeLive,
		SuiteIDs:             []string{"live-mom-core"},
		SuiteExecutors:       map[string]string{"live-mom-core": liveRuntimeExecutorID},
		TrackIDs:             []TrackID{"routing", "model_pool", "joint"},
		PolicySnapshotDigest: digestString("sealed-live-mom-policy"),
		Target: ManifestTarget{
			SchemaVersion: SchemaVersion, ID: "mom-default",
			BackendTopologyDigest: digestString("sealed-live-mom-topology"),
			Mixture: &ManifestMixture{ModelArms: []ModelArm{
				{ID: "arm-fast", Model: "model-fast"},
				{ID: "arm-strong", Model: "model-strong"},
			}},
		},
	}
	records := recordAttestation{
		validated: true,
		PlannedCaseIDsByTrack: map[TrackID]map[string]struct{}{
			"routing": copySet(), "model_pool": copySet(), "joint": copySet(),
		},
		EvaluatedCaseIDsByTrack: map[TrackID]map[string]struct{}{
			"routing": copySet(), "model_pool": copySet(), "joint": copySet(),
		},
		CellEvidence: map[TrackID]map[string]*recordCellAttestation{
			"routing":    {"case-1": {Rows: 1, EvidenceKinds: map[string]struct{}{liveMoMRoutingEvidenceKind: {}}}},
			"model_pool": {"case-1": {Rows: 2, EvidenceKinds: map[string]struct{}{liveMoMModelPoolEvidenceKind: {}}}},
			"joint":      {"case-1": {Rows: 1, EvidenceKinds: map[string]struct{}{liveMoMJointEvidenceKind: {}}}},
		},
	}
	armFast, armStrong := "arm-fast", "arm-strong"
	attestation := executionAttestation{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: manifest.RunID, ManifestDigest: manifest.ManifestDigest, TargetID: manifest.Target.ID,
		Mode: ModeLive, PolicySnapshotDigest: manifest.PolicySnapshotDigest,
		BackendTopologyDigest: manifest.Target.BackendTopologyDigest,
		Digest:                digestString("sealed-live-mom-attestation"),
		Entries: []executionAttestationEntry{
			{Operation: workerBrokerListModels, BrokerReceipt: digestString("models")},
			{Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: "case-1", BrokerReceipt: digestString("routing")},
			{Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1", ArmID: &armFast, BrokerReceipt: digestString("pool-fast")},
			{Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1", ArmID: &armStrong, BrokerReceipt: digestString("pool-strong")},
			{Operation: workerBrokerRoutedChatCompletion, TrackID: "joint", CaseID: "case-1", BrokerReceipt: digestString("joint")},
		},
	}
	return manifest, records, attestation
}

func sealedLiveMultimodalEvidenceFixture() (RunManifest, recordAttestation, executionAttestation) {
	cases := map[string]struct{}{"image-1": {}, "image-2": {}}
	copyCases := func() map[string]struct{} {
		result := make(map[string]struct{}, len(cases))
		for caseID := range cases {
			result[caseID] = struct{}{}
		}
		return result
	}
	manifest := RunManifest{
		SchemaVersion: SchemaVersion, RunID: "b9542faf-b559-4b80-8192-bdab391b775c",
		ManifestDigest: digestString("sealed-live-multimodal-manifest"), Mode: ModeLive,
		SuiteIDs:             []string{"mmr-bench"},
		SuiteExecutors:       map[string]string{"mmr-bench": normalizedSuiteLiveExecutorID},
		TrackIDs:             []TrackID{"multimodal"},
		PolicySnapshotDigest: digestString("sealed-live-multimodal-policy"),
		Target: ManifestTarget{
			SchemaVersion: SchemaVersion, ID: "mom-default",
			BackendTopologyDigest: digestString("sealed-live-multimodal-topology"),
			Mixture:               &ManifestMixture{ModelArms: []ModelArm{{ID: "arm-vision", Model: "vision-model"}}},
		},
	}
	records := recordAttestation{
		validated: true,
		PlannedCaseIDsByTrack: map[TrackID]map[string]struct{}{
			"multimodal": copyCases(),
		},
		EvaluatedCaseIDsByTrack: map[TrackID]map[string]struct{}{
			"multimodal": copyCases(),
		},
		CellEvidence: map[TrackID]map[string]*recordCellAttestation{
			"multimodal": {
				"image-1": {Rows: 1, EvidenceKinds: map[string]struct{}{normalizedSuiteLiveExecutorID: {}}},
				"image-2": {Rows: 1, EvidenceKinds: map[string]struct{}{normalizedSuiteLiveExecutorID: {}}},
			},
		},
	}
	attestation := executionAttestation{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: manifest.RunID, ManifestDigest: manifest.ManifestDigest, TargetID: manifest.Target.ID,
		Mode: ModeLive, PolicySnapshotDigest: manifest.PolicySnapshotDigest,
		BackendTopologyDigest: manifest.Target.BackendTopologyDigest,
		Digest:                digestString("sealed-live-multimodal-attestation"),
		Entries: []executionAttestationEntry{
			{Operation: workerBrokerListModels, BrokerReceipt: digestString("multimodal-models")},
			{Operation: workerBrokerRoutedChatCompletion, TrackID: "multimodal", CaseID: "image-1", BrokerReceipt: digestString("multimodal-image-1")},
			{Operation: workerBrokerRoutedChatCompletion, TrackID: "multimodal", CaseID: "image-2", BrokerReceipt: digestString("multimodal-image-2")},
		},
	}
	return manifest, records, attestation
}

func sealedLiveMoMEvidenceFixtureCopies(
	records recordAttestation,
	attestation executionAttestation,
) (recordAttestation, executionAttestation) {
	copiedRecords := records
	copiedRecords.PlannedCaseIDsByTrack = make(map[TrackID]map[string]struct{}, len(records.PlannedCaseIDsByTrack))
	copiedRecords.EvaluatedCaseIDsByTrack = make(map[TrackID]map[string]struct{}, len(records.EvaluatedCaseIDsByTrack))
	copiedRecords.CellEvidence = make(map[TrackID]map[string]*recordCellAttestation, len(records.CellEvidence))
	for trackID, values := range records.PlannedCaseIDsByTrack {
		copiedRecords.PlannedCaseIDsByTrack[trackID] = make(map[string]struct{}, len(values))
		for caseID := range values {
			copiedRecords.PlannedCaseIDsByTrack[trackID][caseID] = struct{}{}
		}
	}
	for trackID, values := range records.EvaluatedCaseIDsByTrack {
		copiedRecords.EvaluatedCaseIDsByTrack[trackID] = make(map[string]struct{}, len(values))
		for caseID := range values {
			copiedRecords.EvaluatedCaseIDsByTrack[trackID][caseID] = struct{}{}
		}
	}
	for trackID, values := range records.CellEvidence {
		copiedRecords.CellEvidence[trackID] = make(map[string]*recordCellAttestation, len(values))
		for caseID, cell := range values {
			copied := *cell
			copiedRecords.CellEvidence[trackID][caseID] = &copied
		}
	}
	attestation.Entries = append([]executionAttestationEntry(nil), attestation.Entries...)
	return copiedRecords, attestation
}
