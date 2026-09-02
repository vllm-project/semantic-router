package evaluationplane

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

type declaredShiftLiveFixture struct {
	runDir        string
	manifest      RunManifest
	qualification suiteGateQualification
	records       recordAttestation
	attestation   executionAttestation
}

type declaredShiftLiveFixtureSource struct {
	service                    *Service
	root                       string
	revision                   string
	qualificationReceiptDigest string
	perturbationArtifactDigest string
	pairDigest                 string
	sourceCaseID               string
	targetCaseID               string
}

func testJSONLines(t *testing.T, rows ...map[string]any) []byte {
	t.Helper()
	data := make([]byte, 0)
	for _, row := range rows {
		encoded, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		data = append(data, encoded...)
		data = append(data, '\n')
	}
	return data
}

func writeDeclaredShiftSuiteSource(t *testing.T) declaredShiftLiveFixtureSource {
	t.Helper()
	service, root := newTestService(t, &controlledProcess{}, 1)
	sourceVisible := map[string]any{
		"schema_version": SchemaVersion, "id": "source", "track_ids": []TrackID{"routing"},
		"messages": []map[string]any{{"role": "user", "content": "source"}}, "modality": "text", "tags": []string{},
	}
	targetVisible := map[string]any{
		"schema_version": SchemaVersion, "id": "perturbed", "track_ids": []TrackID{"routing"},
		"messages": []map[string]any{{"role": "user", "content": "perturbed"}}, "modality": "text", "tags": []string{},
	}
	sourceGrading := map[string]any{"schema_version": SchemaVersion, "case_id": "source", "weight": 1.0}
	targetGrading := map[string]any{"schema_version": SchemaVersion, "case_id": "perturbed", "weight": 1.0}
	pairDigest := digestString("declared-shift-source-row")
	perturbations := testJSONLines(t, map[string]any{
		"schema_version": normalizedSuiteSchemaVersion, "pair_id": "pair-1",
		"source_case_id": "source", "perturbed_case_id": "perturbed", "relation": "invariant",
		"slice_ids": []string{"declared:paraphrase"}, "native_pair_count": 1, "source_record_digest": pairDigest,
	})
	revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "declared-shift", importedSuiteFixtureOptions{
		adapterID: "routerarena", trackIDs: []TrackID{"routing"}, origin: "registered_parser_import", parserVerified: true,
		visibleCaseBytes:  testJSONLines(t, sourceVisible, targetVisible),
		gradingCaseBytes:  testJSONLines(t, sourceGrading, targetGrading),
		perturbationBytes: perturbations, caseCount: 2,
	})
	document, err := loadInstalledSuiteDocument(service.registrySource.suiteStorePath, "declared-shift")
	if err != nil {
		t.Fatal(err)
	}
	receiptBytes, err := json.Marshal(document.Manifest.QualificationReceipt)
	if err != nil {
		t.Fatal(err)
	}
	receiptDigest, err := canonicalJSONDigest(receiptBytes)
	if err != nil {
		t.Fatal(err)
	}
	var artifactRefs map[string]suiteArtifactReference
	if decodeArtifactsErr := json.Unmarshal(document.Manifest.Artifacts, &artifactRefs); decodeArtifactsErr != nil {
		t.Fatal(decodeArtifactsErr)
	}
	return declaredShiftLiveFixtureSource{
		service: service, root: root, revision: revision,
		qualificationReceiptDigest: receiptDigest,
		perturbationArtifactDigest: artifactRefs["perturbations"].Digest,
		pairDigest:                 pairDigest,
		sourceCaseID:               normalizedOpaqueID("case", revision, "case", "source"),
		targetCaseID:               normalizedOpaqueID("case", revision, "case", "perturbed"),
	}
}

func writeDeclaredShiftLiveFixture(t *testing.T, targetArmID string) declaredShiftLiveFixture {
	t.Helper()
	source := writeDeclaredShiftSuiteSource(t)
	service, root, revision := source.service, source.root, source.revision
	sourceCaseID, targetCaseID := source.sourceCaseID, source.targetCaseID
	manifest := RunManifest{
		SchemaVersion: SchemaVersion, RunID: "declared-shift-live-run", ManifestDigest: digestString("declared-shift-manifest"),
		Mode: ModeLive, SampleLimit: 2, Seed: 19, TrackIDs: []TrackID{"routing"},
		SuiteIDs: []string{"declared-shift"}, SuiteRevisions: map[string]string{"declared-shift": revision},
		SuiteExecutors:       map[string]string{"declared-shift": normalizedSuiteLiveExecutorID},
		PolicySnapshotDigest: digestString("declared-shift-policy"),
		Target: ManifestTarget{ID: "mom-declared-shift", Kind: "mixture-of-models", BackendTopologyDigest: digestString("declared-shift-topology"), Mixture: &ManifestMixture{
			ID: "mom-declared-shift", EntrypointModel: "entrypoint", RecipeName: "recipe",
			ModelArms: []ModelArm{{ID: "arm-a", Model: "model-a"}, {ID: "arm-b", Model: "model-b"}},
		}},
	}
	runDir := filepath.Join(root, "runs", manifest.RunID)
	if createRunDirErr := os.MkdirAll(runDir, 0o700); createRunDirErr != nil {
		t.Fatal(createRunDirErr)
	}
	sourceCase := validVisibleCaseRow(sourceCaseID)
	targetCase := validVisibleCaseRow(targetCaseID)
	writeJSONLinesForTest(t, filepath.Join(runDir, "cases.jsonl"), sourceCase, targetCase)
	sourceReceipt := digestString("declared-shift-source-receipt")
	targetReceipt := digestString("declared-shift-target-receipt")
	sourceRecord := validExecutionRecordRow("routing-source", sourceCaseID)
	sourceRecord["success"] = true
	sourceRecord["selected_arm_id"] = "arm-a"
	sourceRecord["evidence_kind"] = declaredShiftLiveEvidenceSourceID
	sourceRecord["broker_receipt"] = sourceReceipt
	targetRecord := validExecutionRecordRow("routing-target", targetCaseID)
	targetRecord["success"] = true
	targetRecord["selected_arm_id"] = targetArmID
	targetRecord["evidence_kind"] = declaredShiftLiveEvidenceSourceID
	targetRecord["broker_receipt"] = targetReceipt
	targetRecord["robustness"] = map[string]any{
		"method_id": declaredShiftLiveMethodID, "suite_id": "declared-shift", "suite_revision": revision,
		"qualification_receipt_digest": source.qualificationReceiptDigest, "perturbation_artifact_digest": source.perturbationArtifactDigest,
		"pair_id": "pair-1", "source_case_id": sourceCaseID, "target_case_id": targetCaseID,
		"shift_type": "paraphrase", "relation": "invariant", "source_action_id": "arm-a",
		"slice_ids": []string{"declared:paraphrase"}, "native_pair_count": 1, "source_record_digest": source.pairDigest,
	}
	writeJSONLinesForTest(t, filepath.Join(runDir, "records.jsonl"), sourceRecord, targetRecord)
	if writeFailureSummaryErr := writeJSONAtomic(filepath.Join(runDir, "failure-summary.json"), map[string]any{
		"schema_version": SchemaVersion, "total_records": 2, "failed": 0, "unavailable": 0,
		"by_track": []map[string]any{{"track_id": "routing", "succeeded": 2, "failed": 0, "unavailable": 0}},
	}); writeFailureSummaryErr != nil {
		t.Fatal(writeFailureSummaryErr)
	}
	identities := normalizedSuiteIdentityLineage{
		SchemaVersion: normalizedSuiteSchemaVersion, SuiteRevisions: manifest.SuiteRevisions,
		CaseIdentities: []normalizedLineageIdentity{
			{SuiteID: "declared-shift", OpaqueID: sourceCaseID, SourceID: "source"},
			{SuiteID: "declared-shift", OpaqueID: targetCaseID, SourceID: "perturbed"},
		}, ArmIdentities: []normalizedLineageIdentity{}, ActionIdentities: []normalizedLineageIdentity{},
	}
	if writeLineageErr := writeJSONAtomic(filepath.Join(runDir, "lineage.json"), map[string]any{
		"schema_version":              SchemaVersion,
		"resolved_snapshot":           testResolvedLineageSnapshot(manifest.ManifestDigest),
		"normalized_suite_identities": identities,
	}); writeLineageErr != nil {
		t.Fatal(writeLineageErr)
	}
	records, err := validateRecordsAndFailureSummary(
		runDir, manifest, builtinExecutorContractForTest(t, normalizedSuiteLiveExecutorID),
	)
	if err != nil {
		t.Fatalf("validate declared-shift records: %v", err)
	}
	qualification, err := resolveSuiteGateQualification(
		service.registrySource.suiteStorePath, manifest, builtinExecutorContractForTest(t, normalizedSuiteLiveExecutorID),
	)
	if err != nil {
		t.Fatal(err)
	}
	attestation := executionAttestation{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: manifest.RunID, ManifestDigest: manifest.ManifestDigest, TargetID: manifest.Target.ID, Mode: ModeLive,
		PolicySnapshotDigest: manifest.PolicySnapshotDigest, BackendTopologyDigest: manifest.Target.BackendTopologyDigest,
		Digest: digestString("declared-shift-attestation"),
		Entries: []executionAttestationEntry{
			{Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: sourceCaseID, BrokerReceipt: sourceReceipt},
			{Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: targetCaseID, BrokerReceipt: targetReceipt},
		},
	}
	return declaredShiftLiveFixture{runDir: runDir, manifest: manifest, qualification: qualification, records: records, attestation: attestation}
}

func TestLiveDeclaredShiftServerReducerSealsPassAndFailAtE4(t *testing.T) {
	for _, test := range []struct {
		name, targetArm string
		passed          bool
	}{
		{name: "pass", targetArm: "arm-a", passed: true},
		{name: "fail", targetArm: "arm-b", passed: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := writeDeclaredShiftLiveFixture(t, test.targetArm)
			method := fixture.records.Methods.Robustness
			if !method.SourceQualified || method.Passed == nil || *method.Passed != test.passed || method.PairCount != 1 {
				t.Fatalf("server reducer=%+v", method)
			}
			levels, err := deriveSealedEvidenceLevels(
				fixture.runDir, fixture.manifest, fixture.records, fixture.qualification,
				builtinExecutorContractForTest(t, normalizedSuiteLiveExecutorID), nil, &fixture.attestation,
			)
			if err != nil {
				t.Fatal(err)
			}
			if levels.ByTrack["routing"] != "E4" || levels.Run != "E4" || !fixture.qualification.withSealedEvidenceLevels(levels).qualifies("G4") {
				t.Fatalf("sealed levels=%+v qualification=%+v", levels, fixture.qualification)
			}
		})
	}
}

func TestLiveDeclaredShiftSealRejectsMissingOrForgedReceiptEvidence(t *testing.T) {
	fixture := writeDeclaredShiftLiveFixture(t, "arm-a")
	for _, test := range []struct {
		name   string
		mutate func(*executionAttestation)
	}{
		{name: "missing receipt", mutate: func(value *executionAttestation) { value.Entries = value.Entries[:1] }},
		{name: "wrong case", mutate: func(value *executionAttestation) { value.Entries[1].CaseID = "other" }},
		{name: "forged attestation digest", mutate: func(value *executionAttestation) { value.Digest = "" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := fixture.attestation
			candidate.Entries = append([]executionAttestationEntry(nil), fixture.attestation.Entries...)
			test.mutate(&candidate)
			levels, err := deriveSealedEvidenceLevels(
				fixture.runDir, fixture.manifest, fixture.records, fixture.qualification,
				builtinExecutorContractForTest(t, normalizedSuiteLiveExecutorID), nil, &candidate,
			)
			if err != nil {
				t.Fatal(err)
			}
			if levels.ByTrack["routing"] != "E0" || fixture.qualification.withSealedEvidenceLevels(levels).qualifies("G4") {
				t.Fatalf("forged attestation levels=%+v", levels)
			}
		})
	}
}

func mutateDeclaredShiftRecords(
	t *testing.T,
	fixture declaredShiftLiveFixture,
	mutate func([]map[string]any),
) (recordAttestation, error) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(fixture.runDir, "records.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSpace(data), []byte("\n"))
	rows := make([]map[string]any, len(lines))
	for index, line := range lines {
		if err := json.Unmarshal(line, &rows[index]); err != nil {
			t.Fatal(err)
		}
	}
	mutate(rows)
	writeJSONLinesForTest(t, filepath.Join(fixture.runDir, "records.jsonl"), rows...)
	return validateRecordsAndFailureSummary(
		fixture.runDir, fixture.manifest,
		builtinExecutorContractForTest(t, normalizedSuiteLiveExecutorID),
	)
}

func TestLiveDeclaredShiftRejectsForgedInstalledRelationBindings(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func([]map[string]any)
	}{
		{name: "native count drift", mutate: func(rows []map[string]any) {
			rows[1]["robustness"].(map[string]any)["native_pair_count"] = float64(2)
		}},
		{name: "qualification receipt drift", mutate: func(rows []map[string]any) {
			rows[1]["robustness"].(map[string]any)["qualification_receipt_digest"] = digestString("forged-qualification")
		}},
		{name: "source record drift", mutate: func(rows []map[string]any) {
			rows[1]["robustness"].(map[string]any)["source_record_digest"] = digestString("forged-source")
		}},
		{name: "artifact digest drift", mutate: func(rows []map[string]any) {
			rows[1]["robustness"].(map[string]any)["perturbation_artifact_digest"] = digestString("forged-artifact")
		}},
		{name: "unregistered evidence source", mutate: func(rows []map[string]any) {
			rows[1]["evidence_kind"] = "forged-live-evidence.v1"
		}},
		{name: "duplicate broker receipt", mutate: func(rows []map[string]any) {
			rows[1]["broker_receipt"] = rows[0]["broker_receipt"]
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := writeDeclaredShiftLiveFixture(t, "arm-a")
			_, err := mutateDeclaredShiftRecords(t, fixture, test.mutate)
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("forged declared-shift binding error=%v", err)
			}
		})
	}
}

func TestLiveDeclaredShiftMissingPairOrSourceReceiptStaysUnavailableE0(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func([]map[string]any)
	}{
		{name: "missing pair method", mutate: func(rows []map[string]any) { delete(rows[1], "robustness") }},
		{name: "missing source receipt", mutate: func(rows []map[string]any) { delete(rows[0], "broker_receipt") }},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := writeDeclaredShiftLiveFixture(t, "arm-a")
			records, err := mutateDeclaredShiftRecords(t, fixture, test.mutate)
			if err != nil {
				t.Fatal(err)
			}
			levels, err := deriveSealedEvidenceLevels(
				fixture.runDir, fixture.manifest, records, fixture.qualification,
				builtinExecutorContractForTest(t, normalizedSuiteLiveExecutorID), nil, &fixture.attestation,
			)
			if err != nil {
				t.Fatal(err)
			}
			if records.Methods.Robustness.Passed != nil || levels.ByTrack["routing"] != "E0" ||
				fixture.qualification.withSealedEvidenceLevels(levels).qualifies("G4") {
				t.Fatalf("incomplete relation records=%+v levels=%+v", records.Methods.Robustness, levels)
			}
		})
	}
}

func TestReplayCannotClaimServerLiveDeclaredShiftMethod(t *testing.T) {
	suiteID, revision := "suite", digestString("revision")
	receipt, artifact := digestString("receipt"), digestString("artifact")
	source, target, arm, broker := "source", "target", "arm-a", digestString("broker")
	kind := declaredShiftLiveEvidenceSourceID
	success := true
	record := executionRecordEvidence{
		TrackID: "routing", CaseID: target, Status: "succeeded", Success: &success,
		SelectedArmID: &arm, BrokerReceipt: &broker, EvidenceKind: &kind,
		Robustness: &robustnessMethodEvidence{
			MethodID: declaredShiftLiveMethodID, SuiteID: &suiteID, SuiteRevision: &revision,
			QualificationReceiptDigest: &receipt, PerturbationArtifactDigest: &artifact,
			PairID: "pair", SourceCaseID: source, TargetCaseID: target, ShiftType: "paraphrase",
			Relation: "invariant", SourceActionID: arm, SliceIDs: []string{"slice"}, NativePairCount: 1,
			SourceRecordDigest: digestString("source-record"),
		},
	}
	if err := validateMethodRecord(record, builtinExecutorContractForTest(t, normalizedSuiteExecutorID)); err == nil {
		t.Fatal("normalized replay accepted a server-live declared-shift method")
	}
}

func TestImportedNormalizedSuiteCannotQualifyLiveDeclaredShift(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "unverified-import")
	manifest := normalizedReplayManifest("unverified-import", revision)
	manifest.Mode = ModeLive
	manifest.SuiteExecutors["unverified-import"] = normalizedSuiteLiveExecutorID
	qualification, err := resolveSuiteGateQualification(
		service.registrySource.suiteStorePath, manifest, builtinExecutorContractForTest(t, normalizedSuiteLiveExecutorID),
	)
	if err != nil {
		t.Fatal(err)
	}
	if qualification.qualifies("G4") {
		t.Fatal("user-provided normalized import qualified G4")
	}
}
