package evaluationplane

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNormalizedWorkloadBindingUsesInstalledHiddenLabelsForReplayAndLive(t *testing.T) {
	for _, mode := range []Mode{ModeReplay, ModeLive} {
		t.Run(string(mode), func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "trusted-routing", importedSuiteFixtureOptions{
				adapterID: "routerarena", trackIDs: []TrackID{"routing"},
				gradingCaseOverrides: map[string]any{
					"expected_answer":  "server-owned-answer",
					"expected_route":   "arm-a",
					"preferred_arm_id": "arm-a",
				},
			})
			runDir := filepath.Join(root, "runs", "normalized-workload-"+string(mode))
			if err := os.MkdirAll(runDir, 0o700); err != nil {
				t.Fatal(err)
			}
			manifest, identities, visible, grading := normalizedWorkloadTestCase(mode, revision)
			writeJSONLinesForTest(t, filepath.Join(runDir, "cases.jsonl"), visible)
			writeJSONLinesForTest(t, filepath.Join(runDir, "grading-cases.jsonl"), grading)
			writeNormalizedWorkloadLineageForTest(t, runDir, identities)

			executor := builtinExecutorContractForTest(t, manifest.SuiteExecutors[manifest.SuiteIDs[0]])
			if err := validateNormalizedWorkloadFromLineage(runDir, manifest, executor); err != nil {
				t.Fatalf("validate trusted normalized %s workload: %v", mode, err)
			}

			tamperedGrading := cloneAnyMap(grading)
			tamperedGrading["expected_answer"] = "worker-chosen-answer"
			writeJSONLinesForTest(t, filepath.Join(runDir, "grading-cases.jsonl"), tamperedGrading)
			err := validateNormalizedWorkloadFromLineage(runDir, manifest, executor)
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "installed label") {
				t.Fatalf("tampered normalized %s hidden label error=%v, want server-owned rejection", mode, err)
			}

			writeJSONLinesForTest(t, filepath.Join(runDir, "grading-cases.jsonl"), grading)
			tamperedVisible := cloneAnyMap(visible)
			tamperedVisible["messages"] = []map[string]any{{"role": "user", "content": "worker-chosen-prompt"}}
			writeJSONLinesForTest(t, filepath.Join(runDir, "cases.jsonl"), tamperedVisible)
			err = validateNormalizedWorkloadFromLineage(runDir, manifest, executor)
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "installed case") {
				t.Fatalf("tampered normalized %s visible case error=%v, want server-owned rejection", mode, err)
			}
		})
	}
}

func TestNormalizedWorkloadBindingRejectsWorkerControlledCaseOrder(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	revisionA := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "suite-a", importedSuiteFixtureOptions{
		adapterID: "routerarena", trackIDs: []TrackID{"routing"},
		gradingCaseOverrides: map[string]any{"expected_answer": "answer-a"},
	})
	revisionB := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "suite-b", importedSuiteFixtureOptions{
		adapterID: "routerarena", trackIDs: []TrackID{"routing"},
		gradingCaseOverrides: map[string]any{"expected_answer": "answer-b"},
	})
	runDir := filepath.Join(root, "runs", "normalized-order")
	if err := os.MkdirAll(runDir, 0o700); err != nil {
		t.Fatal(err)
	}
	manifest := RunManifest{
		Mode: ModeReplay, Target: ManifestTarget{ID: "benchmark-source", Kind: "normalized-benchmark-source"},
		SuiteIDs:       []string{"suite-a", "suite-b"},
		SuiteRevisions: map[string]string{"suite-a": revisionA, "suite-b": revisionB},
		SuiteExecutors: map[string]string{"suite-a": normalizedSuiteExecutorID, "suite-b": normalizedSuiteExecutorID},
		TrackIDs:       []TrackID{"routing"}, SampleLimit: 1, Seed: 19,
	}
	caseA, gradingA := normalizedReplayRows(revisionA, "answer-a")
	caseB, gradingB := normalizedReplayRows(revisionB, "answer-b")
	identities := normalizedSuiteIdentityLineage{
		SchemaVersion:  normalizedSuiteSchemaVersion,
		SuiteRevisions: manifest.SuiteRevisions,
		CaseIdentities: []normalizedLineageIdentity{
			{SuiteID: "suite-a", OpaqueID: caseA["id"].(string), SourceID: "case-1"},
			{SuiteID: "suite-b", OpaqueID: caseB["id"].(string), SourceID: "case-1"},
		},
		ArmIdentities: []normalizedLineageIdentity{
			{SuiteID: "suite-a", OpaqueID: normalizedOpaqueID("arm", revisionA, "arm", "arm-a"), SourceID: "arm-a"},
			{SuiteID: "suite-b", OpaqueID: normalizedOpaqueID("arm", revisionB, "arm", "arm-a"), SourceID: "arm-a"},
		},
		ActionIdentities: []normalizedLineageIdentity{},
	}
	writeNormalizedWorkloadLineageForTest(t, runDir, identities)
	executor := builtinExecutorContractForTest(t, normalizedSuiteExecutorID)

	writeJSONLinesForTest(t, filepath.Join(runDir, "cases.jsonl"), caseB, caseA)
	writeJSONLinesForTest(t, filepath.Join(runDir, "grading-cases.jsonl"), gradingA, gradingB)
	err := validateNormalizedWorkloadFromLineage(runDir, manifest, executor)
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "installed case") {
		t.Fatalf("worker-selected visible order error=%v, want frozen-order rejection", err)
	}

	writeJSONLinesForTest(t, filepath.Join(runDir, "cases.jsonl"), caseA, caseB)
	writeJSONLinesForTest(t, filepath.Join(runDir, "grading-cases.jsonl"), gradingB, gradingA)
	err = validateNormalizedWorkloadFromLineage(runDir, manifest, executor)
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "installed label") {
		t.Fatalf("worker-selected grading order error=%v, want frozen-order rejection", err)
	}
}

func TestNormalizedLiveWorkloadBindingRejectsUnmappedInstalledArmLabels(t *testing.T) {
	for _, test := range []struct {
		name  string
		arms  []ModelArm
		match string
	}{
		{
			name:  "unknown",
			arms:  []ModelArm{{ID: "runtime-other", Model: "other-model"}},
			match: "does not identify a frozen Mixture arm",
		},
		{
			name: "ambiguous",
			arms: []ModelArm{
				{ID: "arm-a", Model: "model-one"},
				{ID: "runtime-two", Model: "arm-a"},
			},
			match: "ambiguous in the frozen Mixture",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "trusted-routing", importedSuiteFixtureOptions{
				adapterID: "routerarena", trackIDs: []TrackID{"routing"},
				gradingCaseOverrides: map[string]any{"expected_route": "arm-a"},
			})
			runDir := filepath.Join(root, "runs", "normalized-live-"+test.name)
			if err := os.MkdirAll(runDir, 0o700); err != nil {
				t.Fatal(err)
			}
			manifest, identities, visible, grading := normalizedWorkloadTestCase(ModeLive, revision)
			manifest.Target.Mixture.ModelArms = test.arms
			writeJSONLinesForTest(t, filepath.Join(runDir, "cases.jsonl"), visible)
			writeJSONLinesForTest(t, filepath.Join(runDir, "grading-cases.jsonl"), grading)
			writeNormalizedWorkloadLineageForTest(t, runDir, identities)

			err := validateNormalizedWorkloadFromLineage(
				runDir, manifest, builtinExecutorContractForTest(t, normalizedSuiteLiveExecutorID),
			)
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("%s installed route label error=%v, want fail-closed rejection containing %q", test.name, err, test.match)
			}
		})
	}
}

func TestNormalizedArmLabelRequiresReplayOrFrozenLiveMixture(t *testing.T) {
	source := "arm-a"
	if _, err := normalizedExpectedArmID(RunManifest{Mode: Mode("invalid")}, "revision", &source); err == nil ||
		!strings.Contains(err.Error(), "require replay or a frozen live Mixture") {
		t.Fatalf("non-live normalized arm label error=%v, want fail-closed mode rejection", err)
	}
}

func normalizedWorkloadTestCase(
	mode Mode,
	revision string,
) (RunManifest, normalizedSuiteIdentityLineage, map[string]any, map[string]any) {
	caseID := normalizedOpaqueID("case", revision, "case", "case-1")
	manifest := RunManifest{
		Mode: mode, SuiteIDs: []string{"trusted-routing"},
		SuiteRevisions: map[string]string{"trusted-routing": revision},
		TrackIDs:       []TrackID{"routing"}, SampleLimit: 1, Seed: 19,
	}
	armIdentities := []normalizedLineageIdentity{}
	expectedArmID := normalizedOpaqueID("arm", revision, "arm", "arm-a")
	if mode == ModeReplay {
		manifest.Target = ManifestTarget{ID: "benchmark-source", Kind: "normalized-benchmark-source"}
		manifest.SuiteExecutors = map[string]string{"trusted-routing": normalizedSuiteExecutorID}
		armIdentities = append(armIdentities, normalizedLineageIdentity{
			SuiteID: "trusted-routing", OpaqueID: expectedArmID, SourceID: "arm-a",
		})
	} else {
		manifest.SuiteExecutors = map[string]string{"trusted-routing": normalizedSuiteLiveExecutorID}
		manifest.Target.Mixture = &ManifestMixture{ModelArms: []ModelArm{{ID: "runtime-arm-a", Model: "arm-a"}}}
		expectedArmID = "runtime-arm-a"
	}
	identities := normalizedSuiteIdentityLineage{
		SchemaVersion:  normalizedSuiteSchemaVersion,
		SuiteRevisions: manifest.SuiteRevisions,
		CaseIdentities: []normalizedLineageIdentity{{
			SuiteID: "trusted-routing", OpaqueID: caseID, SourceID: "case-1",
		}},
		ArmIdentities: armIdentities, ActionIdentities: []normalizedLineageIdentity{},
	}
	visible, grading := normalizedReplayRows(revision, "server-owned-answer")
	grading["expected_route"] = expectedArmID
	grading["preferred_arm_id"] = expectedArmID
	return manifest, identities, visible, grading
}

func normalizedReplayRows(revision, answer string) (map[string]any, map[string]any) {
	caseID := normalizedOpaqueID("case", revision, "case", "case-1")
	return map[string]any{
			"schema_version": SchemaVersion,
			"id":             caseID, "track_ids": []TrackID{"routing"},
			"messages": []map[string]any{{"role": "user", "content": "private"}},
			"modality": "text", "tags": []string{},
		}, map[string]any{
			"schema_version": SchemaVersion, "case_id": caseID,
			"expected_answer": answer, "expected_tools": []string{}, "weight": 1.0,
		}
}

func writeNormalizedWorkloadLineageForTest(
	t *testing.T,
	runDir string,
	identities normalizedSuiteIdentityLineage,
) {
	t.Helper()
	if err := writeJSONAtomic(filepath.Join(runDir, "lineage.json"), map[string]any{
		"schema_version":              SchemaVersion,
		"resolved_snapshot":           testResolvedLineageSnapshot("sha256:" + strings.Repeat("0", 64)),
		"normalized_suite_identities": identities,
	}); err != nil {
		t.Fatal(err)
	}
}

func cloneAnyMap(source map[string]any) map[string]any {
	cloned := make(map[string]any, len(source))
	for key, value := range source {
		cloned[key] = value
	}
	return cloned
}
