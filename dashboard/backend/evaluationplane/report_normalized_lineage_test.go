package evaluationplane

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNormalizedLiveLineageBindsOnlyTheSelectedInstalledWorkload(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "target.routing")
	if err := os.WriteFile(
		filepath.Join(service.registrySource.suiteStorePath, "index", "unrelated.json"),
		[]byte("corrupt unrelated suite"),
		0o600,
	); err != nil {
		t.Fatalf("write unrelated corrupt suite: %v", err)
	}

	runDir := filepath.Join(root, "runs", "target-lineage")
	if err := os.MkdirAll(runDir, 0o700); err != nil {
		t.Fatalf("create run dir: %v", err)
	}
	caseID := normalizedOpaqueID("case", revision, "case", "case-1")
	visible := []byte(`{"schema_version":"evaluation.v1","id":"` + caseID + `"}` + "\n")
	if err := os.WriteFile(filepath.Join(runDir, "cases.jsonl"), visible, 0o600); err != nil {
		t.Fatalf("write visible cases: %v", err)
	}
	manifest := RunManifest{
		Mode:           ModeLive,
		TrackIDs:       []TrackID{"routing"},
		SuiteIDs:       []string{"target.routing"},
		SuiteRevisions: map[string]string{"target.routing": revision},
		SuiteExecutors: map[string]string{"target.routing": normalizedSuiteLiveExecutorID},
		SampleLimit:    1,
		Seed:           19,
	}
	identities := normalizedSuiteIdentityLineage{
		SchemaVersion:  normalizedSuiteSchemaVersion,
		SuiteRevisions: map[string]string{"target.routing": revision},
		CaseIdentities: []normalizedLineageIdentity{{
			SuiteID: "target.routing", OpaqueID: caseID, SourceID: "case-1",
		}},
		ArmIdentities: []normalizedLineageIdentity{}, ActionIdentities: []normalizedLineageIdentity{},
	}
	raw, err := json.Marshal(identities)
	if err != nil {
		t.Fatalf("marshal identities: %v", err)
	}
	validated, err := validateNormalizedSuiteLineageForTest(t, runDir, manifest, raw)
	if err != nil {
		t.Fatalf("validate normalized target lineage: %v", err)
	}
	if validated == nil || len(validated.CaseIdentities) != 1 || validated.CaseIdentities[0].OpaqueID != caseID {
		t.Fatalf("validated identities=%+v", validated)
	}

	replayManifest := manifest
	replayManifest.Mode = ModeReplay
	replayManifest.SuiteExecutors = map[string]string{"target.routing": normalizedSuiteExecutorID}
	replayIdentities := identities
	replayIdentities.ArmIdentities = []normalizedLineageIdentity{{
		SuiteID:  "target.routing",
		OpaqueID: normalizedOpaqueID("arm", revision, "arm", "arm-a"),
		SourceID: "arm-a",
	}}
	replayRaw, _ := json.Marshal(replayIdentities)
	if _, err := validateNormalizedSuiteLineageForTest(t, runDir, replayManifest, replayRaw); err != nil {
		t.Fatalf("validate normalized replay source identities: %v", err)
	}
	missingReplayArm, _ := json.Marshal(identities)
	if _, err := validateNormalizedSuiteLineageForTest(t, runDir, replayManifest, missingReplayArm); !errors.Is(err, ErrInvalid) {
		t.Fatalf("missing replay source arm error=%v, want ErrInvalid", err)
	}

	withHistoricalArm := identities
	withHistoricalArm.ArmIdentities = []normalizedLineageIdentity{{
		SuiteID:  "target.routing",
		OpaqueID: normalizedOpaqueID("arm", revision, "arm", "arm-a"),
		SourceID: "arm-a",
	}}
	raw, _ = json.Marshal(withHistoricalArm)
	if _, err := validateNormalizedSuiteLineageForTest(t, runDir, manifest, raw); !errors.Is(err, ErrInvalid) {
		t.Fatalf("historical arm identity error=%v, want ErrInvalid", err)
	}

	tampered := identities
	tampered.CaseIdentities = append([]normalizedLineageIdentity(nil), identities.CaseIdentities...)
	tampered.CaseIdentities[0].OpaqueID = normalizedOpaqueID("case", revision, "case", "other-case")
	raw, _ = json.Marshal(tampered)
	if _, err := validateNormalizedSuiteLineageForTest(t, runDir, manifest, raw); !errors.Is(err, ErrInvalid) {
		t.Fatalf("tampered case identity error=%v, want ErrInvalid", err)
	}
}

func TestNormalizedLineageRequiresItsInstalledManifestRevision(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "target.routing")
	runDir := filepath.Join(root, "runs", "target-lineage-revision")
	if err := os.MkdirAll(runDir, 0o700); err != nil {
		t.Fatalf("create run dir: %v", err)
	}
	caseID := normalizedOpaqueID("case", revision, "case", "case-1")
	if err := os.WriteFile(
		filepath.Join(runDir, "cases.jsonl"),
		[]byte(`{"schema_version":"evaluation.v1","id":"`+caseID+`"}`+"\n"),
		0o600,
	); err != nil {
		t.Fatalf("write visible cases: %v", err)
	}
	manifest := RunManifest{
		Mode:           ModeReplay,
		TrackIDs:       []TrackID{"routing"},
		SuiteIDs:       []string{"target.routing"},
		SuiteRevisions: map[string]string{"target.routing": "sha256:" + strings.Repeat("0", 64)},
		SuiteExecutors: map[string]string{"target.routing": normalizedSuiteExecutorID},
		SampleLimit:    1,
		Seed:           19,
	}
	identities := normalizedSuiteIdentityLineage{
		SchemaVersion:  normalizedSuiteSchemaVersion,
		SuiteRevisions: manifest.SuiteRevisions,
		CaseIdentities: []normalizedLineageIdentity{{
			SuiteID: "target.routing", OpaqueID: caseID, SourceID: "case-1",
		}},
		ArmIdentities: []normalizedLineageIdentity{}, ActionIdentities: []normalizedLineageIdentity{},
	}
	raw, _ := json.Marshal(identities)
	if _, err := validateNormalizedSuiteLineageForTest(t, runDir, manifest, raw); !errors.Is(err, ErrInvalid) {
		t.Fatalf("uninstalled revision error=%v, want ErrInvalid", err)
	}
}
