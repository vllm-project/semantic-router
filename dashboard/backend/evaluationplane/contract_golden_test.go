package evaluationplane

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func goldenFile(t *testing.T, name string) string {
	t.Helper()
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve golden test source path")
	}
	return filepath.Join(filepath.Dir(source), "..", "..", "..", "src", "vllm-sr", "tests", "fixtures", "evaluation", name)
}

func decodeGoldenStrict(t *testing.T, name string, destination any) {
	t.Helper()
	file, err := os.Open(goldenFile(t, name))
	if err != nil {
		t.Fatalf("open cross-package golden %s: %v", name, err)
	}
	t.Cleanup(func() {
		if err := file.Close(); err != nil {
			t.Errorf("close cross-package golden %s: %v", name, err)
		}
	})
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		t.Fatalf("strict decode cross-package golden %s: %v", name, err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		t.Fatalf("cross-package golden %s trailing data: %v", name, err)
	}
}

func TestPythonWorkerDraftAndInputGoldenContractsDecodeStrictly(t *testing.T) {
	var catalog Catalog
	decodeGoldenStrict(t, "catalog.json", &catalog)
	var manifest RunManifest
	decodeGoldenStrict(t, "manifest.json", &manifest)
	var liveManifest RunManifest
	decodeGoldenStrict(t, "live-manifest.json", &liveManifest)
	var report workerReport
	decodeGoldenStrict(t, "worker-report-draft.json", &report)

	for name, version := range map[string]string{
		"catalog": catalog.SchemaVersion, "manifest": manifest.SchemaVersion,
		"manifest target": manifest.Target.SchemaVersion, "worker draft": report.SchemaVersion,
		"live manifest": liveManifest.SchemaVersion, "live manifest target": liveManifest.Target.SchemaVersion,
		"worker run": report.Run.SchemaVersion, "worker provenance": report.Provenance.SchemaVersion,
	} {
		if version != SchemaVersion {
			t.Fatalf("%s schema_version=%q, want %q", name, version, SchemaVersion)
		}
	}
	if report.Run.ID != manifest.RunID {
		t.Fatalf("golden worker draft run=%q differs from manifest=%q", report.Run.ID, manifest.RunID)
	}
	if !digestPattern.MatchString(manifest.PolicySnapshotDigest) ||
		!digestPattern.MatchString(liveManifest.PolicySnapshotDigest) {
		t.Fatalf("golden manifests lack required policy snapshot identity: replay=%q live=%q",
			manifest.PolicySnapshotDigest, liveManifest.PolicySnapshotDigest)
	}
	if catalog.GateContractVersion != GateContractVersion || manifest.ChangeProfile != report.Run.ChangeProfile ||
		!validChangeProfile(manifest.ChangeProfile) || !validChangeProfile(liveManifest.ChangeProfile) {
		t.Fatalf("golden gate/profile contract drift: catalog=%q manifest=%q live=%q report=%q",
			catalog.GateContractVersion, manifest.ChangeProfile, liveManifest.ChangeProfile, report.Run.ChangeProfile)
	}
	if liveManifest.Target.Mixture == nil {
		t.Fatal("live manifest omits its frozen mixture")
	}
	if err := validateMixtureContract(liveManifest.Target.Mixture); err != nil {
		t.Fatalf("live manifest mixture contract: %v", err)
	}
	if err := validateTargetContract(
		liveManifest.Target.RouterAPIKey,
		liveManifest.Target.EnvoyAPIKey,
		liveManifest.Target.Mixture.ModelArms,
		liveManifest.Target.BackendTopologyDigest,
	); err != nil {
		t.Fatalf("live manifest target contract: %v", err)
	}
	if liveManifest.Mode != ModeLive || !digestPattern.MatchString(liveManifest.Target.BackendTopologyDigest) || len(liveManifest.Target.Mixture.ModelArms) < 2 {
		t.Fatalf("live manifest lacks a runnable topology snapshot: %+v", liveManifest.Target)
	}
	for name, goldenManifest := range map[string]RunManifest{"replay": manifest, "live": liveManifest} {
		recomputed, err := manifestSemanticDigest(goldenManifest)
		if err != nil {
			t.Errorf("compute %s golden manifest digest: %v", name, err)
			continue
		}
		if recomputed != goldenManifest.ManifestDigest {
			t.Errorf("%s golden manifest_digest=%q, server-owned digest=%q", name, goldenManifest.ManifestDigest, recomputed)
		}
	}
	if liveManifest.Target.Mixture.ModelArms[0].InputCostPerMillionTokensUSD != 0 ||
		liveManifest.Target.Mixture.ModelArms[0].OutputCostPerMillionTokensUSD != 1 ||
		liveManifest.Target.Mixture.ModelArms[1].InputCostPerMillionTokensUSD != 1e-7 ||
		!math.Signbit(liveManifest.Target.Mixture.ModelArms[1].OutputCostPerMillionTokensUSD) ||
		liveManifest.Target.Mixture.ModelArms[1].Model != "公共-strong-模型" || liveManifest.CreatedAt.Nanosecond() != 123456000 {
		t.Fatalf("live golden lost cross-language numeric/time/unicode cases: %+v", liveManifest)
	}
	for _, gate := range report.Gates {
		if err := validateReportGate(gate, report.Run.ChangeProfile); err != nil {
			t.Fatalf("golden gate %q: %v", gate.ID, err)
		}
	}
}

func TestBuiltinCatalogMatchesPythonGoldenDefinitions(t *testing.T) {
	var golden Catalog
	decodeGoldenStrict(t, "catalog.json", &golden)
	registry, err := NewRegistry("", "")
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	actual := registry.Catalog()
	if actual.GateContractVersion != golden.GateContractVersion ||
		!reflect.DeepEqual(actual.ChangeProfiles, golden.ChangeProfiles) {
		t.Fatalf("Go/Python gate catalog drift:\nGo:     %q %+v\nPython: %q %+v",
			actual.GateContractVersion, actual.ChangeProfiles, golden.GateContractVersion, golden.ChangeProfiles)
	}
	if len(actual.Tracks) != len(golden.Tracks) || len(actual.Suites) != len(golden.Suites) {
		t.Fatalf("Go/Python catalog size drift: tracks=%d/%d suites=%d/%d", len(actual.Tracks), len(golden.Tracks), len(actual.Suites), len(golden.Suites))
	}

	goldenTracks := make(map[TrackID]CatalogTrack, len(golden.Tracks))
	for _, track := range golden.Tracks {
		goldenTracks[track.ID] = track
	}
	for _, track := range actual.Tracks {
		if !reflect.DeepEqual(track, goldenTracks[track.ID]) {
			t.Fatalf("Go/Python track drift for %s:\nGo:     %+v\nPython: %+v", track.ID, track, goldenTracks[track.ID])
		}
	}
	for index := range actual.Tracks {
		if actual.Tracks[index].ID != golden.Tracks[index].ID {
			t.Fatalf("Go/Python track order drift at %d: Go=%s Python=%s", index, actual.Tracks[index].ID, golden.Tracks[index].ID)
		}
	}

	goldenSuites := make(map[string]CatalogSuite, len(golden.Suites))
	for _, suite := range golden.Suites {
		if len(suite.Tags) == 0 {
			suite.Tags = nil
		}
		goldenSuites[suite.ID] = suite
	}
	for _, suite := range actual.Suites {
		if len(suite.Tags) == 0 {
			suite.Tags = nil
		}
		if !reflect.DeepEqual(suite, goldenSuites[suite.ID]) {
			t.Fatalf("Go/Python suite drift for %s:\nGo:     %+v\nPython: %+v", suite.ID, suite, goldenSuites[suite.ID])
		}
	}
	for index := range actual.Suites {
		if actual.Suites[index].ID != golden.Suites[index].ID {
			t.Fatalf("Go/Python suite order drift at %d: Go=%s Python=%s", index, actual.Suites[index].ID, golden.Suites[index].ID)
		}
	}
	actualTargetsJSON, actualTargetsErr := json.Marshal(actual.Targets)
	goldenTargetsJSON, goldenTargetsErr := json.Marshal(golden.Targets)
	if actualTargetsErr != nil || goldenTargetsErr != nil || string(actualTargetsJSON) != string(goldenTargetsJSON) {
		t.Fatalf("Go/Python target drift:\nGo:     %s\nPython: %s", actualTargetsJSON, goldenTargetsJSON)
	}
}

func TestCapabilityMatrixMatchesPythonCurrentContract(t *testing.T) {
	var matrix struct {
		SchemaVersion string `json:"schema_version"`
		Cases         []struct {
			Name           string         `json:"name"`
			Valid          bool           `json:"valid"`
			Target         ManifestTarget `json:"target"`
			ExpectedTracks []TrackID      `json:"expected_tracks"`
		} `json:"cases"`
	}
	decodeGoldenStrict(t, "capability-matrix.json", &matrix)
	if matrix.SchemaVersion != SchemaVersion || len(matrix.Cases) == 0 {
		t.Fatalf("invalid capability matrix header: %+v", matrix)
	}
	for _, test := range matrix.Cases {
		t.Run(test.Name, func(t *testing.T) {
			err := validateCapabilityMatrixTarget(test.Target)
			if test.Valid && err != nil {
				t.Fatalf("Go rejected Python-valid target: %v", err)
			}
			if !test.Valid && err == nil {
				t.Fatal("Go accepted Python-invalid target")
			}
			if !test.Valid {
				return
			}
			tracks := availableTargetTracks(targetDefinition{
				Public:       CatalogTarget{ID: test.Target.ID, Kind: test.Target.Kind, Mixture: catalogMixtureFromManifest(test.Target.Mixture)},
				Contract:     targetContract{ExecutionProfile: targetProfileRuntime, PolicySnapshot: policySnapshotRuntime, TrackRequirements: runtimeTrackRequirements()},
				RouterAPIURL: test.Target.RouterAPIURL, EnvoyURL: test.Target.EnvoyURL,
				RouterAPIKey: test.Target.RouterAPIKey, EnvoyAPIKey: test.Target.EnvoyAPIKey,
				AgentTaskLedger:     test.Target.AgentTaskLedger,
				FaultRecoveryLedger: test.Target.FaultRecoveryLedger,
				HardPolicyLedger:    test.Target.HardPolicyLedger, ProductionExperimentLedger: test.Target.ProductionExperimentLedger,
				Mixture: test.Target.Mixture, BackendTopologyDigest: test.Target.BackendTopologyDigest,
			})
			if !reflect.DeepEqual(tracks, test.ExpectedTracks) {
				t.Fatalf("runtime tracks=%v, want %v", tracks, test.ExpectedTracks)
			}
		})
	}
}

func validateCapabilityMatrixTarget(target ManifestTarget) error {
	if target.SchemaVersion != SchemaVersion || !portableIDPattern.MatchString(target.ID) ||
		strings.TrimSpace(target.Kind) == "" || len(target.Kind) > 64 {
		return fmt.Errorf("invalid target identity")
	}
	for _, rawURL := range []string{target.RouterAPIURL, target.EnvoyURL} {
		if err := validateServerOrigin(rawURL); err != nil {
			return fmt.Errorf("invalid target URL")
		}
	}
	if err := validateEndpointCredentialBindings(
		target.RouterAPIURL, target.EnvoyURL, target.RouterAPIKey, target.EnvoyAPIKey,
	); err != nil {
		return err
	}
	if err := validateMixtureContract(target.Mixture); err != nil {
		return err
	}
	if err := validateServiceEndpoint("hard_policy_ledger", target.HardPolicyLedger); err != nil {
		return err
	}
	if err := validateServiceEndpoint("fault_recovery_ledger", target.FaultRecoveryLedger); err != nil {
		return err
	}
	if err := validateServiceEndpoint("agent_task_ledger", target.AgentTaskLedger); err != nil {
		return err
	}
	if err := validateServiceEndpoint("production_experiment_ledger", target.ProductionExperimentLedger); err != nil {
		return err
	}
	arms := []ModelArm(nil)
	if target.Mixture != nil {
		arms = target.Mixture.ModelArms
	}
	return validateTargetContract(target.RouterAPIKey, target.EnvoyAPIKey, arms, target.BackendTopologyDigest)
}
