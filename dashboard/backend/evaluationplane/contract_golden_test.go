package evaluationplane

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

func goldenFile(t *testing.T, name string) string {
	t.Helper()
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve golden test source path")
	}
	return filepath.Join(filepath.Dir(source), "..", "..", "..", "src", "vllm-sr", "cli", "evaluation", "golden", name)
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

func TestPythonGoldenContractsDecodeStrictly(t *testing.T) {
	var catalog Catalog
	decodeGoldenStrict(t, "catalog.json", &catalog)
	var manifest RunManifest
	decodeGoldenStrict(t, "manifest.json", &manifest)
	var liveManifest RunManifest
	decodeGoldenStrict(t, "live-manifest.json", &liveManifest)
	var report Report
	decodeGoldenStrict(t, "report.json", &report)

	for name, version := range map[string]string{
		"catalog": catalog.SchemaVersion, "manifest": manifest.SchemaVersion,
		"manifest target": manifest.Target.SchemaVersion, "report": report.SchemaVersion,
		"live manifest": liveManifest.SchemaVersion, "live manifest target": liveManifest.Target.SchemaVersion,
		"report run": report.Run.SchemaVersion, "report provenance": report.Provenance.SchemaVersion,
	} {
		if version != SchemaVersion {
			t.Fatalf("%s schema_version=%q, want %q", name, version, SchemaVersion)
		}
	}
	if report.Run.ID != manifest.RunID {
		t.Fatalf("golden report run=%q differs from manifest=%q", report.Run.ID, manifest.RunID)
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
	if err := validateTargetContract(
		liveManifest.Target.RouterAPIKey,
		liveManifest.Target.EnvoyAPIKey,
		liveManifest.Target.ModelArms,
		liveManifest.Target.BackendTopologyDigest,
	); err != nil {
		t.Fatalf("live manifest target contract: %v", err)
	}
	if liveManifest.Mode != ModeLive || !digestPattern.MatchString(liveManifest.Target.BackendTopologyDigest) || len(liveManifest.Target.ModelArms) < 2 {
		t.Fatalf("live manifest lacks a runnable topology snapshot: %+v", liveManifest.Target)
	}
	for name, goldenManifest := range map[string]RunManifest{"replay": manifest, "live": liveManifest} {
		recomputed, err := manifestSemanticDigest(goldenManifest)
		if err != nil {
			t.Fatalf("compute %s golden manifest digest: %v", name, err)
		}
		if recomputed != goldenManifest.ManifestDigest {
			t.Fatalf("%s golden manifest_digest=%q, server-owned digest=%q", name, goldenManifest.ManifestDigest, recomputed)
		}
	}
	if liveManifest.Target.ModelArms[0].InputCostPerMillionTokensUSD != 0 ||
		liveManifest.Target.ModelArms[0].OutputCostPerMillionTokensUSD != 1 ||
		liveManifest.Target.ModelArms[1].InputCostPerMillionTokensUSD != 1e-7 ||
		!math.Signbit(liveManifest.Target.ModelArms[1].OutputCostPerMillionTokensUSD) ||
		liveManifest.Target.ModelArms[1].Model != "公共-strong-模型" || liveManifest.CreatedAt.Nanosecond() != 123456000 {
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
	if !reflect.DeepEqual(actual.Targets, golden.Targets) {
		t.Fatalf("Go/Python target drift:\nGo:     %+v\nPython: %+v", actual.Targets, golden.Targets)
	}
}
