package evaluationplane

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveImportedSuiteEvidenceIsE0AndQualifiesNoGate(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	firstRevision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "import-a")
	secondRevision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "import-b")
	manifest := RunManifest{
		Mode: ModeReplay, Target: ManifestTarget{ID: "benchmark-source", Kind: "normalized-benchmark-source"},
		SuiteIDs:       []string{"import-a", "import-b"},
		SuiteRevisions: map[string]string{"import-a": firstRevision, "import-b": secondRevision},
		SuiteExecutors: map[string]string{"import-a": normalizedSuiteExecutorID, "import-b": normalizedSuiteExecutorID},
	}

	qualification, err := resolveSuiteGateQualificationForTest(t, service.registrySource.suiteStorePath, manifest)
	if err != nil {
		t.Fatalf("resolve imported suite provenance: %v", err)
	}
	if qualification.qualifies("G4") || qualification.qualifies("G6") || len(qualification.commonGateIDs) != 0 {
		t.Fatalf("imported suites unexpectedly qualified gates: %v", qualification.commonGateIDs)
	}
	for suiteID, levels := range qualification.suiteTrackLevels {
		for trackID, level := range levels {
			if level != "E0" {
				t.Fatalf("suite=%s track=%s level=%s, want E0", suiteID, trackID, level)
			}
		}
	}
}

func TestResolveImportedSuiteRejectsMissingMixedChangedAndCorruptEvidence(t *testing.T) {
	t.Run("missing private suite", func(t *testing.T) {
		service, _ := newTestService(t, &controlledProcess{}, 1)
		manifest := normalizedReplayManifest("missing", "sha256:"+strings.Repeat("a", 64))
		_, err := resolveSuiteGateQualificationForTest(t, service.registrySource.suiteStorePath, manifest)
		if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "index is unavailable") {
			t.Fatalf("missing suite error=%v", err)
		}
	})

	t.Run("mixed executor families", func(t *testing.T) {
		service, _ := newTestService(t, &controlledProcess{}, 1)
		revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "imported")
		manifest := normalizedReplayManifest("imported", revision)
		manifest.SuiteIDs = append(manifest.SuiteIDs, "builtin")
		manifest.SuiteRevisions["builtin"] = "builtin-revision"
		manifest.SuiteExecutors["builtin"] = "fixture-replay.v1"
		_, err := resolveSuiteGateQualification(
			service.registrySource.suiteStorePath,
			manifest,
			builtinExecutorContractForTest(t, normalizedSuiteExecutorID),
		)
		if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "resolved execution contract") {
			t.Fatalf("mixed suites error=%v", err)
		}
	})

	t.Run("frozen revision changed", func(t *testing.T) {
		service, _ := newTestService(t, &controlledProcess{}, 1)
		writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "imported")
		manifest := normalizedReplayManifest("imported", "sha256:"+strings.Repeat("b", 64))
		_, err := resolveSuiteGateQualificationForTest(t, service.registrySource.suiteStorePath, manifest)
		if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "frozen revision") {
			t.Fatalf("changed revision error=%v", err)
		}
	})

	t.Run("manifest content tampered", func(t *testing.T) {
		service, _ := newTestService(t, &controlledProcess{}, 1)
		revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "imported")
		manifest := normalizedReplayManifest("imported", revision)
		indexBytes, readErr := os.ReadFile(filepath.Join(service.registrySource.suiteStorePath, "index", "imported.json"))
		if readErr != nil {
			t.Fatal(readErr)
		}
		var index suiteIndexRecord
		if err := decodeExactJSON(indexBytes, &index); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(service.registrySource.suiteStorePath, "manifests", "sha256", strings.TrimPrefix(index.ManifestDigest, "sha256:"))
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			t.Fatal(readErr)
		}
		if err := os.WriteFile(path, append(data, ' '), 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := resolveSuiteGateQualificationForTest(t, service.registrySource.suiteStorePath, manifest)
		if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "corrupt") {
			t.Fatalf("tampered manifest error=%v", err)
		}
	})
}

func TestResolveImportedSuiteRejectsForgedPromotionAndNativeRunClaims(t *testing.T) {
	tests := []struct {
		name    string
		options importedSuiteFixtureOptions
	}{
		{
			name: "non E0 level",
			options: importedSuiteFixtureOptions{
				adapterID: "routerarena", trackIDs: []TrackID{"routing"}, evidenceLevel: "E4",
			},
		},
		{
			name: "native execution self assertion",
			options: importedSuiteFixtureOptions{
				adapterID: "routerarena", trackIDs: []TrackID{"routing"}, nativeRunAttested: true,
			},
		},
		{
			name: "promotion self assertion",
			options: importedSuiteFixtureOptions{
				adapterID: "routerarena", trackIDs: []TrackID{"routing"}, promotionEligible: true,
			},
		},
		{
			name: "forged parser verification",
			options: importedSuiteFixtureOptions{
				adapterID: "routerarena", trackIDs: []TrackID{"routing"}, parserVerified: true,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service, _ := newTestService(t, &controlledProcess{}, 1)
			revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "forged", test.options)
			_, err := resolveSuiteGateQualificationForTest(t, service.registrySource.suiteStorePath, normalizedReplayManifest("forged", revision))
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("forged import provenance error=%v", err)
			}
		})
	}
}

func TestResolveImportedSuiteRejectsSourceAndTrackForgeries(t *testing.T) {
	for _, test := range []struct {
		name    string
		options importedSuiteFixtureOptions
	}{
		{
			name: "source pin",
			options: importedSuiteFixtureOptions{
				adapterID: "routerarena", sourceRevisionOverride: strings.Repeat("b", 40), trackIDs: []TrackID{"routing"},
			},
		},
		{
			name: "adapter track",
			options: importedSuiteFixtureOptions{
				adapterID: "routerarena", trackIDs: []TrackID{"safety"},
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			service, _ := newTestService(t, &controlledProcess{}, 1)
			revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "forged", test.options)
			_, err := resolveSuiteGateQualificationForTest(t, service.registrySource.suiteStorePath, normalizedReplayManifest("forged", revision))
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("source/track forgery error=%v", err)
			}
		})
	}
}

func TestImportedSuiteNeverPromotesLiveExecution(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "live-suite")
	manifest := normalizedReplayManifest("live-suite", revision)
	manifest.Mode = ModeLive
	manifest.Target = ManifestTarget{ID: "mom-default", Kind: "mixture-of-models"}
	manifest.SuiteExecutors["live-suite"] = normalizedSuiteLiveExecutorID
	qualification, err := resolveSuiteGateQualificationForTest(t, service.registrySource.suiteStorePath, manifest)
	if err != nil {
		t.Fatal(err)
	}
	if qualification.qualifies("G4") {
		t.Fatal("live execution inherited imported replay qualification")
	}
}

func normalizedReplayManifest(suiteID, revision string) RunManifest {
	return RunManifest{
		Mode: ModeReplay, Target: ManifestTarget{ID: "benchmark-source", Kind: "normalized-benchmark-source"}, SuiteIDs: []string{suiteID},
		SuiteRevisions: map[string]string{suiteID: revision},
		SuiteExecutors: map[string]string{suiteID: normalizedSuiteExecutorID},
	}
}
